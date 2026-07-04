#!/usr/bin/env python3
"""Regenerate the manuscript figures from the solver.

Produces (into writing/manuscript/Figures/):
  * wall_seal_regime.pdf       (Fig. 3) -- the self-sealing attractor
  * condensation_profiles.pdf  -- lip-concentrated, phase-dependent condensation
  * peak_predictor.pdf         (Fig. 4) -- the two mass-flux peaks
  * closing_massflux.pdf       -- diurnal flux as a crack seals
  * phase_overlay.pdf          -- model peaks vs observed peak phases
  * attractor_convergence.pdf  -- convergence to overflow from any initial width

The expensive solver sweeps are cached (``--cache``); re-runs that only change
styling, axis limits, or the observed red dot replot instantly. Use
``--recompute`` to force the sweeps.

Run from the package directory:  python make_figures.py [-v] [--recompute]
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enceladus_plume.config import load_config
from enceladus_plume.liquid_dynamics.solver import (
    liquid_dynamics, compute_overflow_rate, buffer_overflow,
)
from enceladus_plume.utils import build_width_series
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.peaks import predict_peaks
from enceladus_plume.wall_geometry import evolve_geometry_coupled
from enceladus_plume.physics import (
    find_evap_surface, vapor_pressure, evaporation_rate_simple,
)
from enceladus_plume.wall_budget import _RG, _SIGMA, _KT, _TE

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(_HERE, "..", "writing", "manuscript", "Figures"))

CRACK_LENGTH = 5.0e5          # m, summed tiger-stripe length
TB = 272.6
# Approximate observed scenario (red dot, Fig. 4b). Estimates -- refine with the
# exact slab-density main/secondary ratio; the absolute secondary-peak emission
# is not directly measured. Axis limits below are set to 0.5x--2x of these.
OBS_WIDENING_KGS = 150.0      # widening (= observed secondary) peak emission, kg/s
OBS_RATIO = 2.0              # approach/widening (= observed main/secondary)
# Observed diurnal peak phases (mean anomaly), from the Cassini brightness /
# slab-density profiles (Ingersoll et al. 2020; Hedman et al. 2013). Only the
# peak *phases* are used -- no digitized observed flux curve is plotted.
OBS_MAIN_MA = (160.0, 215.0)  # main (brightness) peak phase band
OBS_SEC_MA = (30.0, 70.0)     # secondary peak phase band


def _cfg():
    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 300.0
    cfg.liquid_dynamics.npts_velocity = 150
    return cfg


# --------------------------------------------------------------------------
# Compute (cached)
# --------------------------------------------------------------------------
def _flux_curve(cfg, L, dw, we, lookup):
    cfg.physical.equilibrium_depth = L
    D = L / 10.0
    P = cfg.physical.orbital_period
    f_evap = cfg.physical.latent_heat_fusion / (cfg.physical.latent_heat + cfg.physical.latent_heat_fusion)
    rho_w = cfg.physical.liquid_density
    t_in = np.arange(100, P + 1, 200.0)
    R = 1.0 + dw / we
    w_in = build_width_series(t_in, R, we, orbital_period=P, forcing_model="single-cosine")
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
    _, phi, _, _ = lookup.query_vectorized(TB, np.clip(D - h, lookup.depth.min(), lookup.depth.max()),
                                           np.clip(w, lookup.delta.min(), lookup.delta.max()))
    gas = np.nan_to_num(phi * w)
    ov = np.nan_to_num(f_evap * rho_w * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg))
    ov = buffer_overflow(t, ov, cfg.liquid_dynamics.overflow_tau)  # surface reservoir
    flux = gas + ov
    m = t >= (t[-1] - P)
    o = np.argsort(t[m] - t[m][0])
    return (t[m][o] - t[m][o][0]) / P * 360.0, flux[m][o]


def _liquid_cycle(cfg, L, dw, we, lookup):
    """Last-cycle state (MA, h, w, r) for a crack of effective width ``we``."""
    cfg.physical.equilibrium_depth = L
    D = L / 10.0
    P = cfg.physical.orbital_period
    t_in = np.arange(100, P + 1, 200.0)
    R = 1.0 + dw / we
    w_in = build_width_series(t_in, R, we, orbital_period=P, forcing_model="single-cosine")
    w, h, t, _v = liquid_dynamics(w_in, t_in, L, cfg)
    _, _, _, r = lookup.query_vectorized(TB, np.clip(D - h, lookup.depth.min(), lookup.depth.max()),
                                         np.clip(w, lookup.delta.min(), lookup.delta.max()))
    m = t >= (t[-1] - P)
    o = np.argsort(t[m] - t[m][0])
    ma = (t[m][o] - t[m][o][0]) / P * 360.0
    return ma, h[m][o], w[m][o], np.nan_to_num(r[m][o], nan=1.0)


def _cond_profile(cfg, L, h, w, r, n_z=600):
    """Instantaneous condensation flux vs depth-below-surface at one phase.

    Mirrors the exposed-wall logic of ``wall_mass_budget``: the per-area
    condensation gain is ``-E(d)`` (lip-concentrated within the thermal skin),
    exposed between the water level and the vapour-limited choke height.
    Returns (depth-below-surface [m], condensation flux [kg m^-2 s^-1]).
    """
    Lv = cfg.physical.latent_heat
    D = L / 10.0
    surface = L + D
    zeta_w = L + h                      # water level
    # Refine the grid near the mouth where condensation concentrates.
    zeta_geo = surface - np.geomspace(1e-3, surface, n_z)
    zeta = np.unique(np.clip(np.concatenate([zeta_geo, [zeta_w, surface]]), 0.0, surface))
    d_top = np.clip(surface - zeta, 1e-10, None)
    E = np.array([find_evap_surface(TB, _TE, _SIGMA, d, _KT, Lv) for d in d_top])
    dep = np.clip(-E, 0.0, None)
    # Vapour-limited choke: column flux phi + integral of deposition from the
    # water surface upward reaches zero at the choke height.
    Phi = np.concatenate([[0.0], np.cumsum(0.5 * (dep[1:] + dep[:-1]) * np.diff(zeta))])
    ec = vapor_pressure(273.15) / np.sqrt(2.0 * np.pi * _RG)
    evap0 = evaporation_rate_simple(ec, Lv / _RG, TB)
    demand = (1.0 - r) * evap0 * w
    Phi_w = np.interp(zeta_w, zeta, Phi)
    target = Phi_w + demand
    zeta_c = surface if Phi[-1] <= target else float(np.interp(target, Phi, zeta))
    exposed = (zeta >= zeta_w) & (zeta <= zeta_c)
    return surface - zeta[exposed], dep[exposed]


def compute(lookup, cache, recompute=False):
    """Fill the figure cache. Only missing groups are computed unless
    ``recompute`` forces every group (the expensive Fig-4b scatter included)."""
    cfg = _cfg()
    have = {} if recompute or not os.path.exists(cache) else dict(np.load(cache))

    # --- Fig 3: seal sweep ---
    if "seal_dw" not in have:
        cfg.physical.equilibrium_depth = 5000.0
        seal_dw = np.array([0.003, 0.006, 0.012, 0.025])
        we_all, rise_all, wstar = [], [], []
        for dw in seal_dw:
            res = evolve_geometry_coupled(cfg, float(dw), n_e=11, w_eff_max=0.06, w_floor=2e-3)
            we_all.append(res.w_eff * 1e3)
            rise_all.append((res.water_max - res.L) / 500.0)
            wstar.append(res.w_eff_overflow * 1e3)
        have.update(seal_dw=seal_dw, we_all=np.array(we_all),
                    rise_all=np.array(rise_all), wstar=np.array(wstar))
        print("computed Fig 3 (seal sweep)")

    # --- Fig 4a: example curve ---
    if "MA" not in have:
        cfg.physical.equilibrium_depth = 5000.0
        MA, g = _flux_curve(cfg, 5000.0, 0.010, 0.007, lookup)
        pr = predict_peaks(cfg, 0.010, 0.007, lookup, Tb=TB)
        have.update(MA=MA, g=g, phi_w=pr.phi_widening, phi_a=pr.phi_approach)
        print("computed Fig 4a (example curve)")

    # --- Fig 4b: steady-state scatter (span shallow->deep to cover the observed) ---
    # The two peaks coexist over a small-swing / small-width window at shallow
    # depth (low ratio) and a broader window at depth (high ratio); sample both.
    if "xs" not in have:
        grid = [(L, dw, we)
                for L in (1500.0, 2000.0, 3000.0)
                for dw in (0.008, 0.010, 0.012, 0.015, 0.018)
                for we in (0.002, 0.003, 0.004, 0.005, 0.006)]
        grid += [(L, dw, we)
                 for L in (5000.0, 10000.0, 20000.0)
                 for dw in (0.010, 0.020, 0.040)
                 for we in (0.004, 0.006, 0.008, 0.010)]
        xs, ys, cols = [], [], []
        for L, dw, we in grid:
            cfg.physical.equilibrium_depth = L
            p = predict_peaks(cfg, dw, we, lookup, Tb=TB)
            # depth-aware overflow gate: the surface barrier caps h_max at ~D-2*delta
            if p.has_two_peaks and p.hmax_over_D > 1.0 - 200.0 / L and p.A_widening > 0:
                xs.append(p.A_widening * CRACK_LENGTH); ys.append(p.ratio); cols.append(L / 1e3)
        have.update(xs=np.array(xs), ys=np.array(ys), cols=np.array(cols))
        print(f"computed Fig 4b ({len(xs)} steady-state cases)")

    # --- Fig 5: mass flux of a closing-up crack (early wide vs near close-up) ---
    if "clos_MA_e" not in have:
        L5, dw5 = 5000.0, 0.020
        MA_e, g_e = _flux_curve(cfg, L5, dw5, 0.020, lookup)   # early: wide, weakly sealed
        MA_l, g_l = _flux_curve(cfg, L5, dw5, 0.006, lookup)   # near close-up: sealed
        have.update(clos_MA_e=MA_e, clos_g_e=g_e * CRACK_LENGTH,
                    clos_MA_l=MA_l, clos_g_l=g_l * CRACK_LENGTH,
                    clos_we_e=0.020, clos_we_l=0.006)
        print("computed Fig 5 (closing-up mass flux)")

    # --- Fig 6: condensation vs depth-in-crack at three cycle phases ---
    if "cond_dA" not in have:
        L6, dw6, we6 = 5000.0, 0.020, 0.008
        ma6, h6, w6, r6 = _liquid_cycle(cfg, L6, dw6, we6, lookup)
        D6 = L6 / 10.0
        iA = int(np.argmax(w6))       # wide crack, ~neutral level
        iB = int(np.argmax(h6))       # narrowing, high water level
        iC = int(np.argmin(h6))       # widening, low water level
        for tag, i in (("A", iA), ("B", iB), ("C", iC)):
            d, dep = _cond_profile(cfg, L6, float(h6[i]), float(w6[i]), float(r6[i]))
            have[f"cond_d{tag}"] = d
            have[f"cond_dep{tag}"] = dep
            have[f"cond_meta{tag}"] = np.array([ma6[i], h6[i] / D6, w6[i] * 1e3])
        print("computed Fig 6 (condensation profiles)")

    # --- Fig 8: convergence to the attractor from different initial widths ---
    # One fine w_eff sweep gives the water-rise relation h_max/D(w_eff); stepping
    # w_eff down by the model's condensation supply turns it into a time series.
    if "conv_we" not in have:
        cfg.physical.equilibrium_depth = 5000.0
        res = evolve_geometry_coupled(cfg, 0.012, n_e=44, w_eff_max=0.12, w_floor=1.5e-3)
        have.update(conv_we=res.w_eff * 1e3, conv_hD=(res.water_max - res.L) / res.D,
                    conv_dep=res.deposition_rate,
                    conv_wstar=res.w_eff_overflow * 1e3)
        print(f"computed Fig 8 (convergence; dep={res.deposition_rate:.2f} mm/cycle)")

    np.savez(cache, **have)
    print("cache written")


# --------------------------------------------------------------------------
# Plot (from cache)
# --------------------------------------------------------------------------
def plot(cache):
    d = np.load(cache)
    # Fig 3
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, dw in enumerate(d["seal_dw"]):
        we, rise = d["we_all"][k], d["rise_all"][k]
        line, = ax[0].plot(we, rise, "o-", ms=3, label=f"$\\Delta w$={dw*1e3:.0f} mm")
        i0 = len(we) * 6 // 10
        ax[0].annotate("", xy=(we[i0 + 1], rise[i0 + 1]), xytext=(we[i0], rise[i0]),
                       arrowprops=dict(arrowstyle="-|>", color=line.get_color(), lw=2))
    ax[0].axhline(1.0, color="grey", ls=":", label="overflow")
    ax[0].text(0.30, 0.78, "self-sealing\n(time)", transform=ax[0].transAxes,
               fontsize=9, ha="center", style="italic")
    ax[0].set_xlabel("effective width $w_{\\rm eff}$ [mm]"); ax[0].set_ylabel("max water rise / D")
    ax[0].invert_xaxis(); ax[0].set_title("(a) Every tidal swing seals toward overflow")
    ax[0].legend(fontsize=8)
    ax[1].plot(d["seal_dw"] * 1e3, d["wstar"], "ks-")
    ax[1].set_xlabel("absolute tidal swing $\\Delta w$ [mm]")
    ax[1].set_ylabel("seal depth at overflow $w_{\\rm eff}^*$ [mm]")
    ax[1].set_title("(b) Swing sets only how far it seals"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wall_seal_regime.pdf"))
    print("wrote wall_seal_regime.pdf")

    # Fig 4
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    MA, g = d["MA"], d["g"]
    ax[0].plot(MA, g / g.max(), "k-")
    ax[0].axvline(float(d["phi_w"]), ls="--", c="C0", label=f"widening {float(d['phi_w']):.0f}$^\\circ$")
    ax[0].axvline(float(d["phi_a"]), ls="--", c="C3", label=f"approach {float(d['phi_a']):.0f}$^\\circ$")
    ax[0].set_xlabel("mean anomaly [deg]"); ax[0].set_ylabel("mass flux (normalized)")
    ax[0].set_title("(a) Two gas-flux peaks"); ax[0].legend(fontsize=8)
    xs, ys, cols = d["xs"], d["ys"], d["cols"]
    styles = {1.5: ("o", "tab:blue"), 2.0: ("s", "tab:cyan"), 3.0: ("^", "tab:green"),
              5.0: ("D", "tab:orange"), 10.0: ("v", "tab:purple"), 20.0: ("P", "tab:brown")}
    for L in sorted(set(cols.tolist())):
        sel = cols == L
        mk, co = styles.get(L, ("o", "gray"))
        ax[1].scatter(xs[sel], ys[sel], marker=mk, c=co, s=50, edgecolor="k",
                      linewidth=0.4, zorder=3, label=f"$L$={L:g} km")
    ax[1].scatter([OBS_WIDENING_KGS], [OBS_RATIO], s=260, c="red", marker="*",
                  edgecolor="k", zorder=5, label="approx. observed")
    ax[1].axhline(1.0, color="grey", ls=":")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("widening-peak emission [kg/s]")
    ax[1].set_ylabel("approach / widening (= main / secondary)")
    ax[1].set_title("(b) Steady-state cases vs Enceladus"); ax[1].legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "peak_predictor.pdf"))
    print("wrote peak_predictor.pdf")

    # Fig 5: closing-up crack mass flux (early wide vs near close-up)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(d["clos_MA_e"], d["clos_g_e"], color="tab:blue", lw=1.8,
            label=f"early ($w_{{\\rm eff}}$={float(d['clos_we_e'])*1e3:.0f} mm, wide)")
    ax.plot(d["clos_MA_l"], d["clos_g_l"], color="tab:red", lw=1.8,
            label=f"near close-up ($w_{{\\rm eff}}$={float(d['clos_we_l'])*1e3:.0f} mm)")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("mass flux [kg s$^{-1}$]")
    ax.set_title("Mass flux as a crack seals ($L$=5 km, $\\Delta w$=20 mm)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "closing_massflux.pdf"))
    print("wrote closing_massflux.pdf")

    # Fig 6: condensation vs depth-in-crack at three cycle phases
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    labels = {"A": ("wide crack, neutral level", "tab:green"),
              "B": ("narrowing, high water level", "tab:red"),
              "C": ("widening, low water level", "tab:blue")}
    for tag, (lab, co) in labels.items():
        dd, dep = d[f"cond_d{tag}"], d[f"cond_dep{tag}"]
        ma, hD, wmm = d[f"cond_meta{tag}"]
        keep = dd >= 1e-2                      # view from 1 cm below the mouth down
        ax.plot(dep[keep], dd[keep], color=co, lw=1.8,
                label=f"{lab} (MA {ma:.0f}$^\\circ$, $h/D$={hD:.2f})")
        # mark the water level (deepest exposed point)
        ax.plot(dep[-1], dd[-1], "o", color=co, ms=6, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e3, 1e-2)                      # depth increases downward
    ax.set_xlabel("condensation flux [kg m$^{-2}$ s$^{-1}$]")
    ax.set_ylabel("depth below surface [m]")
    ax.set_title("Condensation is lip-concentrated and phase-dependent")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "condensation_profiles.pdf"))
    print("wrote condensation_profiles.pdf")

    # Fig 7: model peaks vs observed peak phases (single global phase offset)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    MA, g = d["MA"], d["g"]
    pa, pw = float(d["phi_a"]), float(d["phi_w"])
    off = 180.0 - pa                       # align the approach peak to observed main
    mas = (MA + off) % 360.0
    o = np.argsort(mas)
    ax.plot(mas[o], g[o] / g.max(), "k-", lw=1.9, label="model (phase-aligned)", zorder=4)
    for band, co, lab in ((OBS_MAIN_MA, "tab:orange", "observed main"),
                          (OBS_SEC_MA, "tab:blue", "observed secondary")):
        ax.axvspan(band[0], band[1], color=co, alpha=0.25, zorder=1,
                   label=f"{lab} (MA {band[0]:.0f}--{band[1]:.0f}$^\\circ$)")
    # mark the model's two peaks after the offset
    for phi, co in ((pa, "tab:orange"), (pw, "tab:blue")):
        ax.axvline((phi + off) % 360.0, color=co, ls="--", lw=1.3, zorder=3)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.set_ylim(0, 1.08)
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("mass flux (normalized)")
    ax.set_title("Model peaks fall in the observed peak phases")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase_overlay.pdf"))
    print("wrote phase_overlay.pdf")

    # Fig 8: convergence to the attractor from different initial widths
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    we, hD = d["conv_we"], np.clip(d["conv_hD"], 0.0, 1.0)
    dep = float(d["conv_dep"])              # mm/cycle per wall
    wstar = float(d["conv_wstar"])          # overflow seal depth (attractor floor)
    order = np.argsort(we)                  # ascending w_eff for np.interp
    wes, hDs = we[order], hD[order]
    step = 2.0 * dep                        # both walls line -> w_eff narrows
    inits = [(10.0, "tab:blue", "o"), (30.0, "tab:green", "s"),
             (60.0, "tab:orange", "^"), (100.0, "tab:red", "D")]
    for w0, co, mk in inits:
        ns = np.arange(0, 7)
        # w_eff narrows at the deposition rate, then holds at w_eff* where the
        # rising water balances condensation (the overflow attractor).
        weff_n = np.maximum(w0 - step * ns, wstar)
        h = np.interp(weff_n, wes, hDs)
        ax.plot(ns, h, mk + "-", color=co, ms=5, label=f"initial width {w0:.0f} mm")
    ax.axhline(0.9, color="grey", ls=":", label="overflow")
    ax.set_xlabel("diurnal cycles"); ax.set_ylabel("max water rise / D")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 6)
    ax.set_title("Any initial width converges to overflow ($\\Delta w$=12 mm)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "attractor_convergence.pdf"))
    print("wrote attractor_convergence.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookup", default=None)
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "encfig_cache.npz"))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        import logging; logging.basicConfig(level=logging.INFO, format="%(message)s")
    # A valid r-table lookup is needed whenever any group is missing; resolve a
    # persistent path (reused across runs) and generate it only if absent.
    path = args.lookup or os.path.join(tempfile.gettempdir(), "encfig_lut.npz")
    if not os.path.exists(path):
        generate_r_table(np.geomspace(1e-3, 0.08, 24), np.geomspace(0.5, 3000.0, 24),
                         Tb_arr=np.array([272.0, 273.1501]), output_path=path, n_jobs=-1)
    compute(GasLookupTable(path, clean=True), args.cache, recompute=args.recompute)
    plot(args.cache)


if __name__ == "__main__":
    main()
