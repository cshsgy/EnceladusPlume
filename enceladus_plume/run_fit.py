#!/usr/bin/env python3
"""Fit the forward model to the observed diurnal plume profile.

Fits the model total mass-flux-versus-mean-anomaly curve (gas evaporation plus
buffered overflow, via ``peaks._flux_curve``) to the digitized Ingersoll et al.
(2020) slab-density profile (``data/observed_diurnal.csv``).

The crack is placed on the self-sealing attractor, so the effective width is not
free but set by the overflow seal depth ``w_eff*(dw, L)`` from
``evolve_geometry_coupled``. Free parameters:

    dw   : absolute tidal swing            [m]
    L    : effective source depth          [m]
    phi0 : global mean-anomaly phase offset [deg]
    A    : amplitude (slab density per unit model flux) -- a nuisance scale

``phi0`` is scanned and ``A`` is solved in closed form (weighted linear least
squares) at every (dw, L) grid node, so the nonlinear search is only over the
2-D (dw, L) grid; a parabolic refine and Delta-chi^2 = 1 give uncertainties.

Usage:  python run_fit.py [--lookup PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

from enceladus_plume.config import load_config
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import (
    liquid_dynamics, compute_overflow_rate, buffer_overflow,
)
from enceladus_plume.wall_geometry import evolve_geometry_coupled
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data", "observed_diurnal.csv")
TB = 272.6

# (dw, L) search grid. w_eff* (overflow seal depth) is computed per cell.
DW_GRID = np.array([10, 12, 14, 17, 20, 25]) * 1e-3              # m
L_GRID = np.array([3, 5, 7, 10, 14, 20]) * 1e3                   # m
PHI_STEP = 2.0                                                   # deg

# Second-harmonic (double-cosine) forcing: amplitude scale and phase (deg). The
# tidal stress across the tiger stripes is not a pure single sinusoid; a 2f term
# (shifted-double-cosine forcing, MeyerZuWestram2024) is fit per cell. scale=0
# recovers the single-cosine exactly and is always included as the baseline.
HARM_SCALES = np.array([0.0, 1.5, 2.0])
HARM_PHASES = np.array([45.0, 67.5])


def _cfg():
    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 300.0
    cfg.liquid_dynamics.npts_velocity = 40   # h_max/D identical to 150 here, ~2x faster
    return cfg


def _flux_curve(cfg, L, dw, we, lookup, harm_scale=0.0, harm_phase=0.0):
    """Total mass flux (gas + buffered overflow) over the last cycle vs MA.

    Uses the shifted-double-cosine forcing; ``harm_scale=0`` recovers the
    single-cosine exactly. Mirrors make_figures._flux_curve otherwise.
    """
    D = L / 10.0
    P = cfg.physical.orbital_period
    f_evap = cfg.physical.latent_heat_fusion / (
        cfg.physical.latent_heat + cfg.physical.latent_heat_fusion)
    rho_w = cfg.physical.liquid_density
    t_in = np.arange(100, P + 1, 200.0)
    R = 1.0 + dw / we
    w_in = build_width_series(t_in, R, we, orbital_period=P,
                              forcing_model="shifted-double-cosine",
                              second_harmonic_scale=harm_scale,
                              second_harmonic_phase_deg=harm_phase)
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
    _, phi, _, _ = lookup.query_vectorized(
        TB, np.clip(D - h, lookup.depth.min(), lookup.depth.max()),
        np.clip(w, lookup.delta.min(), lookup.delta.max()))
    gas = np.nan_to_num(phi * w)
    ov = np.nan_to_num(f_evap * rho_w * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg))
    ov = buffer_overflow(t, ov, cfg.liquid_dynamics.overflow_tau)
    flux = gas + ov
    m = t >= (t[-1] - P)
    o = np.argsort(t[m] - t[m][0])
    return (t[m][o] - t[m][o][0]) / P * 360.0, flux[m][o]


def _best_phi_A(MA_m, flux_m, ma_o, y_o, w_o):
    """Best phase offset (scan) and closed-form amplitude; returns (phi0,A,chi2)."""
    best = (np.nan, np.nan, np.inf)
    for phi0 in np.arange(0.0, 360.0, PHI_STEP):
        m = np.interp((ma_o - phi0) % 360.0, MA_m, flux_m, period=360.0)
        # weighted linear LSQ for A: minimize sum w (A m - o)^2
        denom = np.sum(w_o * m * m)
        if denom <= 0:
            continue
        A = np.sum(w_o * m * y_o) / denom
        if A <= 0:
            continue
        chi2 = float(np.sum(w_o * (A * m - y_o) ** 2))
        if chi2 < best[2]:
            best = (float(phi0), float(A), chi2)
    return best


def fit(lookup, cfg=None):
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    w_o = 1.0 / sig ** 2

    # harmonic combos: (scale, phase); scale=0 (single-cosine baseline) once.
    combos = [(0.0, 0.0)] + [(float(s), float(p)) for s in HARM_SCALES if s > 0
                             for p in HARM_PHASES]
    chi2 = np.full((len(DW_GRID), len(L_GRID)), np.inf)
    phi = np.full_like(chi2, np.nan)
    amp = np.full_like(chi2, np.nan)
    weff = np.full_like(chi2, np.nan)
    hsc = np.full_like(chi2, np.nan)   # best 2f amplitude scale per cell
    hph = np.full_like(chi2, np.nan)   # best 2f phase per cell
    chi2_single = np.full_like(chi2, np.inf)  # scale=0 baseline per cell
    total = len(DW_GRID) * len(L_GRID)
    t0 = time.time()
    done = 0
    for i, dw in enumerate(DW_GRID):
        for j, L in enumerate(L_GRID):
            cfg.physical.equilibrium_depth = float(L)
            # w_eff* from single-cosine attractor (swing is preserved by the
            # forcing normalization, so the seal depth is ~harmonic-independent).
            res = evolve_geometry_coupled(cfg, float(dw), n_e=7,
                                          w_eff_max=0.06, w_floor=2e-3)
            if res.overflow and np.isfinite(res.w_eff_overflow):
                we = float(res.w_eff_overflow)
                weff[i, j] = we
                for (sc, ph) in combos:
                    MA_m, flux_m = _flux_curve(cfg, float(L), float(dw), we, lookup,
                                               harm_scale=sc, harm_phase=ph)
                    o = np.argsort(MA_m)
                    p0, A, c2 = _best_phi_A(MA_m[o], flux_m[o], ma_o, y_o, w_o)
                    if sc == 0.0:
                        chi2_single[i, j] = c2
                    if c2 < chi2[i, j]:
                        chi2[i, j], phi[i, j], amp[i, j] = c2, p0, A
                        hsc[i, j], hph[i, j] = sc, ph
            done += 1
            el = time.time() - t0
            eta = el / done * (total - done)
            print(f"  [{done:2d}/{total}] dw={dw*1e3:4.0f}mm L={L/1e3:4.1f}km | "
                  f"elapsed {el/60:4.1f}m | ETA {eta/60:4.1f}m | "
                  f"min chi2={np.nanmin(chi2):.1f}", flush=True)

    bi, bj = np.unravel_index(np.argmin(chi2), chi2.shape)
    dof = len(ma_o) - 4
    result = dict(
        dw=float(DW_GRID[bi]), L=float(L_GRID[bj]), w_eff=float(weff[bi, bj]),
        phi0=float(phi[bi, bj]), A=float(amp[bi, bj]),
        harm_scale=float(hsc[bi, bj]), harm_phase=float(hph[bi, bj]),
        chi2=float(chi2[bi, bj]), dof=int(dof), chi2_red=float(chi2[bi, bj] / dof),
        chi2_single_best=float(np.nanmin(chi2_single)),
        DW_GRID=DW_GRID, L_GRID=L_GRID, chi2_grid=chi2, bi=int(bi), bj=int(bj),
    )
    # Delta chi^2 = 1 marginal ranges along each grid axis through the optimum
    def _range(vals, cs):
        m = np.isfinite(cs)
        if m.sum() < 2:
            return (np.nan, np.nan)
        lo = vals[m][cs[m] <= cs.min() + 1.0]
        return (float(lo.min()), float(lo.max()))
    result["dw_1sig"] = _range(DW_GRID, chi2[:, bj])
    result["L_1sig"] = _range(L_GRID, chi2[bi, :])
    return result


def _ensemble_smooth(MA, flux, sigma_deg):
    """Ensemble-average the flux over a Gaussian spread of forcing phase.

    The observed emission sums over ~500 km of tiger-stripe segments that differ
    in local forcing phase; averaging the single-crack curve over that spread is
    a periodic Gaussian convolution in mean anomaly (width ``sigma_deg``).
    Returns a uniform (grid, smoothed-flux). sigma_deg<=0 returns the raw curve.
    """
    g = np.linspace(0.0, 360.0, 720, endpoint=False)
    f = np.interp(g, MA, flux, period=360.0)
    if sigma_deg <= 0:
        return g, f
    dx = g[1] - g[0]
    half = int(np.ceil(4.0 * sigma_deg / dx))
    k = np.arange(-half, half + 1) * dx
    w = np.exp(-0.5 * (k / sigma_deg) ** 2); w /= w.sum()
    fs = np.convolve(np.concatenate([f, f, f]), w, mode="same")[len(f):2 * len(f)]
    return g, fs


def fit_ensemble(result, lookup, cfg=None):
    """Fit the along-strike phase-spread width to the base best-fit model.

    Holds (dw, L, w_eff, harmonic) at the single-crack best fit and scans the
    ensemble spread sigma (deg), re-optimising phi0 and A. Returns the best
    sigma, its chi2, and the single-crack (sigma=0) chi2 for comparison.
    """
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    w_o = 1.0 / sig ** 2
    dw, L, we = float(result["dw"]), float(result["L"]), float(result["w_eff"])
    hs, hp = float(result.get("harm_scale", 0.0)), float(result.get("harm_phase", 0.0))
    cfg.physical.equilibrium_depth = L
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lookup, harm_scale=hs, harm_phase=hp)
    o = np.argsort(MA_m); MA_m, flux_m = MA_m[o], flux_m[o]
    best = (0.0, np.inf, np.nan, np.nan)
    c0 = np.inf
    for sigma in np.arange(0.0, 40.1, 1.0):
        g, fs = _ensemble_smooth(MA_m, flux_m, sigma)
        p0, A, c2 = _best_phi_A(g, fs, ma_o, y_o, w_o)
        if sigma == 0.0:
            c0 = c2
        if c2 < best[1]:
            best = (float(sigma), c2, float(p0), float(A))
    dof = len(ma_o) - 5  # + sigma vs the single-crack's 4
    return dict(sigma=best[0], chi2=best[1], chi2_red=best[1] / dof, dof=int(dof),
                phi0=best[2], A=best[3], chi2_single=c0, chi2_single_red=c0 / (dof + 1),
                dw=dw, L=L, w_eff=we, harm_scale=hs, harm_phase=hp)


def plot_ensemble(result, lookup, cfg=None, out=None):
    """Fig. 10: single-crack vs ensemble-averaged fit to the observed profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = cfg or _cfg()
    e = fit_ensemble(result, lookup, cfg)
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    dw, L, we = e["dw"], e["L"], e["w_eff"]
    cfg.physical.equilibrium_depth = L
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lookup,
                               harm_scale=e["harm_scale"], harm_phase=e["harm_phase"])
    o = np.argsort(MA_m); MA_m, flux_m = MA_m[o], flux_m[o]
    # single-crack (sigma=0): reuse its own best phi0/A
    _, fs0 = _ensemble_smooth(MA_m, flux_m, 0.0)
    from numpy import interp
    w_o = 1.0 / sig ** 2
    p00, A0, _ = _best_phi_A(MA_m, flux_m, ma_o, y_o, w_o)
    gcur = np.linspace(0, 360, 721)
    single = A0 * np.interp((gcur - p00) % 360.0, MA_m, flux_m, period=360.0)
    g, fse = _ensemble_smooth(MA_m, flux_m, e["sigma"])
    ens = e["A"] * np.interp((gcur - e["phi0"]) % 360.0, g, fse, period=360.0)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(ma_o, y_o, yerr=sig, fmt="o", ms=4, color="k", lw=1, capsize=2,
                label="observed (digitized, Ingersoll+ 2020)")
    ax.plot(gcur, single, "--", color="tab:blue", lw=1.5,
            label=f"single crack ($\\chi^2$/dof={e['chi2_single_red']:.2f})")
    ax.plot(gcur, ens, "-", color="tab:red", lw=1.9,
            label=(f"ensemble, $\\sigma_\\phi$={e['sigma']:.0f}$^\\circ$ "
                   f"($\\chi^2$/dof={e['chi2_red']:.2f})"))
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("slab density [kg km$^{-1}$]")
    ax.set_title("Ensemble averaging over tiger-stripe segments")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = out or os.path.normpath(os.path.join(
        _HERE, "..", "writing", "manuscript", "Figures", "diurnal_fit_ensemble.pdf"))
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")
    print(f"  ensemble sigma_phi = {e['sigma']:.0f} deg   "
          f"chi2/dof {e['chi2_single_red']:.2f} (single) -> {e['chi2_red']:.2f} (ensemble)")
    return e


def plot_overlay(result, lookup, cfg=None, out=None):
    """Data-vs-model overlay for the best fit (recomputes one flux curve)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    dw, L, we = float(result["dw"]), float(result["L"]), float(result["w_eff"])
    phi0, A = float(result["phi0"]), float(result["A"])
    hs, hp = float(result.get("harm_scale", 0.0)), float(result.get("harm_phase", 0.0))
    cfg.physical.equilibrium_depth = L
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lookup, harm_scale=hs, harm_phase=hp)
    o = np.argsort(MA_m); MA_m, flux_m = MA_m[o], flux_m[o]
    grid = np.linspace(0, 360, 721)
    model = A * np.interp((grid - phi0) % 360.0, MA_m, flux_m, period=360.0)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(ma_o, y_o, yerr=sig, fmt="o", ms=4, color="k", lw=1,
                capsize=2, label="observed (digitized, Ingersoll+ 2020)")
    ax.plot(grid, model, "-", color="tab:red", lw=1.8,
            label=(f"best fit: $\\Delta w$={dw*1e3:.0f} mm, $L$={L/1e3:.0f} km, "
                   f"2f scale={hs:g}@{hp:g}$^\\circ$, $\\phi_0$={phi0:.0f}$^\\circ$"))
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("slab density [kg km$^{-1}$]")
    ax.set_title("Diurnal profile: model fit to observed emission")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = out or os.path.normpath(os.path.join(
        _HERE, "..", "writing", "manuscript", "Figures", "diurnal_fit.pdf"))
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookup", default=os.path.join(tempfile.gettempdir(), "encfig_lut.npz"))
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "diurnal_fit.npz"))
    ap.add_argument("--plot-only", action="store_true",
                    help="skip fitting; load --out and just (re)draw the overlay")
    ap.add_argument("--ensemble", action="store_true",
                    help="skip fitting; load --out, fit the ensemble spread, draw Fig. 10")
    args = ap.parse_args()
    if args.ensemble:
        r = dict(np.load(args.out))
        plot_ensemble(r, GasLookupTable(args.lookup, clean=True))
        return
    if args.plot_only:
        r = dict(np.load(args.out))
        plot_overlay(r, GasLookupTable(args.lookup, clean=True))
        return
    if not os.path.exists(args.lookup):
        generate_r_table(np.geomspace(1e-3, 0.08, 24), np.geomspace(0.5, 3000.0, 24),
                         Tb_arr=np.array([272.0, 273.1501]), output_path=args.lookup, n_jobs=-1)
    lut = GasLookupTable(args.lookup, clean=True)
    r = fit(lut)
    np.savez(args.out, **{k: v for k, v in r.items()})
    print("\n=== BEST FIT (on-attractor) ===")
    print(f"  dw   = {r['dw']*1e3:.1f} mm   (Delta-chi2=1: "
          f"{r['dw_1sig'][0]*1e3:.0f}-{r['dw_1sig'][1]*1e3:.0f} mm)")
    print(f"  L    = {r['L']/1e3:.1f} km   (Delta-chi2=1: "
          f"{r['L_1sig'][0]/1e3:.1f}-{r['L_1sig'][1]/1e3:.1f} km)")
    print(f"  w_eff*= {r['w_eff']*1e3:.2f} mm (on attractor)")
    print(f"  phi0 = {r['phi0']:.0f} deg")
    print(f"  A    = {r['A']:.3e} (slab per model-flux unit)")
    print(f"  2f harmonic scale={r['harm_scale']:g} phase={r['harm_phase']:g} deg")
    print(f"  chi2/dof = {r['chi2']:.1f}/{r['dof']} = {r['chi2_red']:.2f}  "
          f"(single-cosine best chi2/dof = {r['chi2_single_best']/r['dof']:.2f})")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
