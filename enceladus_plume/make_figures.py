#!/usr/bin/env python3
"""Regenerate the manuscript figures from the solver.

Produces (into writing/manuscript/Figures/):
  * wall_seal_regime.pdf  (Fig. 3) -- the self-sealing attractor
  * peak_predictor.pdf    (Fig. 4) -- the two mass-flux peaks

Run from the package directory:  python make_figures.py [-v]
(uses many liquid solves; minutes on a multi-core box).
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
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate
from enceladus_plume.utils import build_width_series
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.peaks import predict_peaks
from enceladus_plume.wall_geometry import evolve_geometry_coupled

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(_HERE, "..", "writing", "manuscript", "Figures"))

CRACK_LENGTH = 5.0e5          # m, summed tiger-stripe length
TB = 272.6
# Approximate observed scenario for the red dot in Fig. 4b. These are estimates
# (the absolute secondary-peak emission is not directly measured); refine with
# the exact slab-density ratio and emission.
OBS_WIDENING_KGS = 150.0      # widening (= observed secondary) peak emission, kg/s
OBS_RATIO = 2.0               # approach/widening (= observed main/secondary)


def _cfg():
    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 300.0
    cfg.liquid_dynamics.npts_velocity = 150
    return cfg


def _build_lookup(path):
    generate_r_table(np.geomspace(1e-3, 0.08, 24), np.geomspace(0.5, 3000.0, 24),
                     Tb_arr=np.array([272.0, 273.1501]), output_path=path, n_jobs=-1)
    return GasLookupTable(path, clean=True)


# --------------------------------------------------------------------------
# Fig. 3 -- self-sealing attractor (with direction-of-sealing arrows)
# --------------------------------------------------------------------------
def fig_seal():
    cfg = _cfg()
    cfg.physical.equilibrium_depth = 5000.0
    D = 500.0
    dws = [0.003, 0.006, 0.012, 0.025]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    wstar = []
    for dw in dws:
        res = evolve_geometry_coupled(cfg, dw, n_e=11, w_eff_max=0.06, w_floor=2e-3)
        we = res.w_eff * 1e3            # mm
        rise = (res.water_max - res.L) / D
        line, = ax[0].plot(we, rise, "o-", ms=3, label=f"$\\Delta w$={dw*1e3:.0f} mm")
        # arrow showing the direction of evolution (sealing: w_eff decreases, rise grows)
        i0, i1 = len(we) * 6 // 10, len(we) * 6 // 10 + 1
        ax[0].annotate("", xy=(we[i1], rise[i1]), xytext=(we[i0], rise[i0]),
                       arrowprops=dict(arrowstyle="-|>", color=line.get_color(), lw=2))
        wstar.append(res.w_eff_overflow * 1e3)
    ax[0].axhline(1.0, color="grey", ls=":", label="overflow")
    ax[0].text(0.30, 0.78, "self-sealing\n(time)", transform=ax[0].transAxes,
               fontsize=9, ha="center", style="italic")
    ax[0].set_xlabel("effective width $w_{\\rm eff}$ [mm]")
    ax[0].set_ylabel("max water rise / D")
    ax[0].invert_xaxis()
    ax[0].set_title("(a) Every tidal swing seals toward overflow")
    ax[0].legend(fontsize=8)

    ax[1].plot([d * 1e3 for d in dws], wstar, "ks-")
    ax[1].set_xlabel("absolute tidal swing $\\Delta w$ [mm]")
    ax[1].set_ylabel("seal depth at overflow $w_{\\rm eff}^*$ [mm]")
    ax[1].set_title("(b) Swing sets only how far it seals")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wall_seal_regime.pdf"))
    print("wrote wall_seal_regime.pdf")


# --------------------------------------------------------------------------
# Fig. 4 -- the two mass-flux peaks
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
    flux = gas + ov
    m = t >= (t[-1] - P)
    o = np.argsort(t[m] - t[m][0])
    return (t[m][o] - t[m][o][0]) / P * 360.0, flux[m][o]


def fig_peaks(lookup):
    cfg = _cfg()
    # panel (a): example flux curve
    cfg.physical.equilibrium_depth = 5000.0
    MA, g = _flux_curve(cfg, 5000.0, 0.010, 0.007, lookup)
    pr = predict_peaks(cfg, 0.010, 0.007, lookup, Tb=TB)

    # panel (b): steady-state (overflow) cases -> (widening emission, ratio)
    xs, ys, cols = [], [], []
    for L in (5000.0, 20000.0):
        for dw in (0.010, 0.020, 0.040):
            for we in (0.005, 0.007, 0.009, 0.011):
                cfg.physical.equilibrium_depth = L
                p = predict_peaks(cfg, dw, we, lookup, Tb=TB)
                if p.has_two_peaks and p.hmax_over_D > 0.95 and p.A_widening > 0:
                    xs.append(p.A_widening * CRACK_LENGTH)
                    ys.append(p.ratio)
                    cols.append(L / 1e3)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(MA, g / g.max(), "k-")
    ax[0].axvline(pr.phi_widening, ls="--", c="C0", label=f"widening {pr.phi_widening:.0f}$^\\circ$")
    ax[0].axvline(pr.phi_approach, ls="--", c="C3", label=f"approach {pr.phi_approach:.0f}$^\\circ$")
    ax[0].set_xlabel("mean anomaly [deg]")
    ax[0].set_ylabel("mass flux (normalized)")
    ax[0].set_title("(a) Two gas-flux peaks")
    ax[0].legend(fontsize=8)

    sc = ax[1].scatter(xs, ys, c=cols, cmap="viridis", s=45, edgecolor="k", linewidth=0.4, zorder=3)
    ax[1].scatter([OBS_WIDENING_KGS], [OBS_RATIO], s=240, c="red", marker="o",
                  edgecolor="k", zorder=5, label="approx. observed")
    ax[1].axhline(1.0, color="grey", ls=":")
    cb = fig.colorbar(sc, ax=ax[1]); cb.set_label("depth $L$ [km]")
    ax[1].set_xlabel("widening-peak emission [kg/s]")
    ax[1].set_ylabel("approach / widening (= main / secondary)")
    ax[1].set_title("(b) Steady-state cases vs Enceladus")
    ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "peak_predictor.pdf"))
    print(f"wrote peak_predictor.pdf ({len(xs)} steady-state cases)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookup", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        import logging; logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.lookup or os.path.join(tempfile.mkdtemp(), "lut.npz")
    if not (args.lookup and os.path.exists(args.lookup)):
        _build_lookup(path)
    lookup = GasLookupTable(path, clean=True)
    fig_seal()
    fig_peaks(lookup)


if __name__ == "__main__":
    main()
