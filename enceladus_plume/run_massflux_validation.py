#!/usr/bin/env python3
"""Validate the model's absolute mass flux against observations.

The solver returns the diurnal mass flux *per unit crack length*
(``phi_top * width`` for the gas plus the overflow term, kg m^-1 s^-1). Total
plume emission is that times the active fracture length, taken as the
~500 km summed length of the tiger stripes. We compare the cycle-mean and peak
total emission against the observed ~200-400 kg/s (Hansen et al., 2011, 2017),
and the model's secondary/main peak ratio against the observed two-peak profile
(Ingersoll et al., 2020).

Usage:
    python run_massflux_validation.py [--lookup lut.npz] [--crack-length 5e5] [-v]
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

import numpy as np

from enceladus_plume.config import load_config
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate
from enceladus_plume.utils import build_width_series
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.peaks import predict_peaks

logger = logging.getLogger("massflux")

# Representative attractor cases: (L depth [m], absolute swing dW [m], sealed w_eff [m]).
CASES = [
    (5000.0, 0.010, 0.008),
    (5000.0, 0.020, 0.009),
    (20000.0, 0.010, 0.010),
    (20000.0, 0.020, 0.012),
]

OBS_TOTAL = (200.0, 400.0)  # kg/s, Hansen et al.


def _massflux_series(cfg, L, dw, w_eff, lookup, Tb):
    """Diurnal total mass flux per crack length (kg/m/s) over the last cycle."""
    cfg.physical.equilibrium_depth = L
    D = L / 10.0
    P = cfg.physical.orbital_period
    Lv = cfg.physical.latent_heat
    Lf = cfg.physical.latent_heat_fusion
    f_evap = Lf / (Lv + Lf)
    rho_w = cfg.physical.liquid_density
    t_in = np.arange(100, P + 1, 200.0)

    R = 1.0 + dw / w_eff
    w_in = build_width_series(t_in, R, w_eff, orbital_period=P, forcing_model="single-cosine")
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
    dlo, dhi = lookup.depth.min(), lookup.depth.max()
    wlo, whi = lookup.delta.min(), lookup.delta.max()
    _, phi, _, _ = lookup.query_vectorized(Tb, np.clip(D - h, dlo, dhi), np.clip(w, wlo, whi))
    gas = np.nan_to_num(phi * w)
    overflow = np.nan_to_num(f_evap * rho_w * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg))
    flux = gas + overflow
    m = t >= (t[-1] - P)
    return flux[m], h.max(), D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lookup", default=None, help="lookup .npz to reuse/build")
    ap.add_argument("--crack-length", type=float, default=5.0e5, help="active fracture length (m)")
    ap.add_argument("--Tb", type=float, default=272.6)
    ap.add_argument("--n-grid", type=int, default=24)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 300.0
    cfg.liquid_dynamics.npts_velocity = 150

    path = args.lookup or os.path.join(tempfile.mkdtemp(), "lut.npz")
    if not (args.lookup and os.path.exists(args.lookup)):
        depths = np.geomspace(0.5, 3000.0, args.n_grid)
        deltas = np.geomspace(1e-3, 0.08, args.n_grid)
        generate_r_table(deltas, depths, Tb_arr=np.array([272.0, 273.1501]),
                         output_path=path, n_jobs=-1)
    lookup = GasLookupTable(path, clean=True)

    Lc = args.crack_length
    print(f"Active fracture length: {Lc/1e3:.0f} km. "
          f"Observed total emission: {OBS_TOTAL[0]:.0f}-{OBS_TOTAL[1]:.0f} kg/s.\n")
    print(f"{'L(km)':>6} {'dW(mm)':>7} {'w_eff(mm)':>9} {'mean(kg/s)':>11} {'peak(kg/s)':>11} "
          f"{'A_a/A_w':>8} {'sec/main':>9} {'h_max/D':>8}")
    for L, dw, we in CASES:
        flux, hmax, D = _massflux_series(cfg, L, dw, we, lookup, args.Tb)
        mean_tot = float(np.nanmean(flux)) * Lc
        peak_tot = float(np.nanmax(flux)) * Lc
        pr = predict_peaks(cfg, dw, we, lookup, Tb=args.Tb)
        ratio = pr.ratio
        sec_main = (1.0 / ratio) if (ratio == ratio and ratio > 1) else ratio
        print(f"{L/1e3:6.0f} {dw*1e3:7.0f} {we*1e3:9.0f} {mean_tot:11.1f} {peak_tot:11.1f} "
              f"{ratio:8.2f} {sec_main:9.2f} {hmax/D:8.2f}")
    print("\nNote: 'sec/main' is the weaker peak over the stronger; observations show a "
          "secondary peak weaker than the main near MA 50 vs 180 (Ingersoll et al. 2020).")


if __name__ == "__main__":
    main()
