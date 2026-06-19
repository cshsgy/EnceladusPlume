#!/usr/bin/env python3
"""Predict the two diurnal mass-flux peaks (secondary vs main) for a sealed crack.

End-to-end: builds (or reuses) a gas lookup table sized to the case, then reports
the widening and approach peaks --- their phase, strength, the
approach/widening ratio, and which is the main peak.

Usage
-----
    python run_peaks.py --delta-w 0.01 --w-eff 0.007
    python run_peaks.py --delta-w 0.02 --w-eff 0.010 --L 5000 \
                        --lookup lut.npz -v        # caches the lookup in lut.npz

``--delta-w`` is the absolute tidal swing w_max-w_min (m); ``--w-eff`` is the
sealed effective minimum width (m). See enceladus_plume.peaks / wall_geometry.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from enceladus_plume.config import load_config
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.peaks import predict_peaks

logger = logging.getLogger("peaks")


def _build_lookup(path, L, delta_w, w_eff, n_grid, n_jobs):
    """Build a lookup sized to the case (depth and width ranges from L, the swing)."""
    D = L / 10.0
    depths = np.geomspace(0.5, 1.5 * D, n_grid)
    w_max = w_eff + delta_w
    deltas = np.geomspace(max(0.5 * w_eff, 1e-4), 1.3 * w_max, n_grid)
    logger.info("building lookup: depth [%.1f, %.1f] m, width [%.4f, %.4f] m, %dx%d",
                depths.min(), depths.max(), deltas.min(), deltas.max(), n_grid, n_grid)
    generate_r_table(deltas, depths, Tb_arr=np.array([272.0, 273.1501]),
                     output_path=path, n_jobs=n_jobs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta-w", type=float, required=True, help="absolute tidal swing w_max-w_min (m)")
    ap.add_argument("--w-eff", type=float, required=True, help="sealed effective minimum width (m)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--L", type=float, default=None, help="equilibrium water depth (m); overrides config")
    ap.add_argument("--Tb", type=float, default=272.6)
    ap.add_argument("--forcing-model", default="single-cosine")
    ap.add_argument("--lookup", default=None,
                    help="lookup .npz to reuse; built and cached here if missing")
    ap.add_argument("--n-grid", type=int, default=24, help="lookup grid resolution")
    ap.add_argument("--jobs", type=int, default=-1, help="lookup parallel workers")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    cfg = load_config(args.config)
    if args.L is not None:
        cfg.physical.equilibrium_depth = args.L
    L = float(cfg.physical.equilibrium_depth)

    if args.lookup and os.path.exists(args.lookup):
        logger.info("loading lookup %s", args.lookup)
        lookup = GasLookupTable(args.lookup, clean=True)
    else:
        path = args.lookup or os.path.join(os.getcwd(), "_peaks_lookup.npz")
        _build_lookup(path, L, args.delta_w, args.w_eff, args.n_grid, args.jobs)
        lookup = GasLookupTable(path, clean=True)

    pred = predict_peaks(cfg, args.delta_w, args.w_eff, lookup, Tb=args.Tb,
                         forcing_model=args.forcing_model)
    print(pred.summary())


if __name__ == "__main__":
    main()
