#!/usr/bin/env python3
"""Stage B: self-consistent crack geometry from the wall ice budget.

Loads a per-case ``.npz`` produced by ``run_pipeline.py`` (carrying the
``t, h, w, v, r`` series), evolves the wall ice lining to a quasi-steady taper,
and writes the equilibrium geometry plus a plot.

Usage
-----
    python run_wall_geometry.py --case results/wmin0.050_ratio1.3.npz \
                                --config config/run_config.yaml -v
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from enceladus_plume.config import load_config
from enceladus_plume.wall_geometry import evolve_geometry

logger = logging.getLogger("wall_geometry")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="per-case .npz from run_pipeline.py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--Tb", type=float, default=272.6)
    ap.add_argument("--n-z", type=int, default=600)
    ap.add_argument("--n-t", type=int, default=240)
    ap.add_argument("--cycles-per-iter", type=float, default=50.0)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    cfg = load_config(args.config)
    data = np.load(args.case)
    for key in ("t", "h", "w", "v", "r"):
        if key not in data:
            raise SystemExit(
                f"case file {args.case!r} is missing '{key}'. Re-run the pipeline "
                f"(it now saves the velocity series).")

    res = evolve_geometry(
        data["t"], data["h"], data["w"], data["v"], data["r"], cfg, Tb=args.Tb,
        n_z=args.n_z, n_t=args.n_t, cycles_per_iter=args.cycles_per_iter,
        max_iter=args.max_iter)

    print(res.summary())

    out = args.out or (os.path.splitext(args.case)[0] + "_geometry.npz")
    np.savez(out, zeta=res.zeta, e_ice=res.e_ice, w_eff_min=res.w_eff_min,
             L=res.L, D=res.D, water_max_height=res.water_max_height,
             seal_height=res.seal_height, open_top_height=res.open_top_height,
             iterations=res.iterations, converged=res.converged,
             history=np.asarray(res.history))
    logger.info("wrote %s", out)

    if not args.no_plot:
        _plot(res, os.path.splitext(out)[0] + ".png")


def _plot(res, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("plotting skipped (%s)", exc)
        return

    below = res.surface - res.zeta  # depth below surface
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    axes[0].plot(res.e_ice * 1e3, below, color="C0")
    axes[0].set_xlabel("equilibrium ice lining [mm/wall]")
    axes[0].set_ylabel("depth below surface [m]")
    axes[0].set_title("Stage-B taper")

    axes[1].plot(res.w_eff_min * 1e3, below, color="k")
    axes[1].set_xlabel("min effective opening [mm]")
    axes[1].set_title("Effective gap")

    for ax in axes:
        ax.axhline(res.surface - res.water_max_height, ls="--", color="C2",
                   label="water reach")
        ax.axhline(res.surface - res.open_top_height, ls=":", color="C3",
                   label="open-channel top")
        ax.set_ylim(res.surface, 0)
        ax.set_yscale("symlog")
        ax.legend(fontsize=8)
    fig.suptitle("Self-consistent crack geometry (open channel pinned at water reach)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
