#!/usr/bin/env python3
"""CLI entry point for the full coupled solver (2022 paper version)."""

import argparse
import logging
import sys

from enceladus_plume.config import load_config
from enceladus_plume.workflows.full_solver import run_full_solver


def main():
    parser = argparse.ArgumentParser(
        description="Enceladus Plume Solver -- Full coupled solver (2022)")
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file (merged on top of defaults)")
    parser.add_argument("--wmin", type=float, default=None,
                        help="Minimum crack width (m)")
    parser.add_argument("--wmaxmin", type=float, default=None,
                        help="Ratio wmax / wmin")
    parser.add_argument("--depth", type=float, default=None,
                        help="Crack length / equilibrium water depth (m)")
    parser.add_argument("--output", default="run_results.npz",
                        help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    results = run_full_solver(
        wmin=args.wmin,
        wmaxmin=args.wmaxmin,
        depth=args.depth,
        cfg=cfg,
        output_path=args.output,
    )
    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
