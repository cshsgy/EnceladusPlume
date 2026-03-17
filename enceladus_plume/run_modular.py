#!/usr/bin/env python3
"""CLI entry point for the modular solver (2023 revision)."""

import argparse
import logging

from enceladus_plume.config import load_config
from enceladus_plume.workflows.modular_solver import run_modular_solver


def main():
    parser = argparse.ArgumentParser(
        description="Enceladus Plume Solver -- Modular solver (2023)")
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file (merged on top of defaults)")
    parser.add_argument("--times", default=None,
                        help="Path to times data file")
    parser.add_argument("--slips", default=None,
                        help="Path to slip time functions data file")
    parser.add_argument("--output", default="test_run.npz",
                        help="Output file path")
    parser.add_argument("--lookup", default="r_rec.npz",
                        help="Path to gas dynamics lookup table (.npz)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration of the lookup table")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel workers (-1 = all CPUs)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    run_modular_solver(
        times_file=args.times,
        slips_file=args.slips,
        cfg=cfg,
        output_path=args.output,
        lookup_path=args.lookup,
        regenerate_table=args.regenerate,
        n_jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
