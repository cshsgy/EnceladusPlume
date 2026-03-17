"""Modular solver workflow (2023 revision).

Port of run_all.m and slip_run.m -- runs liquid dynamics with a
prescribed slip/width time series, then uses the pre-computed gas
lookup table to get mass flux.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from ..config import Config, load_config
from ..liquid_dynamics.solver import liquid_dynamics
from ..gas_dynamics.lookup import GasLookupTable
from ..gas_dynamics.interpolator import generate_r_table
from ..utils import save_results

logger = logging.getLogger(__name__)


def slip_run(
    t_in: np.ndarray,
    w_in: np.ndarray,
    L: float,
    lookup: GasLookupTable,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one slip profile: liquid dynamics then gas interpolation.

    Parameters
    ----------
    t_in   : time array (s), one orbital period
    w_in   : crack width array (m), same length as t_in
    L      : equilibrium water depth (m)
    lookup : pre-loaded gas lookup table
    cfg    : configuration

    Returns
    -------
    t_rec, w_rec, h_rec, phi_rec, dphi_rec
    """
    P = cfg.physical.orbital_period
    D0 = L / 10.0

    w_rec, h_rec, t_rec, _v_rec = liquid_dynamics(w_in, t_in, L, cfg)

    # Shift to keep one cycle
    while t_rec[-1] > P / 2:
        t_rec = t_rec - P

    t_rec = t_rec + P
    mask = t_rec > 0
    t_rec = t_rec[mask]
    w_rec = w_rec[mask]
    h_rec = h_rec[mask]

    Tb = cfg.physical.water_temperature
    phi_rec = np.zeros(len(t_rec))
    dphi_rec = np.zeros(len(t_rec))

    for i in range(len(t_rec)):
        d = D0 - h_rec[i]
        r, phi, rho, phi0 = lookup.query(Tb, d, w_rec[i])
        phi_rec[i] = phi * w_rec[i]
        dphi_rec[i] = (phi0 - phi) * w_rec[i]

    return t_rec, w_rec, h_rec, phi_rec, dphi_rec


def run_modular_solver(
    times_file: Optional[str] = None,
    slips_file: Optional[str] = None,
    cfg: Optional[Config] = None,
    output_path: str = "test_run.npz",
    lookup_path: str = "r_rec.npz",
    regenerate_table: bool = False,
    n_jobs: int = -1,
) -> dict[str, list[np.ndarray]]:
    """Execute the modular solver over all slip profiles.

    Parameters
    ----------
    times_file : path to times data (one column, degrees -> seconds)
    slips_file : path to slip amplitudes (n_times x n_profiles)
    cfg        : Config object
    output_path: where to save results
    lookup_path: path to gas lookup table (.npz)
    regenerate_table : if True, re-generate the lookup table first
    n_jobs     : parallel workers for the slip-profile loop

    Returns
    -------
    Dictionary with lists of per-profile arrays.
    """
    if cfg is None:
        cfg = load_config()

    P = cfg.physical.orbital_period
    L = cfg.physical.equilibrium_depth
    w_base = cfg.interpolation.w_base
    n_grid = cfg.interpolation.n_grid

    if times_file is None:
        times_file = cfg.modular_solver.times_file
    if slips_file is None:
        slips_file = cfg.modular_solver.slips_file

    t_in_raw = np.loadtxt(times_file)
    slips_in = np.loadtxt(slips_file)

    # Convert from degrees to seconds
    t_in = t_in_raw / 360.0 * P

    # Ensure 2-D
    if slips_in.ndim == 1:
        slips_in = slips_in.reshape(-1, 1)

    n_profiles = slips_in.shape[1]
    logger.info("Loaded %d time steps, %d slip profiles", len(t_in), n_profiles)

    # Optionally regenerate lookup table
    if regenerate_table or not Path(lookup_path).exists():
        w_max = float(np.max(slips_in)) + w_base
        widths = np.exp(np.linspace(np.log(w_base) - 0.01,
                                    np.log(w_max) + 0.01, n_grid))
        depths = np.exp(np.linspace(np.log(10) - 0.01,
                                    np.log(18000) + 0.01, n_grid))
        logger.info("Generating gas dynamics lookup table ...")
        generate_r_table(widths, depths, output_path=lookup_path, n_jobs=n_jobs)

    lookup = GasLookupTable(lookup_path)

    # Skip first row (header / zero row) per MATLAB: t_in(2:end), slips_in(2:end,:)
    t_in_use = t_in[1:]

    def _run_one(idx):
        w_in = slips_in[1:, idx] + w_base
        return slip_run(t_in_use, w_in, L, lookup, cfg)

    logger.info("Running %d slip profiles ...", n_profiles)
    results_list = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(i) for i in range(n_profiles)
    )

    width_data = [r[1] for r in results_list]
    level_data = [r[2] for r in results_list]
    phi_data = [r[3] for r in results_list]
    dphi_data = [r[4] for r in results_list]
    time_data = [r[0] for r in results_list]

    np.savez(output_path,
             width_data=np.array(width_data, dtype=object),
             level_data=np.array(level_data, dtype=object),
             phi_data=np.array(phi_data, dtype=object),
             dphi_data=np.array(dphi_data, dtype=object),
             time_data=np.array(time_data, dtype=object))

    logger.info("Results saved to %s", output_path)
    return {
        "width_data": width_data,
        "level_data": level_data,
        "phi_data": phi_data,
        "dphi_data": dphi_data,
        "time_data": time_data,
    }
