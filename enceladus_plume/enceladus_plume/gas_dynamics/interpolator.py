"""Pre-compute the gas dynamics lookup table.

Port of generate_r_parallel.m, solve_r_function.m, and solve_function.m
from the 2023 revised solver.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from ..friction import fanning_friction_factor
from ..physics import (
    vapor_pressure,
    evaporation_rate_simple,
    find_evap_surface,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching solve_function.m)
# ---------------------------------------------------------------------------
_RG = 8.341 / 0.018
_SIGMA = 5.67e-8


def _solve_function(
    Tb: float,
    depth: float,
    width: float,
    r: float,
    kt: float = 2.4,
    lv: float = 2.8e6,
    g: float = 0.113,
    Te: float = 68.0,
    dz_step: float = 0.1,
    friction_model: str = "constant",
    Cf_constant: float = 0.002,
    mu_vapor: float = 8.0e-6,
    roughness: float = 0.0,
    C_lam: float = 96.0,
) -> tuple[float, float, float, float, float, float]:
    """Integrate the gas column for a given (Tb, depth, width, r).

    Returns (phi_to_zero, rho_to_zero, mach_top, phi_top, rho_top, phi0).
    """
    rg = _RG
    ec = vapor_pressure(273.15) / np.sqrt(2.0 * np.pi * rg)
    bv = lv / rg

    phi = (1.0 - r) * evaporation_rate_simple(ec, bv, Tb)
    phi0 = phi
    rho0 = vapor_pressure(Tb) / (rg * Tb) * r

    z_f = np.arange(0, depth + dz_step, dz_step)
    nz = len(z_f)
    f = np.zeros(nz)
    phi_to_zero = depth
    flag = False
    f_now = 0.0

    for i in range(1, nz):
        d_from_top = depth - z_f[i]
        if d_from_top < 1e-10:
            d_from_top = 1e-10
        ev_now = find_evap_surface(Tb, Te, _SIGMA, d_from_top, kt, lv) / width
        f_now += ev_now * (z_f[i] - z_f[i - 1])
        f[i] = f_now
        if (phi + f_now < 0) and not flag:
            phi_to_zero = z_f[i]
            flag = True

    def _Cf(v_mag, dn):
        return fanning_friction_factor(
            friction_model, v_mag, width, rho=dn, mu=mu_vapor,
            roughness=roughness, Cf_constant=Cf_constant, C_lam=C_lam)

    # Density integration (RK4)
    rho = np.zeros(nz)
    M = np.zeros(nz)
    rho[0] = rho0
    M[0] = phi / rho0 / np.sqrt(1.33 * rg * Tb)
    rho_to_zero = depth

    for i in range(1, nz):
        dn = rho[i - 1]
        v = (phi + f[i]) / dn
        M[i] = v / np.sqrt(1.33 * rg * Tb)
        Cd = _Cf(v, dn)
        k1 = (2.0 * Cd * dn * v**2 / width + dn * g) / (v**2 - rg * Tb)

        if i < nz - 1:
            dn2 = rho[i - 1] + (z_f[i] - z_f[i - 1]) * 0.5 * k1
            v2 = (phi + 0.5 * (f[i] + f[i + 1])) / dn2
            Cd2 = _Cf(v2, dn2)
            k2 = (2.0 * Cd2 * dn2 * v2**2 / width + dn2 * g) / (v2**2 - rg * Tb)

            dn3 = rho[i - 1] + (z_f[i] - z_f[i - 1]) * 0.5 * k2
            v3 = (phi + 0.5 * (f[i] + f[i + 1])) / dn3
            Cd3 = _Cf(v3, dn3)
            k3 = (2.0 * Cd3 * dn3 * v3**2 / width + dn3 * g) / (v3**2 - rg * Tb)

            dn4 = rho[i - 1] + (z_f[i] - z_f[i - 1]) * k3
            v4 = (phi + f[i + 1]) / dn4
            Cd4 = _Cf(v4, dn4)
            k4 = (2.0 * Cd4 * dn4 * v4**2 / width + dn4 * g) / (v4**2 - rg * Tb)
        else:
            k2 = k1
            k3 = k2
            k4 = k3

        rho[i] = rho[i - 1] + (z_f[i] - z_f[i - 1]) / 6.0 * (k1 + k4 + 2.0 * k2 + 2.0 * k3)

        if rho[i] < 0 or (rho[i] - rho[i - 1] > 0):
            rho_to_zero = z_f[i]
            break

    if rho[-1] > 0 and (phi + f[-1]) > 0:
        mach_top = M[nz - 1]
        phi_top = phi + f[nz - 1]
        rho_top = rho[nz - 1]
    else:
        mach_top = 0.0
        phi_top = 0.0
        rho_top = 0.0

    return phi_to_zero, rho_to_zero, mach_top, phi_top, rho_top, phi0


def solve_r_function(
    Tb: float, depth: float, width: float, tol: float = 1e-4,
) -> tuple[float, float, float, float]:
    """Bisection to find *r* for given (Tb, depth, width).

    Returns (r, phi_top, rho_top, phi0).
    """
    r_l = 1e-5
    r_r = 1.0 - 1e-5
    phi_l = phi_r = rho_l = rho_r = phi0_l = phi0_r = 0.0

    while abs(r_l - r_r) > tol:
        r_m = 0.5 * (r_l + r_r)
        phi_to_zero, rho_to_zero, mach_top, phi_top, rho_top, phi0 = \
            _solve_function(Tb, depth, width, r_m)

        if mach_top == 0:
            if rho_to_zero < phi_to_zero:
                r_l = r_m
                phi_l, rho_l, phi0_l = phi_top, rho_top, phi0
            else:
                r_r = r_m
                phi_r, rho_r, phi0_r = phi_top, rho_top, phi0
        else:
            if mach_top > 1.6:
                r_l = r_m
                phi_l, rho_l, phi0_l = phi_top, rho_top, phi0
            else:
                r_r = r_m
                phi_r, rho_r, phi0_r = phi_top, rho_top, phi0

    r = 0.5 * (r_l + r_r)
    return (r,
            0.5 * (phi_l + phi_r),
            0.5 * (rho_l + rho_r),
            0.5 * (phi0_l + phi0_r))


def _solve_single(delta_val, depth_val, Tb_val):
    """Helper for parallel map."""
    return solve_r_function(Tb_val, depth_val, delta_val)


def generate_r_table(
    deltas: np.ndarray,
    depths: np.ndarray,
    Tb_arr: Optional[np.ndarray] = None,
    output_path: str = "r_rec.npz",
    n_jobs: int = -1,
) -> None:
    """Pre-compute gas dynamics lookup table and save to disk.

    Port of generate_r_parallel.m / generate_function.

    Parameters
    ----------
    deltas : 1-D array of crack widths (m)
    depths : 1-D array of water column depths (m)
    Tb_arr : 1-D array of base temperatures (K); default [272, 273.1501]
    output_path : file to save .npz lookup table
    n_jobs : number of parallel workers (-1 = all CPUs)
    """
    if Tb_arr is None:
        Tb_arr = np.array([272.0, 273.1501])

    nd = len(deltas)
    nD = len(depths)
    nT = len(Tb_arr)

    r_rec = np.zeros((nd, nD, nT))
    phi_rec = np.zeros((nd, nD, nT))
    rho_rec = np.zeros((nd, nD, nT))
    phi0_rec = np.zeros((nd, nD, nT))

    tasks = []
    for i, d in enumerate(deltas):
        for j, D in enumerate(depths):
            for k, Tb in enumerate(Tb_arr):
                tasks.append((i, j, k, d, D, Tb))

    logger.info("Generating lookup table: %d tasks (%d widths x %d depths x %d temps)",
                len(tasks), nd, nD, nT)

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_solve_single)(d, D, Tb) for (_, _, _, d, D, Tb) in tasks
    )

    for idx, (i, j, k, *_) in enumerate(tasks):
        r_rec[i, j, k] = results[idx][0]
        phi_rec[i, j, k] = results[idx][1]
        rho_rec[i, j, k] = results[idx][2]
        phi0_rec[i, j, k] = results[idx][3]

    np.savez(output_path,
             r_rec=r_rec, phi_rec=phi_rec, rho_rec=rho_rec, phi0_rec=phi0_rec,
             Tb=Tb_arr, delta=deltas, depth=depths)
    logger.info("Lookup table saved to %s", output_path)


def build_lookup_from_bounds(
    depth_range: tuple[float, float],
    width_range: tuple[float, float],
    n_grid: int = 25,
    Tb: Optional[np.ndarray] = None,
    output_path: str = "r_rec.npz",
    n_jobs: int = -1,
    depth_margin: float = 0.1,
    width_margin: float = 0.1,
    width_spacing: str = "log",
) -> str:
    """Build a gas dynamics lookup table covering the given ranges.

    Adds fractional margins to both axes so interpolation stays in-bounds
    even at the extremes of the liquid dynamics results.

    Parameters
    ----------
    width_spacing : ``"log"`` (default) concentrates grid points at narrow
        widths where the gas dynamics vary most.  ``"linear"`` uses uniform
        spacing.

    Returns the output file path.
    """
    d_lo, d_hi = depth_range
    w_lo, w_hi = width_range

    d_span = max(d_hi - d_lo, 1.0)
    w_span = max(w_hi - w_lo, 1e-3)

    d_lo = max(d_lo - depth_margin * d_span, 0.1)
    d_hi = d_hi + depth_margin * d_span
    w_lo = max(w_lo - width_margin * w_span, 0.02)
    w_hi = w_hi + width_margin * w_span

    depths = np.linspace(d_lo, d_hi, n_grid)

    if width_spacing == "log":
        deltas = np.geomspace(w_lo, w_hi, n_grid)
    else:
        deltas = np.linspace(w_lo, w_hi, n_grid)

    logger.info("Lookup bounds: depth [%.1f, %.1f] m (%d pts), "
                "width [%.4f, %.4f] m (%d pts, %s)",
                d_lo, d_hi, n_grid, w_lo, w_hi, n_grid, width_spacing)

    if Tb is None:
        Tb = np.array([272.0, 273.1501])
    else:
        Tb = np.asarray(Tb, dtype=float)

    generate_r_table(deltas, depths, Tb_arr=Tb,
                     output_path=output_path, n_jobs=n_jobs)
    return output_path
