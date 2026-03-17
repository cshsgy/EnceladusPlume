"""Full coupled solver workflow (2022 paper version).

Port of composed_main_func.m -- runs crack dynamics, then the coupled
gas-dynamics + heat-diffusion loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import Config, load_config
from ..physics import surface_temperature
from ..liquid_dynamics.solver import liquid_dynamics_2022
from ..gas_dynamics.solver import gas_dynamics_march_constrained
from ..heat_diffusion.solver import heat_diffusion_step
from ..utils import width_new, simplify_crack_dynamics_data, save_results

logger = logging.getLogger(__name__)


def run_full_solver(
    wmin: Optional[float] = None,
    wmaxmin: Optional[float] = None,
    depth: Optional[float] = None,
    cfg: Optional[Config] = None,
    output_path: str = "run_results.npz",
) -> dict[str, np.ndarray]:
    """Execute the full coupled solver.

    Parameters
    ----------
    wmin     : minimum crack width (m); overrides config if given
    wmaxmin  : wmax/wmin ratio; overrides config if given
    depth    : crack length / equilibrium water depth (m); overrides config
    cfg      : Config object (loaded from default.yaml if None)
    output_path : where to save the results

    Returns
    -------
    Dictionary of result arrays.
    """
    if cfg is None:
        cfg = load_config()

    wmin = wmin if wmin is not None else cfg.full_solver.wmin
    wmaxmin = wmaxmin if wmaxmin is not None else cfg.full_solver.wmaxmin
    depth = depth if depth is not None else cfg.full_solver.depth
    L = depth

    phys = cfg.physical
    hdp = cfg.heat_diffusion
    gp = cfg.gas_dynamics

    Period = phys.orbital_period
    fp = cfg.friction

    Kappa = phys.thermal_diffusivity
    K = phys.thermal_conductivity
    Lv = phys.latent_heat
    G = phys.gravity
    Te = phys.effective_surface_temp
    rg = phys.gas_constant_vapor

    Dx = hdp.dx
    Max_x = hdp.max_x
    Nz = hdp.n_vertical_levels
    Periods_Run = hdp.periods_to_run

    # ------------------------------------------------------------------
    # Phase 1: Crack dynamics
    # ------------------------------------------------------------------
    logger.info("Phase 1: solving crack (liquid) dynamics ...")
    t_rec, v_rec, h_rec, w_rec = liquid_dynamics_2022(
        L, wmin, wmaxmin, width_new, cfg)

    # Simplify to one period
    t_rec, h_rec, w_rec = simplify_crack_dynamics_data(t_rec, Period, h_rec, w_rec)
    D = L / 10.0  # crack depth = 10% of L

    if np.max(h_rec) > D:
        logger.error("Water reaches the surface -- aborting.")
        return {}

    Dt = 0.1 / Kappa * Dx**2

    # ------------------------------------------------------------------
    # Phase 2: Heat diffusion + gas dynamics
    # ------------------------------------------------------------------
    logger.info("Phase 2: coupled gas dynamics + heat diffusion ...")

    z_wet = np.linspace(np.min(h_rec), D, Nz + 1)[:-1]
    dz = (np.max(z_wet) - np.min(z_wet)) / (Nz - 1) if Nz > 1 else 1.0
    xs = np.arange(0.0, Max_x + Dx, Dx)
    nx = len(xs)

    # Initialise wall temperature field
    T = np.ones((Nz, nx)) * 273.15
    T[z_wet > np.max(h_rec), :] -= 5.0
    for i in range(Nz):
        Ts = surface_temperature(Te, T[i, 0], D - z_wet[i] + xs[-1], K)
        if Ts < 0:
            Ts = Te
        T[i, :] -= xs * (T[i, 0] - Ts) * 2.0 / np.pi / (D - z_wet[i] + xs[-1])

    phi_rec_list: list[float] = []
    r_rec_list: list[float] = [0.6]
    time_rec_list: list[float] = []
    width_rec_list: list[float] = []

    t = 0.0
    iteration = 0

    while t < Periods_Run * Period:
        time_rec_list.append(t)
        if iteration % 200 == 0:
            logger.info("Heat-diffusion iteration %d, t = %.1f / %.1f",
                        iteration, t, Periods_Run * Period)

        height = float(np.interp(t % Period, t_rec, h_rec))
        cur_depth = D - height
        cur_width = float(np.interp(t % Period, t_rec, w_rec))
        width_rec_list.append(cur_width)

        zs = np.linspace(0, cur_depth, Nz)
        T2 = np.interp(zs + height, z_wet, T[:, 1])

        PhiTop, Tw, Ev, r = gas_dynamics_march_constrained(
            cur_width, zs, T2, K, Lv, G, Dx,
            r_rec_list[-1], float(np.max(h_rec)),
            max_change=gp.constrained_max_change,
            tol=gp.bisection_tol, rg=rg, gamma_eff=gp.gamma_eff,
            friction_model=fp.gas_model,
            Cf_constant=fp.gas_Cf_constant,
            mu_vapor=phys.vapor_viscosity,
            roughness=fp.roughness,
            C_lam=fp.C_lam,
        )
        phi_rec_list.append(PhiTop)
        r_rec_list.append(r)

        # Interpolate Tw and Ev back to z_wet grid
        Tw_wet = np.interp(z_wet, zs + height, Tw)
        Ev_wet = np.interp(z_wet, zs + height, Ev)
        Tw_wet[z_wet < height] = 273.15
        Ev_wet[z_wet < height] = 0.0

        # Heat diffusion at each vertical level
        for ii in range(Nz):
            T[ii, :] = heat_diffusion_step(
                z_wet[ii], xs, height, cur_depth,
                T[ii, :], Tw_wet[ii], Kappa, K, Te, Dx, Dt)

        t += Dt
        iteration += 1

    results = {
        "Period": np.array(Period),
        "time_rec": np.array(time_rec_list),
        "phi_rec": np.array(phi_rec_list),
        "width_rec": np.array(width_rec_list),
        "r_rec": np.array(r_rec_list),
        "t_rec": t_rec,
        "h_rec": h_rec,
    }

    save_results(output_path, **results)
    logger.info("Results saved to %s", output_path)
    return results
