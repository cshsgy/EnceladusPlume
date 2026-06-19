"""Tests for the stage-B self-consistent geometry evolution."""

import numpy as np
import pytest

from enceladus_plume.config import load_config
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics
from enceladus_plume.utils import build_width_series
from enceladus_plume.wall_geometry import evolve_geometry, evolve_geometry_coupled

pytestmark = pytest.mark.slow  # runs a full liquid solve


def test_coupled_fixed_dw_reaches_overflow():
    """At fixed absolute swing, sealing (smaller w_eff) drives water to overflow."""
    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 1
    cfg.liquid_dynamics.max_step = 600.0
    cfg.liquid_dynamics.npts_velocity = 120
    cfg.physical.equilibrium_depth = 5000.0
    res = evolve_geometry_coupled(cfg, delta_w=0.01, n_e=7, w_eff_max=0.05,
                                  w_floor=3e-3)
    # effective width sweeps high -> low; R_eff = 1 + dw/w_eff and grows
    assert res.w_eff[0] > res.w_eff[-1]
    assert np.allclose(res.R_eff, 1.0 + res.delta_w / res.w_eff)
    assert np.all(np.diff(res.R_eff) > 0)
    # water rise increases as the crack seals, and overflow is reached
    rise = res.water_max - res.L
    assert rise[-1] > rise[0]
    assert res.overflow
    assert res.w_eff[-1] <= res.w_eff_overflow <= res.w_eff[0]


def _small_case():
    cfg = load_config()
    P = cfg.physical.orbital_period
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 400.0
    L = 5000.0
    cfg.physical.equilibrium_depth = L
    t_in = np.arange(100, P + 1, 200.0)
    w_in = build_width_series(t_in, 1.3, 0.05, orbital_period=P,
                              forcing_model="single-cosine")
    w_rec, h_rec, t_rec, v_rec = liquid_dynamics(w_in, t_in, L, cfg)
    return cfg, t_rec, h_rec, w_rec, v_rec


def test_geometry_converges_and_pins_to_water_reach():
    cfg, t, h, w, v = _small_case()
    r = np.full_like(t, 0.8)  # constant inlet ratio (no lookup dependency)
    res = evolve_geometry(t, h, w, v, r, cfg, Tb=272.6, n_z=300, n_t=80,
                          cycles_per_iter=50.0, max_iter=500)

    assert res.converged
    assert np.all(res.e_ice >= 0.0)
    # water keeps the wall ice-free up to its reach height
    assert np.allclose(res.e_ice[res.zeta <= res.water_max_height], 0.0)
    # ice accumulates somewhere above the water line (sealing taper)
    assert np.any(res.e_ice[res.zeta > res.water_max_height] > 0.0)
    # the open channel cannot extend above where water reaches
    assert res.open_top_height <= res.water_max_height + 50.0


def test_no_deposition_leaves_crack_open():
    cfg, t, h, w, v = _small_case()
    r = np.full_like(t, np.nan)  # NaN disables deposition entirely
    res = evolve_geometry(t, h, w, v, r, cfg, Tb=272.6, n_z=300, n_t=80,
                          max_iter=50)
    assert res.converged
    assert np.allclose(res.e_ice, 0.0)
