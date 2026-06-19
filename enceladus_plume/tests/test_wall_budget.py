"""Tests for the wall ice mass-budget diagnostic (stage A)."""

import numpy as np
import pytest

from enceladus_plume.config import load_config
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics
from enceladus_plume.utils import build_width_series
from enceladus_plume.wall_budget import wall_mass_budget

pytestmark = pytest.mark.slow  # each test runs a full liquid solve


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


def test_budget_shapes_and_signs():
    cfg, t, h, w, v = _small_case()
    r = np.full_like(t, np.nan)  # uncapped: no lookup dependency
    res = wall_mass_budget(t, h, w, v, r, cfg, Tb=272.6, n_z=400)

    n = len(res.zeta)
    assert res.deposition.shape == (n,)
    assert res.melt.shape == (n,)
    # grid spans floor to surface
    assert res.zeta[0] == 0.0
    assert np.isclose(res.zeta[-1], res.surface)
    # both fluxes are non-negative; net is their difference
    assert np.all(res.deposition >= 0)
    assert np.all(res.melt >= 0)
    assert np.allclose(res.net_sigma, res.deposition - res.melt)
    assert np.allclose(res.net_thickness, res.net_sigma / res.rho_ice)


def test_deposition_above_melt_below():
    """Condensation lives above the water level; melt lives below it."""
    cfg, t, h, w, v = _small_case()
    r = np.full_like(t, np.nan)
    res = wall_mass_budget(t, h, w, v, r, cfg, Tb=272.6, n_z=400)
    L = res.L
    # essentially all deposition is above the equilibrium water level
    assert np.trapz(res.deposition[res.zeta >= L], res.zeta[res.zeta >= L]) > 0
    assert np.allclose(res.deposition[res.zeta < L - 50.0], 0.0)
    # melt only occurs below the water level
    assert np.allclose(res.melt[res.zeta > L + 50.0], 0.0)


def test_choke_cap_never_exceeds_uncapped():
    """A finite choke cap can only reduce (or equal) deposition vs uncapped."""
    cfg, t, h, w, v = _small_case()
    res_uncapped = wall_mass_budget(t, h, w, v, np.full_like(t, np.nan),
                                    cfg, Tb=272.6, n_z=400)
    # a strong cap: r close to 1 -> tiny inlet flux phi -> early choke
    res_capped = wall_mass_budget(t, h, w, v, np.full_like(t, 0.999),
                                  cfg, Tb=272.6, n_z=400)
    tot_un = np.trapz(res_uncapped.deposition, res_uncapped.zeta)
    tot_cap = np.trapz(res_capped.deposition, res_capped.zeta)
    assert tot_cap <= tot_un + 1e-9
