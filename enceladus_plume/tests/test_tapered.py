"""Parity tests for the z-dependent (tapered) liquid solver.

The tapered kinematics and momentum derivative must reduce exactly to the
existing uniform-width solver when the gap is spatially constant.
"""

import numpy as np
import pytest

from enceladus_plume.config import load_config
from enceladus_plume.liquid_dynamics.helpers import vel_now, friction, additional_term
from enceladus_plume.liquid_dynamics.solver import _derivative, liquid_dynamics
from enceladus_plume.liquid_dynamics.tapered import (
    vel_profile_tapered,
    friction_tapered,
    additional_term_tapered,
    derivative_tapered,
    liquid_dynamics_tapered,
)
from enceladus_plume.utils import build_width_series

_FRIC = dict(model="constant", Cf_constant=0.004, rho=1000.0, mu=1.8e-3,
             roughness=0.0, C_lam=96.0)

_CASES = [
    (0.3, 50.0, 5000.0, 0.05, 1e-6, 1e-10),
    (-0.2, -100.0, 8000.0, 0.02, -5e-6, 2e-10),
    (0.1, 10.0, 2000.0, 0.03, 3e-6, -1e-10),
]


def _uniform_arrays(h, L, w, dwdt, dwdt2, npts=1000):
    zs = np.linspace(-L, h, npts)
    return (zs, np.full(npts, w), np.full(npts, dwdt), np.full(npts, dwdt2))


@pytest.mark.parametrize("v0,h,L,w,dwdt,dwdt2", _CASES)
def test_kinematics_reduce_to_uniform(v0, h, L, w, dwdt, dwdt2):
    zs, dz, d1, d2 = _uniform_arrays(h, L, w, dwdt, dwdt2)
    _, vp_ref = vel_now(v0, h, L, w, dwdt, len(zs))
    vp = vel_profile_tapered(v0, zs, dz, d1)
    assert np.allclose(vp, vp_ref, atol=1e-12)
    assert np.isclose(friction_tapered(zs, vp, dz, **_FRIC),
                      friction(zs, vp_ref, w, **_FRIC), rtol=1e-10)
    assert np.isclose(additional_term_tapered(zs, v0, vp, dz, d1, d2),
                      additional_term(zs, w, dwdt, dwdt2), rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("v0,h,L,w,dwdt,dwdt2", _CASES)
def test_derivative_reduces_to_uniform(v0, h, L, w, dwdt, dwdt2):
    ref = _derivative(v0, h, L, 0.113, w, dwdt, dwdt2, 1000, _FRIC)
    zs, dz, d1, d2 = _uniform_arrays(h, L, w, dwdt, dwdt2)
    tap = derivative_tapered(v0, h, 0.113, zs, dz, d1, d2, _FRIC)
    assert np.isclose(tap[0], ref[0], rtol=1e-11, atol=1e-15)  # dv0/dt
    assert np.isclose(tap[1], ref[1], rtol=1e-12, atol=1e-15)  # dh/dt


def test_tapered_neck_accelerates_flow():
    """A narrowing taper must speed the flow at the narrow top (continuity)."""
    L, w, v0, dwdt = 5000.0, 0.05, 0.3, 1e-6
    zs = np.linspace(-L, 50.0, 500)
    d1 = np.full(len(zs), dwdt)
    v_uniform = vel_profile_tapered(v0, zs, np.full(len(zs), w), d1)
    v_taper = vel_profile_tapered(v0, zs, np.linspace(w, w / 5, len(zs)), d1)
    assert abs(v_taper[-1]) > abs(v_uniform[-1])


@pytest.mark.slow
def test_solver_parity_uniform():
    """liquid_dynamics_tapered with no ice must reproduce liquid_dynamics."""
    cfg = load_config()
    P = cfg.physical.orbital_period
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 400.0
    L = 5000.0
    cfg.physical.equilibrium_depth = L
    t_in = np.arange(100, P + 1, 200.0)
    w_in = build_width_series(t_in, 1.3, 0.05, orbital_period=P,
                              forcing_model="single-cosine")
    _, hr, _, vr = liquid_dynamics(w_in, t_in, L, cfg)
    _, ht, _, vt = liquid_dynamics_tapered(w_in, t_in, L, cfg)
    assert np.max(np.abs(ht - hr)) < 1e-3 * max(np.max(np.abs(hr)), 1.0)
    assert np.max(np.abs(vt - vr)) < 1e-6
