"""Parity checks between the C++ core and the pure-Python reference.

Skipped automatically when the ``_enceladus_core`` extension has not been
built (see cpp/). When built, every bound function must match its Python
counterpart to within floating-point tolerance.
"""

import math

import pytest

from enceladus_plume import friction as py_friction
from enceladus_plume import physics as py_physics
from enceladus_plume._native import CORE, HAVE_NATIVE
from enceladus_plume.gas_dynamics import interpolator as py_gas
from enceladus_plume.liquid_dynamics import helpers as py_liquid_helpers

pytestmark = pytest.mark.skipif(not HAVE_NATIVE, reason="C++ core not built")

_CASES = [
    # (model, v, w, rho, mu)
    ("constant", 1.0, 0.05, 1000.0, 1.8e-3),
    ("laminar", 0.5, 0.01, 1000.0, 1.8e-3),
    ("laminar", 3.0, 0.005, 1000.0, 1.8e-3),
    ("churchill", 0.5, 0.01, 1000.0, 1.8e-3),
    ("churchill", 5.0, 0.002, 1000.0, 1.8e-3),
    ("churchill", 1e-6, 0.05, 1000.0, 1.8e-3),  # near-zero velocity edge
]


@pytest.mark.parametrize("model,v,w,rho,mu", _CASES)
def test_fanning_parity(model, v, w, rho, mu):
    expected = py_friction.fanning_friction_factor(model, v, w, rho=rho, mu=mu)
    got = CORE.fanning_friction_factor(model, v, w, rho=rho, mu=mu)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-15)


def test_helpers_parity():
    assert CORE.hydraulic_diameter(0.03) == py_friction.hydraulic_diameter(0.03)
    assert math.isclose(
        CORE.reynolds_number(1000.0, 2.0, 0.06, 1.8e-3),
        py_friction.reynolds_number(1000.0, 2.0, 0.06, 1.8e-3),
        rel_tol=1e-12,
    )


# --- physics --------------------------------------------------------------

def test_physics_parity():
    assert math.isclose(CORE.vapor_pressure(260.0),
                        py_physics.vapor_pressure(260.0), rel_tol=1e-13)
    assert math.isclose(
        CORE.find_evap_surface(272.0, 68.0, 5.67e-8, 100.0, 2.4, 2.8e6),
        py_physics.find_evap_surface(272.0, 68.0, 5.67e-8, 100.0, 2.4, 2.8e6),
        rel_tol=1e-12,
    )


# --- liquid derivative ----------------------------------------------------

def _py_derivative(v, h, L, g, w, dwdt, dwdt2, npts, fric_kw):
    """Pure-Python reference (mirrors solver.py::_derivative)."""
    col_height = h + L
    if col_height <= 0.0:
        return 0.0, 0.0
    zs, vp = py_liquid_helpers.vel_now(v, h, L, w, dwdt, npts)
    fric = py_liquid_helpers.friction(zs, vp, w, **fric_kw)
    rhs = -0.5 * (vp[-1] ** 2 - vp[0] ** 2) - g * h - fric
    rhs -= py_liquid_helpers.additional_term(zs, w, dwdt, dwdt2)
    return rhs / col_height, float(vp[-1])


_LIQUID_CASES = [
    # (v, h, L, w, dwdt, dwdt2, model)
    (0.3, 50.0, 5000.0, 0.05, 1e-6, 1e-10, "constant"),
    (-0.2, -100.0, 8000.0, 0.02, -5e-6, 2e-10, "constant"),
    (0.1, 10.0, 2000.0, 0.03, 3e-6, -1e-10, "churchill"),
]


@pytest.mark.parametrize("v,h,L,w,dwdt,dwdt2,model", _LIQUID_CASES)
def test_liquid_derivative_parity(v, h, L, w, dwdt, dwdt2, model):
    fric_kw = dict(model=model, Cf_constant=0.004, rho=1000.0, mu=1.8e-3,
                   roughness=0.0, C_lam=96.0)
    exp_dvdt, exp_dhdt = _py_derivative(v, h, L, 0.113, w, dwdt, dwdt2, 1000, fric_kw)
    got_dvdt, got_dhdt = CORE.liquid_derivative(
        v, h, L, 0.113, w, dwdt, dwdt2, 1000, model, 0.004, 1000.0, 1.8e-3,
        0.0, 96.0)
    assert math.isclose(got_dvdt, exp_dvdt, rel_tol=1e-10, abs_tol=1e-15)
    assert math.isclose(got_dhdt, exp_dhdt, rel_tol=1e-12, abs_tol=1e-15)


# --- gas solver -----------------------------------------------------------

@pytest.mark.parametrize("depth,width", [(200.0, 0.05), (500.0, 0.02)])
def test_gas_solve_r_parity(depth, width):
    py_r = py_gas.solve_r_function(272.0, depth, width)
    cc_r = tuple(CORE.solve_r_function(272.0, depth, width))
    # r and phi_top integrate to the same bisection result; rho_top can differ
    # at the 1e-12 level due to RK operation ordering.
    for got, exp in zip(cc_r, py_r):
        assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-14)
