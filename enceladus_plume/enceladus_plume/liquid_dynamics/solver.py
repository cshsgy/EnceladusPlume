"""Liquid dynamics solver using scipy.integrate.solve_ivp.

Port of main_func.m (2023) and the crack-dynamics portion of
composed_main_func.m (2022).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ..config import Config, FrictionParams, LiquidDynamicsParams, PhysicalParams
from .helpers import vel_now, friction, additional_term

logger = logging.getLogger(__name__)


def _derivative(v: float, h: float, L: float, g: float,
                w: float, dwdt: float, dwdt2: float,
                npts: int, fric_kw: dict) -> tuple[float, float]:
    """Compute dv/dt and dh/dt for the liquid column.

    Parameters
    ----------
    fric_kw : keyword arguments forwarded to :func:`friction`.

    Returns (dvdt, dhdt).
    """
    col_height = h + L
    if col_height <= 0.0:
        return 0.0, 0.0

    zs, v_now_arr = vel_now(v, h, L, w, dwdt, npts)
    fric = friction(zs, v_now_arr, w, **fric_kw)
    rhs = -0.5 * (v_now_arr[0] ** 2 + v_now_arr[-1] ** 2) - g * h - fric
    add_term = additional_term(zs, w, dwdt, dwdt2)
    rhs = rhs - add_term
    dvdt = rhs / col_height
    dhdt = float(v_now_arr[-1])
    return dvdt, dhdt


def _make_clamped_rhs(raw_rhs, D: float, L: float):
    """Wrap *raw_rhs* with smooth overflow/floor barriers.

    Uses a stiff exponential restoring force that activates within a thin
    transition layer near h = +D (surface) and h = -L (ice floor).
    The barrier is smooth so explicit ODE solvers remain efficient.
    """
    _DELTA = 10.0    # m, transition layer thickness
    _K = 10.0        # 1/s², barrier stiffness (restoring acceleration)
    _DAMP_COEF = 5.0 # 1/s, velocity damping rate in the barrier zone

    def clamped_rhs(t, y):
        v, h = float(y[0]), float(y[1])
        dvdt, dhdt = raw_rhs(t, y)

        if h > D - _DELTA:
            pen = max(0.0, (h - (D - _DELTA)) / _DELTA)
            pen2 = pen * pen
            dhdt = dhdt * (1.0 - pen2) if dhdt > 0 else dhdt
            dvdt -= _K * pen2 * (h - D + _DELTA) + _DAMP_COEF * pen2 * v
        elif h < -L + _DELTA:
            pen = max(0.0, ((-L + _DELTA) - h) / _DELTA)
            pen2 = pen * pen
            dhdt = dhdt * (1.0 - pen2) if dhdt < 0 else dhdt
            dvdt += _K * pen2 * (-L + _DELTA - h) - _DAMP_COEF * pen2 * v

        return [dvdt, dhdt]

    return clamped_rhs


# ---------------------------------------------------------------------------
# 2023-style solver (prescribed w(t) time series)
# ---------------------------------------------------------------------------

def liquid_dynamics(
    w_in: np.ndarray,
    t_in: np.ndarray,
    L: float,
    cfg: Optional[Config] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve liquid dynamics driven by a prescribed crack-width time series.

    This is the direct port of the 2023 ``main_func.m``
    (``liquid_dynamics_func``).

    Parameters
    ----------
    w_in  : 1-D array of crack widths (m), same length as *t_in*.
    t_in  : 1-D array of times (s) within one orbital period.
    L     : equilibrium water depth (m).
    cfg   : optional Config; defaults are used if None.

    Returns
    -------
    w_rec, h_rec, t_rec, v_rec : recorded width, water level, time, and
        reference velocity arrays.
    """
    if cfg is None:
        from ..config import load_config
        cfg = load_config()

    phys: PhysicalParams = cfg.physical
    lp: LiquidDynamicsParams = cfg.liquid_dynamics
    fp: FrictionParams = cfg.friction
    P = phys.orbital_period
    g = phys.gravity

    fric_kw = dict(
        model=fp.liquid_model,
        Cf_constant=fp.liquid_Cf_constant,
        rho=phys.liquid_density,
        mu=phys.liquid_viscosity,
        roughness=fp.roughness,
        C_lam=fp.C_lam,
    )

    w_ext = np.concatenate([w_in, w_in, w_in])
    t_ext = np.concatenate([t_in - P, t_in, t_in + P])
    dwdt_ext = np.gradient(w_ext, t_ext)
    dwdt2_ext = np.gradient(dwdt_ext, t_ext)

    D = L / 10.0
    t_stop = P * lp.n_periods

    def _interp(arr, t_now):
        return float(np.interp(t_now % P, t_ext, arr))

    def raw_rhs(t, y):
        v, h = float(y[0]), float(y[1])
        w = _interp(w_ext, t)
        dwdt = _interp(dwdt_ext, t)
        dwdt2 = _interp(dwdt2_ext, t)
        dvdt, dhdt = _derivative(v, h, L, g, w, dwdt, dwdt2,
                                 lp.npts_velocity, fric_kw)
        return [dvdt, dhdt]

    rhs = _make_clamped_rhs(raw_rhs, D, L)
    n_output = max(int(t_stop / 10.0), 2000)
    t_eval = np.linspace(0.0, t_stop, n_output)

    sol = solve_ivp(
        rhs,
        (0.0, t_stop),
        [0.0, 0.0],
        method=lp.ode_method,
        rtol=lp.rtol,
        atol=lp.atol,
        max_step=lp.max_step,
        t_eval=t_eval,
    )

    t_all = sol.t
    v_rec = sol.y[0]
    h_rec = sol.y[1]
    w_rec = np.array([_interp(w_ext, t) for t in t_all])
    return w_rec, h_rec, t_all, v_rec


# ---------------------------------------------------------------------------
# 2022-style solver (analytic width function)
# ---------------------------------------------------------------------------

def liquid_dynamics_2022(
    L: float,
    wmin: float,
    wmaxmin: float,
    width_func,
    cfg: Optional[Config] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve liquid dynamics with an analytic width function (2022 style).

    Parameters
    ----------
    L         : equilibrium water depth (m)
    wmin      : minimum crack width (m)
    wmaxmin   : ratio wmax / wmin
    width_func: callable(z, t, wmaxmin, wmin) -> width
    cfg       : optional Config

    Returns
    -------
    t_rec, v_rec, h_rec, w_rec
    """
    if cfg is None:
        from ..config import load_config
        cfg = load_config()

    phys = cfg.physical
    lp = cfg.liquid_dynamics
    fp = cfg.friction
    g = phys.gravity
    P = phys.orbital_period

    fric_kw = dict(
        model=fp.liquid_model,
        Cf_constant=fp.liquid_Cf_constant,
        rho=phys.liquid_density,
        mu=phys.liquid_viscosity,
        roughness=fp.roughness,
        C_lam=fp.C_lam,
    )

    _eps_t = 0.01

    def _get_w_derivs(t_now):
        w = width_func(0, t_now, wmaxmin, wmin)
        wp = width_func(0, t_now + _eps_t, wmaxmin, wmin)
        wm = width_func(0, t_now - _eps_t, wmaxmin, wmin)
        dwdt = (wp - wm) / (2 * _eps_t)
        dwdt2 = (wp - 2 * w + wm) / (_eps_t ** 2)
        return w, dwdt, dwdt2

    D = L / 10.0
    t_stop = P * 100

    def raw_rhs(t, y):
        v, h = float(y[0]), float(y[1])
        w, dwdt, dwdt2 = _get_w_derivs(t)
        dvdt, dhdt = _derivative(v, h, L, g, w, dwdt, dwdt2,
                                 lp.npts_velocity, fric_kw)
        return [dvdt, dhdt]

    rhs = _make_clamped_rhs(raw_rhs, D, L)
    n_output = max(int(t_stop / 10.0), 2000)
    t_eval = np.linspace(0.0, t_stop, n_output)

    sol = solve_ivp(
        rhs,
        (0.0, t_stop),
        [0.0, 0.0],
        method=lp.ode_method,
        rtol=lp.rtol,
        atol=lp.atol,
        max_step=lp.max_step,
        t_eval=t_eval,
    )

    t_all = sol.t
    v_rec = sol.y[0]
    h_rec = sol.y[1]
    w_rec = np.array([width_func(0, t, wmaxmin, wmin) for t in t_all])
    return t_all, v_rec, h_rec, w_rec


# ---------------------------------------------------------------------------
# Overflow post-processing
# ---------------------------------------------------------------------------

_BARRIER_DELTA = 10.0  # must match _DELTA in _make_clamped_rhs


def compute_overflow_rate(
    t_rec: np.ndarray,
    v_rec: np.ndarray,
    h_rec: np.ndarray,
    w_in: np.ndarray,
    t_in: np.ndarray,
    L: float,
    cfg: Optional[Config] = None,
) -> np.ndarray:
    """Compute the overflow velocity dh/dt lost to the barrier at each output point.

    Re-evaluates the *raw* (unclamped) derivative at every ``(t, v, h)``
    output point and returns the positive overflow component (m/s, >= 0).
    Multiply by ``rho_water * w`` to get an overflow mass-flux per unit
    crack length.

    Parameters
    ----------
    t_rec, v_rec, h_rec : solution arrays from :func:`liquid_dynamics`.
    w_in, t_in           : the *input* width/time arrays (one period).
    L                    : equilibrium water depth (m).
    cfg                  : Config (uses defaults if None).

    Returns
    -------
    overflow_dhdt : 1-D array (same length as *t_rec*), non-negative.
    """
    if cfg is None:
        from ..config import load_config
        cfg = load_config()

    phys = cfg.physical
    lp = cfg.liquid_dynamics
    fp = cfg.friction
    P = phys.orbital_period
    g = phys.gravity
    D = L / 10.0

    fric_kw = dict(
        model=fp.liquid_model,
        Cf_constant=fp.liquid_Cf_constant,
        rho=phys.liquid_density,
        mu=phys.liquid_viscosity,
        roughness=fp.roughness,
        C_lam=fp.C_lam,
    )

    w_ext = np.concatenate([w_in, w_in, w_in])
    t_ext = np.concatenate([t_in - P, t_in, t_in + P])
    dwdt_ext = np.gradient(w_ext, t_ext)
    dwdt2_ext = np.gradient(dwdt_ext, t_ext)

    def _interp(arr, t_now):
        return float(np.interp(t_now % P, t_ext, arr))

    n = len(t_rec)
    overflow = np.zeros(n)

    for i in range(n):
        h = float(h_rec[i])
        if h <= D - _BARRIER_DELTA:
            continue

        t = float(t_rec[i])
        v = float(v_rec[i])
        w = _interp(w_ext, t)
        dwdt = _interp(dwdt_ext, t)
        dwdt2 = _interp(dwdt2_ext, t)

        _dvdt_raw, dhdt_raw = _derivative(
            v, h, L, g, w, dwdt, dwdt2, lp.npts_velocity, fric_kw)

        if dhdt_raw <= 0.0:
            continue

        pen = max(0.0, (h - (D - _BARRIER_DELTA)) / _BARRIER_DELTA)
        pen2 = pen * pen
        overflow[i] = dhdt_raw * pen2

    return overflow
