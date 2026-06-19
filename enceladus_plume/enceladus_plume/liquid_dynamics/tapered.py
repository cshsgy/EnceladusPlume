"""z-dependent (tapered) crack kinematics for the full stage-B liquid solver.

The wall gap ``delta(z, t)`` is *prescribed* (tidal forcing plus the ice
lining), so the in-crack velocity follows algebraically from incompressible
continuity

    d(delta)/dt + d(delta v)/dz = 0
    =>  v(z) = [ delta(z_b) v0 - integral_{z_b}^{z} d(delta)/dt dz' ] / delta(z)

with ``z_b`` the crack floor and ``v0 = v(z_b)``. For a spatially uniform gap
these reduce to the existing helpers (``vel_now``/``friction``); that limit is
the regression target (see tests).

This module provides the kinematic pieces (velocity profile and the wall
friction integral). The momentum integral that advances ``v0(t)`` builds on
these and is added on top.
"""

from __future__ import annotations

import numpy as np

from ..friction import fanning_friction_factor

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of *y* over *x*, starting at 0."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def vel_profile_tapered(
    v0: float,
    zs: np.ndarray,
    delta_z: np.ndarray,
    dddt_z: np.ndarray,
) -> np.ndarray:
    """Velocity profile along a tapered crack from continuity.

    Parameters
    ----------
    v0      : velocity at the crack floor ``zs[0]``.
    zs      : 1-D grid of heights (ascending), floor at ``zs[0]``.
    delta_z : gap ``delta(z)`` at each grid point (m).
    dddt_z  : local wall-motion rate ``d(delta)/dt`` at each grid point (m/s).

    Returns the velocity at each grid point.
    """
    cum = _cumtrapz(dddt_z, zs)  # integral of d(delta)/dt from the floor
    return (delta_z[0] * v0 - cum) / delta_z


def additional_term_tapered(
    zs: np.ndarray,
    v0: float,
    v: np.ndarray,
    delta_z: np.ndarray,
    dddt_z: np.ndarray,
    ddddt_z: np.ndarray,
) -> float:
    """Integral of the explicit profile-unsteadiness term ``A(z)``.

    ``A(z)`` is the part of ``dv/dt`` not proportional to ``dv0/dt``; integrating
    it gives the generalized ``additional_term``. Derived from
    ``v = (delta(z_b) v0 - C)/delta(z)`` with ``C = int d(delta)/dt`` and using
    ``delta v = delta(z_b) v0 - C``:

        A(z) = [ d(delta)/dt|_{z_b} v0 - dC/dt - v d(delta)/dt ] / delta(z)

    For a uniform gap this reduces to ``((ddot/delta)^2 - ddot2/delta)(z-z_b)``,
    matching :func:`enceladus_plume.liquid_dynamics.helpers.additional_term`.
    """
    Cdot = _cumtrapz(ddddt_z, zs)  # d/dt of the continuity integral
    A = (dddt_z[0] * v0 - Cdot - v * dddt_z) / delta_z
    return float(_trapz(A[1:], zs[1:]))


def friction_tapered(
    zs: np.ndarray,
    v: np.ndarray,
    delta_z: np.ndarray,
    *,
    model: str = "constant",
    Cf_constant: float = 0.004,
    rho: float = 1000.0,
    mu: float = 1.8e-3,
    roughness: float = 0.0,
    C_lam: float = 96.0,
) -> float:
    """Integral of wall friction ``2 Cf(z)/delta(z) * v|v|`` along a tapered crack."""
    npts = len(zs)
    dfdt = np.empty(npts)
    for i in range(npts):
        if model == "constant":
            Cf_i = Cf_constant
        else:
            Cf_i = fanning_friction_factor(
                model, v[i], delta_z[i], rho=rho, mu=mu,
                roughness=roughness, Cf_constant=Cf_constant, C_lam=C_lam)
        dfdt[i] = 2.0 * Cf_i / delta_z[i] * v[i] * abs(v[i])
    return float(_trapz(dfdt, zs))


def derivative_tapered(
    v0: float,
    h: float,
    g: float,
    zs: np.ndarray,
    delta_z: np.ndarray,
    dddt_z: np.ndarray,
    ddddt_z: np.ndarray,
    fric_kw: dict,
) -> tuple[float, float]:
    """dv0/dt and dh/dt for a tapered crack (generalizes solver.py::_derivative).

    The inertia coefficient ``(h+L)`` is replaced by ``int delta(z_b)/delta(z) dz``
    (the narrow neck weights the inertia); friction and the unsteady term use the
    tapered profile. The crack floor ``z_b`` is ``zs[0]``. Reduces exactly to
    ``_derivative`` for a uniform gap.
    """
    coeff = float(_trapz(delta_z[0] / delta_z, zs))  # replaces (h + L)
    if coeff <= 0.0:
        return 0.0, 0.0
    v = vel_profile_tapered(v0, zs, delta_z, dddt_z)
    fric = friction_tapered(zs, v, delta_z, **fric_kw)
    # Advection from the momentum integral: -1/2 (v_h^2 - v0^2).
    rhs = -0.5 * (v[-1] ** 2 - v[0] ** 2) - g * h - fric
    rhs -= additional_term_tapered(zs, v0, v, delta_z, dddt_z, ddddt_z)
    return rhs / coeff, float(v[-1])


def liquid_dynamics_tapered(
    w_in: np.ndarray,
    t_in: np.ndarray,
    L: float,
    cfg=None,
    e_ice_zeta: np.ndarray | None = None,
    e_ice_values: np.ndarray | None = None,
    w_floor: float = 1e-4,
):
    """Liquid dynamics in a z-dependent (ice-lined) crack.

    Same structure as :func:`enceladus_plume.liquid_dynamics.solver.liquid_dynamics`
    but the effective gap is ``delta_eff(z) = delta_tidal(t) - 2 e_ice(zeta)`` with
    ``zeta = z + L`` the absolute height from the floor. The ice lining is
    quasi-static, so ``d delta/dt`` and its second derivative are the tidal ones.
    With ``e_ice = 0`` this reduces to the uniform-width solver (regression
    target).

    Returns ``(w_rec, h_rec, t_rec, v_rec)`` where ``w_rec`` is the tidal width.
    """
    from scipy.integrate import solve_ivp
    from .solver import _make_clamped_rhs

    if cfg is None:
        from ..config import load_config
        cfg = load_config()

    phys = cfg.physical
    lp = cfg.liquid_dynamics
    fp = cfg.friction
    P = phys.orbital_period
    g = phys.gravity

    fric_kw = dict(
        model=fp.liquid_model, Cf_constant=fp.liquid_Cf_constant,
        rho=phys.liquid_density, mu=phys.liquid_viscosity,
        roughness=fp.roughness, C_lam=fp.C_lam,
    )

    w_ext = np.concatenate([w_in, w_in, w_in])
    t_ext = np.concatenate([t_in - P, t_in, t_in + P])
    dwdt_ext = np.gradient(w_ext, t_ext)
    dwdt2_ext = np.gradient(dwdt_ext, t_ext)

    D = L / 10.0
    t_stop = P * lp.n_periods
    npts = lp.npts_velocity
    have_ice = e_ice_zeta is not None and e_ice_values is not None

    def _interp(arr, t_now):
        return float(np.interp(t_now % P, t_ext, arr))

    def raw_rhs(t, y):
        v, h = float(y[0]), float(y[1])
        w = _interp(w_ext, t)
        dwdt = _interp(dwdt_ext, t)
        dwdt2 = _interp(dwdt2_ext, t)
        zs = np.linspace(-L, h, npts)
        if have_ice:
            e = np.interp(zs + L, e_ice_zeta, e_ice_values)
            delta_z = np.clip(w - 2.0 * e, w_floor, None)
        else:
            delta_z = np.full(npts, w)
        dddt_z = np.full(npts, dwdt)
        ddddt_z = np.full(npts, dwdt2)
        dvdt, dhdt = derivative_tapered(v, h, g, zs, delta_z, dddt_z, ddddt_z, fric_kw)
        return [dvdt, dhdt]

    rhs = _make_clamped_rhs(raw_rhs, D, L)
    n_output = max(int(t_stop / 10.0), 2000)
    t_eval = np.linspace(0.0, t_stop, n_output)

    sol = solve_ivp(rhs, (0.0, t_stop), [0.0, 0.0], method=lp.ode_method,
                    rtol=lp.rtol, atol=lp.atol, max_step=lp.max_step, t_eval=t_eval)

    t_all = sol.t
    v_rec = sol.y[0]
    h_rec = sol.y[1]
    w_rec = np.array([_interp(w_ext, t) for t in t_all])
    return w_rec, h_rec, t_all, v_rec
