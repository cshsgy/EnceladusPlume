"""Helper functions for the liquid dynamics solver.

Ports of vel_now.m, friction.m, additional_term.m, pvpt.m from the
MATLAB codebase, with Darcy-Weisbach friction support.
"""

from __future__ import annotations

import numpy as np

from ..friction import fanning_friction_factor

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def vel_now(v0: float, h: float, L: float, w: float, dwdt: float,
            npts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Compute vertical velocity profile along the crack.

    Parameters
    ----------
    v0   : velocity at the bottom of the water column
    h    : water level displacement from equilibrium
    L    : equilibrium water depth
    w    : current crack width
    dwdt : time derivative of crack width
    npts : number of grid points

    Returns
    -------
    zs          : 1-D array of z positions from -L to h
    vel_profile : 1-D array of velocity at each z
    """
    zs = np.linspace(-L, h, npts)
    dvdz = -dwdt / w
    vel_profile = v0 + dvdz * (zs - zs[0])
    return zs, vel_profile


def pvpt(zs: np.ndarray, w: float, dwdt: float, dwdt2: float) -> float:
    """Integrate (dwdt^2/w^2 - dwdt2/w) over the sub-array *zs*.

    This is dv/dt with the dv0/dt contribution peeled off.
    The integrand is spatially constant, so the result is simply
    (dwdt^2/w^2 - dwdt2/w) * (zs[-1] - zs[0]).
    """
    integrand_val = dwdt**2 / w**2 - dwdt2 / w
    if len(zs) < 2:
        return 0.0
    return integrand_val * (zs[-1] - zs[0])


def friction(
    zs: np.ndarray,
    v: np.ndarray,
    w: float,
    *,
    model: str = "constant",
    Cf_constant: float = 0.004,
    rho: float = 1000.0,
    mu: float = 1.8e-3,
    roughness: float = 0.0,
    C_lam: float = 96.0,
) -> float:
    """Integrate wall friction  2*Cf(z)/w * v|v|  along the crack.

    When *model* is ``"constant"`` the Fanning coefficient ``Cf_constant``
    is used at every point (legacy behaviour).  Otherwise the coefficient
    is evaluated at each grid point via :func:`fanning_friction_factor`
    using the local velocity to compute the Reynolds number.

    Returns the scalar friction integral.
    """
    npts = len(zs)
    dfdt = np.empty(npts)
    for i in range(npts):
        Cf_i = fanning_friction_factor(
            model, v[i], w, rho=rho, mu=mu,
            roughness=roughness, Cf_constant=Cf_constant, C_lam=C_lam,
        )
        dfdt[i] = 2.0 * Cf_i / w * v[i] * abs(v[i])

    return float(_trapz(dfdt, zs))


def additional_term(zs: np.ndarray, w: float, dwdt: float,
                    dwdt2: float) -> float:
    """Double integral of the pvpt term along the crack.

    For each z_i, compute  P(z_i) = pvpt(zs[0:i+1], w, dwdt, dwdt2),
    then integrate P over z using the trapezoidal rule.

    Because pvpt's integrand is constant in space, P(z) is linear in z,
    and the double integral simplifies to an analytic expression.
    """
    npts = len(zs)
    if npts < 3:
        return 0.0

    integrand_val = dwdt**2 / w**2 - dwdt2 / w

    P = integrand_val * (zs - zs[0])

    result = float(_trapz(P[1:], zs[1:]))
    return result
