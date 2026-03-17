"""1-D heat diffusion solver for the ice wall.

Port of HeatDiffusionMarchInTime.m (2022).
"""

from __future__ import annotations

import numpy as np

from ..physics import surface_temperature


def heat_diffusion_step(
    z: float,
    x: np.ndarray,
    h: float,
    depth: float,
    T: np.ndarray,
    Tw: float,
    Kappa: float,
    K: float,
    Te: float,
    Dx: float,
    Dt: float,
) -> np.ndarray:
    """Advance the wall temperature profile by one explicit Euler step.

    Parameters
    ----------
    z     : vertical position of this level (m)
    x     : wall-normal coordinates (m), shape (nx,)
    h     : current water level displacement (m)
    depth : current crack depth above water (m)
    T     : temperature profile at this level, shape (nx,); modified in-place
    Tw    : wall temperature from the gas solver (K)
    Kappa : thermal diffusivity (m^2/s)
    K     : thermal conductivity (W/(m*K))
    Te    : effective surface temperature (K)
    Dx    : wall-normal grid spacing (m)
    Dt    : time step (s)

    Returns
    -------
    T : updated temperature profile, shape (nx,)
    """
    T = T.copy()
    nx = len(x)
    expose = z > h

    # Inner-wall boundary condition
    if not expose:
        T[0] = 273.15  # submerged -> ocean temperature
    else:
        T[0] = Tw

    # Interior diffusion (explicit Euler)
    dTdt = np.zeros(nx)
    for j in range(1, nx - 1):
        dTdt[j] = Kappa * (T[j - 1] - 2.0 * T[j] + T[j + 1]) / (Dx ** 2)
    T += dTdt * Dt

    # Far-wall boundary: radiative equilibrium
    Ts = surface_temperature(Te, T[-1], depth, K)
    if Ts > 0:
        T[-1] = T[-2] - (T[-2] - Ts) / depth * Dx

    return T
