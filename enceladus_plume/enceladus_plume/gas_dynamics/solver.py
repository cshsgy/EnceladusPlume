"""Gas dynamics solver -- MikiModelFull and bisection wrapper.

Port of MikiModelFull.m and GasDynamicsMarchInTimeConstrained.m (2022),
with Darcy-Weisbach friction support.
"""

from __future__ import annotations

import numpy as np

from ..friction import fanning_friction_factor
from ..physics import (
    vapor_pressure,
    evaporation_rate,
    miki_find_evap,
)


def _gas_Cf(model: str, v: float, width: float, rho: float,
            mu: float, roughness: float, Cf_constant: float,
            C_lam: float) -> float:
    """Fanning coefficient for the gas column at a single point."""
    return fanning_friction_factor(
        model, v, width, rho=rho, mu=mu,
        roughness=roughness, Cf_constant=Cf_constant, C_lam=C_lam,
    )


def miki_model_full(
    r: float,
    width: float,
    zs: np.ndarray,
    T2: np.ndarray,
    K: float,
    Lv: float,
    G: float,
    Dx: float,
    hmax: float,
    rg: float = 8.314 / 0.018,
    gamma_eff: float = 1.33,
    friction_model: str = "constant",
    Cf_constant: float = 0.002,
    mu_vapor: float = 8.0e-6,
    roughness: float = 0.0,
    C_lam: float = 96.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Integrate the gas column from water surface to crack top.

    Uses RK4 for the density equation, with evaporation from the
    secant-method wall temperature solver at each level.

    Parameters
    ----------
    r              : condensation ratio (0 < r < 1)
    width          : crack width (m)
    zs             : vertical grid (m), 0 = water surface, increasing upward
    T2             : ice temperature at one grid spacing from the wall, per level
    K              : thermal conductivity (W/(m*K))
    Lv             : latent heat (J/kg)
    G              : gravity (m/s^2)
    Dx             : wall-normal grid spacing (m)
    hmax           : maximum water level (m) -- above this, evap is suppressed
    rg             : specific gas constant for vapor
    gamma_eff      : effective heat-capacity ratio
    friction_model : ``"constant"``, ``"laminar"``, or ``"churchill"``
    Cf_constant    : Fanning coefficient when friction_model="constant"
    mu_vapor       : dynamic viscosity of vapor (Pa*s)
    roughness      : absolute wall roughness (m)
    C_lam          : laminar-regime constant (96 for parallel plates)

    Returns
    -------
    Tw      : wall temperature profile (K)
    Ev      : evaporation rate profile (kg/(m^2*s))
    MTop    : Mach number at the top level (-1 if rho or phi went negative)
    PhiTop  : mass flux at the top level (kg/(m^2*s))
    """
    Tb = 273.15
    nz = len(zs)
    dz = zs[1] - zs[0] if nz > 1 else 1.0

    phi0 = (1.0 - r) * vapor_pressure(Tb) / np.sqrt(2.0 * np.pi * rg * Tb)
    rho0 = vapor_pressure(Tb) / (rg * Tb) * r

    rho = np.zeros(nz)
    M = np.zeros(nz)
    T = np.zeros(nz)
    Tw = np.zeros(nz)
    Ev = np.zeros(nz)

    rho[0] = rho0
    M[0] = phi0 / rho0 / np.sqrt(gamma_eff * rg * Tb)
    T[0] = Tb
    Tw[0] = Tb
    phi = phi0

    def _Cf(v_mag, dn):
        return _gas_Cf(friction_model, v_mag, width, dn, mu_vapor,
                       roughness, Cf_constant, C_lam)

    i_final = 0
    for i in range(1, nz):
        i_final = i
        dn = rho[i - 1]
        pb = dn * rg * T[i - 1]
        Tw[i], Ev[i] = miki_find_evap(T2[i], T[i - 1], pb, Dx, K, Lv, rg)

        if zs[i] > hmax and Ev[i] > 0:
            Ev[i] = 0.0

        if Ev[i] > 0:
            T[i] = (phi * T[i - 1] + 2.0 * dz * Ev[i] * Tw[i]) / (phi + 2.0 * Ev[i])
        else:
            T[i] = T[i - 1]

        pbt = dn * rg * T[i]
        if i < nz - 1:
            _, Evt = miki_find_evap(T2[i + 1], T[i], pbt, Dx, K, Lv, rg)
        else:
            _, Evt = miki_find_evap(T2[i], T[i], pbt, Dx, K, Lv, rg)
        if zs[i] > hmax and Evt > 0:
            Evt = 0.0

        v = (phi + 2.0 * dz * Ev[i]) / dn
        M[i] = v / np.sqrt(gamma_eff * rg * T[i - 1])
        Cd = _Cf(v, dn)

        k1 = (2.0 * Cd * dn * v**2 / width + dn * G) / (v**2 - rg * T[i - 1])

        dn2 = rho[i - 1] + dz * 0.5 * k1
        v2 = (phi + dz * (Ev[i] + Evt)) / dn2
        Cd2 = _Cf(v2, dn2)
        T_mid = 0.5 * (T[i - 1] + T[i])
        k2 = (2.0 * Cd2 * dn2 * v2**2 / width + dn2 * G) / (v2**2 - rg * T_mid)

        dn3 = rho[i - 1] + dz * 0.5 * k2
        v3 = (phi + dz * (Ev[i] + Evt)) / dn3
        Cd3 = _Cf(v3, dn3)
        k3 = (2.0 * Cd3 * dn3 * v3**2 / width + dn3 * G) / (v3**2 - rg * T_mid)

        dn4 = rho[i - 1] + dz * k3
        v4 = (phi + 2.0 * dz * Evt) / dn4
        Cd4 = _Cf(v4, dn4)
        k4 = (2.0 * Cd4 * dn4 * v4**2 / width + dn4 * G) / (v4**2 - rg * T[i])

        rho[i] = rho[i - 1] + dz / 6.0 * (k1 + k4 + 2.0 * k2 + 2.0 * k3)

        if rho[i] - rho[i - 1] > 0:
            rho[i] = -1.0
        if rho[i] < 0:
            break
        if M[i] > 1.6:
            break

        phi += dz * Ev[i]
        if phi < 0:
            break

    Ev[0] = Ev[1] if nz > 1 else 0.0

    if rho[i_final] > 0 and phi > 0:
        MTop = M[i_final]
        PhiTop = phi
    else:
        MTop = -1.0
        PhiTop = phi

    return Tw, Ev, MTop, PhiTop


def gas_dynamics_march_constrained(
    width: float,
    zs: np.ndarray,
    T2: np.ndarray,
    K: float,
    Lv: float,
    G: float,
    Dx: float,
    r_last: float,
    hmax: float,
    max_change: float = 0.005,
    tol: float = 1e-4,
    rg: float = 8.314 / 0.018,
    gamma_eff: float = 1.33,
    friction_model: str = "constant",
    Cf_constant: float = 0.002,
    mu_vapor: float = 8.0e-6,
    roughness: float = 0.0,
    C_lam: float = 96.0,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Bisection search for *r* near the previous value.

    Returns (PhiTop, Tw, Ev, r).
    """
    r_l = r_last - max_change
    r_r = r_last + max_change

    Tw = np.zeros_like(zs)
    Ev = np.zeros_like(zs)
    rs = 0.5 * (r_l + r_r)

    while r_r - r_l > tol:
        rs = 0.5 * (r_r + r_l)
        Tw, Ev, M, PhiTop = miki_model_full(
            rs, width, zs, T2, K, Lv, G, Dx, hmax, rg, gamma_eff,
            friction_model, Cf_constant, mu_vapor, roughness, C_lam)

        if PhiTop < 0:
            r_r = rs
        elif M < 0:
            r_l = rs
        elif M > 1.6:
            r_l = rs
        else:
            r_r = rs

    mask = Tw == 0
    Tw[mask] = T2[mask]

    return PhiTop, Tw, Ev, rs
