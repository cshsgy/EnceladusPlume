"""Shared thermodynamic / physics functions used across all solvers."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Vapor pressure
# ---------------------------------------------------------------------------

def vapor_pressure(T: float | np.ndarray) -> float | np.ndarray:
    """Clausius-Clapeyron vapor pressure of water ice (Pa).

    Pv = 3.63e12 * exp(-6147 / T)
    """
    return 3.63e12 * np.exp(-6147.0 / T)


# ---------------------------------------------------------------------------
# Evaporation rate (2022 full-solver style, one-side)
# ---------------------------------------------------------------------------

def evaporation_rate(Pg: float, Pw: float, Tg: float, Tw: float,
                     rg: float) -> float:
    """Net evaporation rate from Hertz-Knudsen relation (one side).

    E = -Pg / sqrt(2*pi*rg*Tg) + Pw / sqrt(2*pi*rg*Tw)

    Parameters
    ----------
    Pg : gas-phase pressure (Pa)
    Pw : vapor pressure at wall temperature (Pa)
    Tg : gas temperature (K)
    Tw : wall temperature (K)
    rg : specific gas constant for water vapor (J/(kg*K))
    """
    return -Pg / np.sqrt(2.0 * np.pi * rg * Tg) + Pw / np.sqrt(2.0 * np.pi * rg * Tw)


# ---------------------------------------------------------------------------
# Evaporation rate (2023 interpolator style)
# ---------------------------------------------------------------------------

def evaporation_rate_simple(ec: float, bv: float, Tm: float) -> float:
    """Simplified evaporation rate used in the 2023 gas interpolator.

    evap = ec / sqrt(Tm) * exp(bv * (1/T0 - 1/Tm))
    """
    T0 = 273.15
    return ec / np.sqrt(Tm) * np.exp(bv * (1.0 / T0 - 1.0 / Tm))


# ---------------------------------------------------------------------------
# Heat-balance residual for wall temperature (2022 full solver)
# ---------------------------------------------------------------------------

def function_to_solve(Ev: float, T2: float, Tw: float,
                      Dx: float, K: float, Lv: float) -> float:
    """Residual: conduction flux minus evaporative latent heat flux.

    v = K * (T2 - Tw) / Dx  -  Lv * Ev
    """
    return K * (T2 - Tw) / Dx - Lv * Ev


def miki_find_evap(T2: float, Tg: float, Pb: float,
                   Dx: float, K: float, Lv: float,
                   rg: float) -> tuple[float, float]:
    """Secant method for wall temperature Tw that balances heat and evap.

    Returns (Tw, Ev).
    """
    Tw1 = 273.15
    Tw2 = T2

    Ev1 = evaporation_rate(Pb, vapor_pressure(Tw1), Tg, Tw1, rg)
    v1 = function_to_solve(Ev1, T2, Tw1, Dx, K, Lv)
    Ev2 = evaporation_rate(Pb, vapor_pressure(Tw2), Tg, Tw2, rg)
    v2 = function_to_solve(Ev2, T2, Tw2, Dx, K, Lv)

    for _ in range(200):
        if abs(Tw1 - Tw2) <= 1e-3:
            break
        if abs(v1 - v2) < 1e-30:
            break
        Tw3 = Tw1 - v1 * (Tw1 - Tw2) / (v1 - v2)
        Tw1 = Tw2
        v1 = v2
        Tw2 = Tw3
        Ev2 = evaporation_rate(Pb, vapor_pressure(Tw2), Tg, Tw2, rg)
        v2 = function_to_solve(Ev2, T2, Tw2, Dx, K, Lv)

    Tw = Tw2
    Ev = evaporation_rate(Pb, vapor_pressure(Tw), Tg, Tw, rg)
    return Tw, Ev


# ---------------------------------------------------------------------------
# Surface temperature (radiative equilibrium at far wall)
# ---------------------------------------------------------------------------

def surface_temperature(Te: float, Tw: float, depth: float,
                        K: float, sigma: float = 5.67e-8) -> float:
    """Solve  Ts^4 + c*Ts = Te^4 + c*Tw  for the positive real root.

    c = 2*K / (sigma * pi * depth)

    Uses numpy polynomial root-finding on the quartic.
    Returns -1 if no valid root is found.
    """
    c = 2.0 * K / (sigma * np.pi * depth)
    rhs = Te**4 + c * Tw
    # Quartic: Ts^4 + c*Ts - rhs = 0  =>  coefficients [1, 0, 0, c, -rhs]
    roots = np.roots([1.0, 0.0, 0.0, c, -rhs])
    Ts = -1.0
    for r in roots:
        if np.isreal(r) and float(r.real) > 0:
            Ts = float(r.real)
    return Ts


# ---------------------------------------------------------------------------
# Surface evaporation helper used in 2023 gas interpolator (find_evap)
# ---------------------------------------------------------------------------

def find_evap_surface(Tw: float, Te: float, sigma: float,
                      d: float, k: float, L: float) -> float:
    """Bisection for surface temperature then return evap rate.

    Solves  2*sigma*(Ts^4 - Te^4) + (4*k/(pi*d))*(Ts - Tw) = 0
    for Ts, then  E = -sigma/L * (Ts^4 - Te^4).
    """
    c1 = 2.0 * sigma
    c2 = 4.0 * k / (np.pi * d)

    T_l = 1.0
    T_r = Tw
    for _ in range(200):
        if T_r - T_l <= 1e-8:
            break
        T_m = 0.5 * (T_l + T_r)
        v_l = c1 * (T_l**4 - Te**4) + c2 * (T_l - Tw)
        v_m = c1 * (T_m**4 - Te**4) + c2 * (T_m - Tw)
        if v_l * v_m < 0:
            T_r = T_m
        else:
            T_l = T_m

    Ts = 0.5 * (T_l + T_r)
    return -sigma / L * (Ts**4 - Te**4)
