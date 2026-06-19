"""Diagnostic A: wall ice mass budget over a diurnal cycle.

Tracks how much ice is *added* to (condensation/healing) or *removed* from
(frictional melting) each point of the crack wall over one diurnal cycle. The
underlying fluxes are already computed by the model; this module integrates
them over the cycle and accumulates them along the wall, which the rest of the
pipeline currently discards.

Geometry (absolute height ``zeta`` measured from the crack floor):

    zeta = 0          crack floor
    zeta = L          equilibrium water level
    zeta = L + h(t)   instantaneous water level   (h is displacement, in [-L, D])
    zeta = L + D      surface (crack top),  D = L / 10

Two regimes, gated by the moving water level:

* **Exposed wall** (``zeta`` above the water level, below the plume choke
  height): net condensation. The per-wall-area flux is

      E(zeta) = find_evap_surface(Tb, Te, sigma, surface - zeta, kt, Lv)

  which depends only on the (fixed) distance below the surface, so it is
  *time-independent per wall point*; only the exposure window and the choke
  height vary in time. ``E < 0`` means vapour deposits onto the wall.
  Deposition is vapour-limited: above the height where the rising column flux
  ``phi(t) + f`` reaches zero (the choke), there is no vapour left to deposit,
  exactly as in the gas-column solver.

* **Covered wall** (below the water level): melting only, no refreezing
  (per the modelling choice). Turbulent wall dissipation supplies heat at

      q_fric(zeta, t) = Cf * rho_w * |v(zeta, t)|^3      [W/m^2]

  removing ice at ``q_fric / Lf``.  (The O(1) prefactor follows the model's
  Fanning ``Cf`` convention; it can be tied exactly to the resolved
  dissipation profile later. The healing argument rests on the profile shape
  and order of magnitude.)

The net secular tendency is ``Delta_sigma(zeta) = deposition - melt`` per
cycle (kg/m^2), or a thickness change ``Delta_e = Delta_sigma / rho_ice``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .friction import fanning_friction_factor
from .physics import (
    vapor_pressure,
    evaporation_rate_simple,
    find_evap_surface,
)

# Gas-column constants, matching gas_dynamics/interpolator.py.
_RG = 8.341 / 0.018
_SIGMA = 5.67e-8
_KT = 2.4
_TE = 68.0

# Pure water-ice density (kg/m^3).
RHO_ICE = 917.0

_SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass
class WallBudgetResult:
    """Per-cycle wall ice budget along the crack (height ``zeta`` from floor)."""

    zeta: np.ndarray          # m, absolute height from crack floor
    deposition: np.ndarray    # kg/m^2 per cycle, condensation gain (>= 0)
    melt: np.ndarray          # kg/m^2 per cycle, frictional removal (>= 0)
    net_sigma: np.ndarray     # kg/m^2 per cycle, deposition - melt
    net_thickness: np.ndarray  # m per cycle, net_sigma / rho_ice
    L: float                  # equilibrium water depth (m)
    D: float                  # surface barrier above equilibrium (m)
    rho_ice: float
    healing_cycles: float     # cycles to close the opening (NaN if none)
    healing_years: float

    @property
    def surface(self) -> float:
        return self.L + self.D

    def summary(self) -> str:
        zc = self.zeta
        top = self.net_thickness[zc >= self.L]  # above equilibrium water level
        bot = self.net_thickness[zc < self.L]
        return (
            f"wall mass budget over one cycle:\n"
            f"  net deposition (top, zeta>=L):  peak {np.nanmax(self.net_thickness)*1e3:+.3e} mm/cycle\n"
            f"  net change above water level:   mean {np.nanmean(top)*1e3:+.3e} mm/cycle\n"
            f"  net change below water level:   mean {np.nanmean(bot)*1e3:+.3e} mm/cycle\n"
            f"  total deposited:  {np.trapz(self.deposition, zc):.3e} kg/m per cycle\n"
            f"  total melted:     {np.trapz(self.melt, zc):.3e} kg/m per cycle\n"
            f"  healing time to close opening: {self.healing_cycles:.3e} cycles "
            f"({self.healing_years:.3e} yr)"
        )


def _last_period_mask(t: np.ndarray, P: float) -> np.ndarray:
    """Boolean mask selecting the last full orbital period of the solution."""
    t_end = float(t[-1])
    return t >= (t_end - P)


def wall_mass_budget(
    t: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    cfg: Config,
    Tb: float,
    n_z: int = 400,
    rho_ice: float = RHO_ICE,
) -> WallBudgetResult:
    """Compute the per-cycle wall ice budget for a single solved case.

    Parameters
    ----------
    t, h, w, v : recorded time, water-level displacement, crack width, and
        entrance velocity time series from :func:`liquid_dynamics`.
    r : inlet ratio time series (from the gas lookup table) used only to gate
        deposition by the plume choke height. May be all-NaN to disable the
        choke cap (deposition then fills the whole exposed wall, an upper
        bound).
    cfg : configuration (physical + friction parameters).
    Tb : base temperature used for the gas/evaporation calculation.
    """
    t = np.asarray(t, float)
    h = np.asarray(h, float)
    w = np.asarray(w, float)
    v = np.asarray(v, float)
    r = np.asarray(r, float)

    phys = cfg.physical
    fp = cfg.friction
    P = float(phys.orbital_period)
    L = float(phys.equilibrium_depth)
    D = L / 10.0
    surface = L + D
    Lv = float(phys.latent_heat)
    Lf = float(phys.latent_heat_fusion)
    rho_w = float(phys.liquid_density)

    # dw/dt for the in-crack velocity profile (full series, then restrict).
    dwdt = np.gradient(w, t)

    # Restrict to the last full period (periodic steady state).
    m = _last_period_mask(t, P)
    t_c, h_c, w_c, v_c, r_c, dwdt_c = t[m], h[m], w[m], v[m], r[m], dwdt[m]

    # Wall grid: condensation is concentrated within the thermal skin depth
    # (~0.3 m) of the surface, so the grid is refined geometrically near the
    # top (distance-below-surface d_top -> 0) while a uniform part resolves the
    # smooth frictional-melt region over the full water column.
    n_lin = max(n_z // 2, 50)
    n_geo = max(n_z - n_lin, 50)
    zeta_lin = np.linspace(0.0, surface, n_lin)
    zeta_geo = surface - np.geomspace(1e-3, surface, n_geo)
    zeta = np.unique(np.concatenate([zeta_lin, zeta_geo, [0.0, surface]]))
    zeta = np.clip(zeta, 0.0, surface)
    d_top = np.clip(surface - zeta, 1e-10, None)
    E = np.array([find_evap_surface(Tb, _TE, _SIGMA, d, _KT, Lv) for d in d_top])
    dep_flux = np.clip(-E, 0.0, None)  # condensation gain rate, kg/m^2/s

    # Cumulative deposition demand from the floor, for choke-height inversion.
    # Phi(zeta) = integral_0^zeta dep_flux dz'  (per unit width factor folded
    # out below).  Vapour removed from the column between the water surface and
    # zeta is (Phi(zeta) - Phi(zeta_w)) / width; the column chokes when that
    # equals phi(t).
    Phi = np.concatenate([[0.0], np.cumsum(0.5 * (dep_flux[1:] + dep_flux[:-1]) * np.diff(zeta))])

    # Inlet flux phi(t) = (1 - r) * evap_simple(Tb).
    ec = vapor_pressure(273.15) / np.sqrt(2.0 * np.pi * _RG)
    bv = Lv / _RG
    evap0 = evaporation_rate_simple(ec, bv, Tb)
    phi_t = (1.0 - r_c) * evap0

    deposition = np.zeros(len(zeta))
    melt = np.zeros(len(zeta))

    # Time weights (trapezoidal) over the last period.
    dt = np.gradient(t_c)

    # Constant-model friction coefficient (per-point model handled in loop).
    constant_model = (fp.liquid_model == "constant")
    Cf_const = float(fp.liquid_Cf_constant)

    for k in range(len(t_c)):
        zeta_w = L + h_c[k]          # water level
        wk = w_c[k]
        dtk = dt[k]

        # --- exposed wall: condensation, gated by choke height ---
        if np.isfinite(r_c[k]) and wk > 0:
            demand = phi_t[k] * wk    # column flux available to deposit
            Phi_w = np.interp(zeta_w, zeta, Phi)
            target = Phi_w + demand
            if Phi[-1] <= target:
                zeta_c = surface       # never chokes within the column
            else:
                zeta_c = float(np.interp(target, Phi, zeta))
        else:
            zeta_c = surface           # no choke info -> fill whole exposed wall

        exposed = (zeta > zeta_w) & (zeta <= zeta_c)
        deposition[exposed] += dep_flux[exposed] * dtk

        # --- covered wall: frictional melt only ---
        covered = zeta <= zeta_w
        if np.any(covered):
            zc = zeta[covered]
            v_prof = v_c[k] - (dwdt_c[k] / wk) * zc
            if constant_model:
                Cf = Cf_const
            else:
                Cf = np.array([
                    fanning_friction_factor(
                        fp.liquid_model, vv, wk, rho=rho_w,
                        mu=float(phys.liquid_viscosity),
                        roughness=float(fp.roughness), C_lam=float(fp.C_lam))
                    for vv in v_prof
                ])
            q_fric = Cf * rho_w * np.abs(v_prof) ** 3
            melt[covered] += q_fric / Lf * dtk

    net_sigma = deposition - melt
    net_thickness = net_sigma / rho_ice

    healing_cycles, healing_years = _healing_time(
        zeta, net_thickness, L, w_c, P)

    return WallBudgetResult(
        zeta=zeta, deposition=deposition, melt=melt,
        net_sigma=net_sigma, net_thickness=net_thickness,
        L=L, D=D, rho_ice=rho_ice,
        healing_cycles=healing_cycles, healing_years=healing_years,
    )


def _healing_time(zeta, net_thickness, L, w_cycle, P):
    """Estimate cycles/years to close the crack opening near the top.

    Each wall thickens by ``net_thickness`` per cycle where deposition wins, so
    the gap (two walls) closes at ``2 * net_thickness``. We compare against the
    minimum width reached during the cycle (the effective opening to seal).
    """
    upper = zeta >= L  # above equilibrium water level
    rate = np.nanmax(net_thickness[upper]) if np.any(upper) else np.nan
    if not np.isfinite(rate) or rate <= 0:
        return float("nan"), float("nan")
    opening = float(np.nanmin(w_cycle))
    cycles = opening / (2.0 * rate)
    years = cycles * P / _SECONDS_PER_YEAR
    return cycles, years
