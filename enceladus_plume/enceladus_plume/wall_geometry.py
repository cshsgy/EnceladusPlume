"""Stage B: self-consistent crack geometry from the wall ice budget.

Evolves a height-dependent ice lining ``e_ice(zeta)`` on each crack wall under
the per-cycle condensation/melt budget until it reaches a quasi-steady taper.

Model (see also :mod:`enceladus_plume.wall_budget`):

* The tidal (rock) opening ``delta(t)`` stays uniform along height; the taper is
  carried by an ice lining ``e_ice(zeta)``. The effective gas/flow gap is

      w_eff(zeta, t) = delta(t) - 2 * e_ice(zeta)        (>= w_floor)

* **Condensation** adds ice on exposed wall (above the water level, below the
  plume choke). The per-wall-area flux ``-E(zeta)`` is fixed in time
  (``E = find_evap_surface(...)`` depends only on distance below the surface);
  what changes as the taper grows is the **choke height**, computed from the
  z-dependent ``w_eff`` (the column chokes where ``phi(t) + f`` reaches zero,
  ``f = integral E / w_eff dz``). This is feedback #1 (choke self-limit).

* **Frictional melt** removes ice on the covered wall, ``q_fric/Lf`` with
  ``q_fric = Cf rho_w |v|^3``.

* **Overflow reopening** (feedback #2): wherever liquid water periodically
  reaches (``zeta <= L + max h(t)``), the ~273 K water keeps the wall ice-free,
  so the ice lining is held at zero there. Sealing can therefore only advance
  down to the water-reach height and stalls there.

Where the ice goes -- the modelling decision
---------------------------------------------
``find_evap_surface`` (intended) concentrates condensation within the thermal
skin depth (~0.3 m) of the crack *top*. The fully z-dependent solver
(:func:`evolve_geometry_full`) shows that, taken literally, ice therefore piles
up only at the lip and -- for deep water -- never narrows the channel the water
flows through, so the water does not climb. Physically a thin sealed lip plug
cannot persist under the tidal flexing: it fractures and/or is redistributed by
**ice convection (viscoelastic creep)**. We therefore represent the *net*
geometric effect of that redistribution as an **effective channel narrowing**
(the reduced-order effective-delta feedback below). The detailed ice rheology
(convection / creep) is left as the OPEN QUESTION.

The mechanism then closes: condensation at the lip -> ice convection
redistributes it (open question) -> effective channel narrowing -> water
squeezed faster -> climbs to overflow -> secondary peak.

Routines (most to least physical for this problem):

* :func:`evolve_geometry_coupled` -- ADOPTED reduced-order model, parametrized by
  the absolute tidal swing ``delta_w = w_max - w_min`` (which the lining leaves
  unchanged). As the crack seals, its effective minimum width ``w_eff`` shrinks
  while ``R_eff = 1 + delta_w / w_eff`` grows, so the water rise
  (``h_max ~ delta_w^2 w_eff^{-3}``) diverges and *every* flexing crack overflows
  once sealed far enough -- there is no forcing threshold; ``delta_w`` sets only
  the seal depth ``w_eff*`` at overflow.

* :func:`evolve_geometry_full` -- REFERENCE z-dependent solver (continuity
  ``d delta/dt + d(delta v)/dz = 0``). Faithful, but with lip-only condensation
  it shows no climb for deep water -- this is *why* the convection-redistribution
  step (and hence the effective-delta model) is needed.

* :func:`evolve_geometry` -- OBSOLETE frozen-liquid first attempt (``h(t)``
  computed once); kept only for regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .friction import fanning_friction_factor
from .physics import vapor_pressure, evaporation_rate_simple, find_evap_surface
from .liquid_dynamics.solver import liquid_dynamics
from .liquid_dynamics.tapered import liquid_dynamics_tapered
from .utils import build_width_series
from .wall_budget import _RG, _SIGMA, _KT, _TE, RHO_ICE, wall_mass_budget

_W_FLOOR = 1e-4  # m, minimum effective gap before a height is treated as sealed


@dataclass
class GeometryEvolutionResult:
    """Equilibrium taper from the stage-B geometry evolution."""

    zeta: np.ndarray            # m, absolute height from floor
    e_ice: np.ndarray           # m, equilibrium ice lining per wall
    w_eff_min: np.ndarray       # m, minimum effective gap over the cycle vs height
    L: float
    D: float
    water_max_height: float     # m, highest point liquid water reaches (L + max h)
    seal_height: float          # m, lowest fully sealed height (NaN if none)
    open_top_height: float      # m, top of the connected open channel from the floor
    iterations: int
    converged: bool
    history: list = field(default_factory=list)  # max e_ice per iteration

    @property
    def surface(self) -> float:
        return self.L + self.D

    def summary(self) -> str:
        return (
            f"stage-B equilibrium geometry:\n"
            f"  iterations: {self.iterations}  converged: {self.converged}\n"
            f"  water reaches up to:   {self.water_max_height:.2f} m "
            f"({self.surface - self.water_max_height:.2f} m below surface)\n"
            f"  open channel up to:    {self.open_top_height:.2f} m "
            f"({self.surface - self.open_top_height:.2f} m below surface)\n"
            f"  seal height:           {self.seal_height:.2f} m\n"
            f"  peak ice lining:       {np.nanmax(self.e_ice)*1e3:.3e} mm per wall\n"
            f"  min effective opening: {np.nanmin(self.w_eff_min)*1e3:.3e} mm"
        )


def _condensation_flux(zeta: np.ndarray, surface: float, Tb: float,
                       Lv: float) -> np.ndarray:
    """Time-independent per-wall-area condensation gain rate -E(zeta) >= 0."""
    d_top = np.clip(surface - zeta, 1e-10, None)
    E = np.array([find_evap_surface(Tb, _TE, _SIGMA, d, _KT, Lv) for d in d_top])
    return np.clip(-E, 0.0, None)


def _make_grid(surface: float, n_z: int) -> np.ndarray:
    """Wall grid refined near the surface (where condensation concentrates)."""
    n_lin = max(n_z // 2, 50)
    n_geo = max(n_z - n_lin, 50)
    zeta = np.unique(np.concatenate([
        np.linspace(0.0, surface, n_lin),
        surface - np.geomspace(1e-3, surface, n_geo),
        [0.0, surface],
    ]))
    return np.clip(zeta, 0.0, surface)


def _deposition_per_cycle(
    zeta, dep_flux, t_c, h_c, w_c, phi_t, r_c, e_ice, L, dt,
) -> np.ndarray:
    """Condensation deposited on each wall point over one cycle (kg/m^2).

    Deposition is gated by exposure (above the moving water level) and by the
    plume choke height, which depends on the current z-dependent effective gap
    ``w_eff = w(t) - 2 e_ice``.
    """
    deposition = np.zeros(len(zeta))
    for k in range(len(t_c)):
        if not np.isfinite(r_c[k]):
            continue
        zeta_w = L + h_c[k]
        w_eff = np.clip(w_c[k] - 2.0 * e_ice, _W_FLOOR, None)
        # column flux f(zeta) = cumulative integral of E / w_eff from the water
        # surface upward (E = -dep_flux). Chokes where phi + f < 0.
        exposed = zeta > zeta_w
        if not np.any(exposed):
            continue
        ze = zeta[exposed]
        integrand = -dep_flux[exposed] / w_eff[exposed]  # E / w_eff <= 0
        f = np.concatenate([[0.0], np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ze))])
        column = phi_t[k] + f
        below_choke = column > 0.0
        idx = np.where(exposed)[0][below_choke]
        deposition[idx] += dep_flux[idx] * dt[k]
    return deposition


def evolve_geometry(
    t: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    cfg: Config,
    Tb: float,
    n_z: int = 600,
    n_t: int = 240,
    cycles_per_iter: float = 50.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    rho_ice: float = RHO_ICE,
) -> GeometryEvolutionResult:
    """Iterate the ice lining to a quasi-steady taper.

    Parameters mirror :func:`enceladus_plume.wall_budget.wall_mass_budget`.
    ``n_t`` subsamples the diurnal cycle for the (repeated) deposition integral;
    ``cycles_per_iter`` is the number of cycles advanced per update.
    """
    t = np.asarray(t, float); h = np.asarray(h, float); w = np.asarray(w, float)
    v = np.asarray(v, float); r = np.asarray(r, float)

    phys = cfg.physical
    fp = cfg.friction
    P = float(phys.orbital_period)
    L = float(phys.equilibrium_depth)
    D = L / 10.0
    surface = L + D
    Lv = float(phys.latent_heat)
    Lf = float(phys.latent_heat_fusion)
    rho_w = float(phys.liquid_density)

    dwdt_full = np.gradient(w, t)

    # last full period, subsampled to n_t points
    m = t >= (t[-1] - P)
    tc, hc, wc, vc, rc, dwc = t[m], h[m], w[m], v[m], r[m], dwdt_full[m]
    if len(tc) > n_t:
        sel = np.linspace(0, len(tc) - 1, n_t).round().astype(int)
        tc, hc, wc, vc, rc, dwc = tc[sel], hc[sel], wc[sel], vc[sel], rc[sel], dwc[sel]
    dt = np.gradient(tc)

    zeta = _make_grid(surface, n_z)
    dep_flux = _condensation_flux(zeta, surface, Tb, Lv)

    # inlet flux phi(t)
    ec = vapor_pressure(273.15) / np.sqrt(2.0 * np.pi * _RG)
    bv = Lv / _RG
    phi_t = (1.0 - rc) * evaporation_rate_simple(ec, bv, Tb)

    # frictional melt per cycle (independent of e_ice in this approximation)
    melt = _friction_melt_per_cycle(zeta, tc, hc, wc, vc, dwc, dt, fp, phys, Lf, L)

    # overflow reopening: water keeps the wall ice-free up to its max reach
    water_max = L + float(np.nanmax(hc))
    open_mask = zeta <= water_max

    e_ice = np.zeros(len(zeta))
    history = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        dep = _deposition_per_cycle(zeta, dep_flux, tc, hc, wc, phi_t, rc, e_ice, L, dt)
        de = (dep - melt) / rho_ice * cycles_per_iter
        e_new = e_ice + de
        e_new[open_mask] = 0.0                       # overflow keeps it open
        e_new = np.clip(e_new, 0.0, wc.min() / 2.0)  # gap cannot go negative
        step = float(np.nanmax(np.abs(e_new - e_ice)))
        history.append(float(np.nanmax(e_new)))
        e_ice = e_new
        if step < tol:
            converged = True
            break

    w_eff_min = np.clip(wc.min() - 2.0 * e_ice, 0.0, None)
    sealed = np.where(w_eff_min <= _W_FLOOR)[0]
    seal_height = float(zeta[sealed[0]]) if len(sealed) else float("nan")
    # top of the connected open channel from the floor
    open_top = surface
    if len(sealed):
        open_top = float(zeta[max(sealed[0] - 1, 0)])

    return GeometryEvolutionResult(
        zeta=zeta, e_ice=e_ice, w_eff_min=w_eff_min, L=L, D=D,
        water_max_height=water_max, seal_height=seal_height,
        open_top_height=open_top, iterations=it, converged=converged,
        history=history,
    )


def _friction_melt_per_cycle(zeta, tc, hc, wc, vc, dwc, dt, fp, phys, Lf, L):
    """Ice removed by turbulent wall dissipation on the covered wall (kg/m^2)."""
    melt = np.zeros(len(zeta))
    rho_w = float(phys.liquid_density)
    constant = (fp.liquid_model == "constant")
    Cf_const = float(fp.liquid_Cf_constant)
    for k in range(len(tc)):
        zeta_w = L + hc[k]
        covered = zeta <= zeta_w
        if not np.any(covered):
            continue
        zc = zeta[covered]
        v_prof = vc[k] - (dwc[k] / wc[k]) * zc
        if constant:
            Cf = Cf_const
        else:
            Cf = np.array([
                fanning_friction_factor(
                    fp.liquid_model, vv, wc[k], rho=rho_w,
                    mu=float(phys.liquid_viscosity),
                    roughness=float(fp.roughness), C_lam=float(fp.C_lam))
                for vv in v_prof])
        melt[covered] += Cf * rho_w * np.abs(v_prof) ** 3 / Lf * dt[k]
    return melt


# ---------------------------------------------------------------------------
# Reduced-order coupled stage B
# ---------------------------------------------------------------------------

@dataclass
class CoupledEquilibriumResult:
    """Reduced-order coupled equilibrium under a fixed absolute tidal swing.

    The controlling forcing parameter is the absolute wall swing
    ``delta_w = w_max - w_min`` (set by the tides and rock mechanics), which the
    ice lining leaves unchanged. As the crack seals its mean width drops --- i.e.
    the effective minimum width ``w_eff`` shrinks --- while the effective ratio
    ``R_eff = 1 + delta_w / w_eff`` grows without bound.
    """

    w_eff: np.ndarray           # m, effective minimum width swept (mean seals down)
    R_eff: np.ndarray           # implied width ratio = 1 + delta_w / w_eff
    water_max: np.ndarray       # m, max water height (L + max h) for each w_eff
    L: float
    D: float
    delta_w: float              # m, absolute tidal swing (held fixed)
    w_eff_overflow: float       # m, effective width at which water overflows (NaN if none)
    overflow: bool              # whether the surface is reached within the sweep
    overflow_frac: float        # rise/D threshold used to define overflow
    deposition_rate: float      # mm/cycle, condensation supply (sanity)

    @property
    def surface_margin(self) -> np.ndarray:
        """Distance of the water from the surface (D - max h); 0 = overflow."""
        return self.D - (self.water_max - self.L)

    def summary(self) -> str:
        if self.overflow:
            line = (f"  OVERFLOW once sealed to w_eff* = {self.w_eff_overflow*1e3:.3f} mm "
                    f"(R_eff = {1.0 + self.delta_w/self.w_eff_overflow:.2f})")
        else:
            closest = self.D - (np.nanmax(self.water_max) - self.L)
            line = (f"  no overflow within sweep; closest approach "
                    f"{closest:.2f} m below surface")
        return (
            f"reduced-order coupled stage B (delta_w={self.delta_w*1e3:.1f} mm, D={self.D:.0f} m):\n"
            f"  water max climbs {self.water_max[0]-self.L:.2f} -> "
            f"{np.nanmax(self.water_max)-self.L:.2f} m above equilibrium as the crack seals\n"
            f"{line}\n"
            f"  condensation supply: {self.deposition_rate:.3e} mm/cycle"
        )


def evolve_geometry_coupled(
    cfg: Config,
    delta_w: float,
    *,
    w_eff_max: float = 0.08,
    forcing_model: str = "single-cosine",
    forcing_params: dict | None = None,
    Tb: float = 272.6,
    n_e: int = 25,
    w_floor: float = 2e-3,
    overflow_frac: float = 0.9,
    log_spacing: bool = True,
) -> CoupledEquilibriumResult:
    """Reduced-order coupled stage B at fixed absolute tidal swing ``delta_w``.

    The ice lining shifts both walls inward by the same amount, so it lowers the
    mean width but leaves the absolute swing ``delta_w = w_max - w_min`` fixed.
    We therefore sweep the effective minimum width ``w_eff`` from ``w_eff_max``
    down to ``w_floor``; for each value the width series is built with ratio
    ``R_eff = 1 + delta_w / w_eff`` and minimum ``w_eff``, and the liquid solver
    is re-run to record how high the water climbs. Because ``h_max`` grows as the
    gap closes (``h_max ~ delta_w^2 w_eff^{-3}``), the water reaches the surface
    once the crack has sealed far enough; ``w_eff_overflow`` is that seal depth
    (interpolated where the rise crosses ``overflow_frac * D``).
    """
    forcing_params = forcing_params or {}
    P = float(cfg.physical.orbital_period)
    L = float(cfg.physical.equilibrium_depth)
    D = L / 10.0
    t_in = np.arange(100, P + 1, 200.0)

    if log_spacing:
        w_eff = np.geomspace(w_eff_max, w_floor, n_e)
    else:
        w_eff = np.linspace(w_eff_max, w_floor, n_e)
    R_eff = 1.0 + delta_w / w_eff
    water_max = np.full(n_e, np.nan)

    dep_rate_mm = np.nan
    for i, we in enumerate(w_eff):
        w_in = build_width_series(t_in, R_eff[i], we, orbital_period=P,
                                  forcing_model=forcing_model, **forcing_params)
        w_rec, h_rec, t_rec, v_rec = liquid_dynamics(w_in, t_in, L, cfg)
        water_max[i] = L + float(np.nanmax(h_rec))
        if i == 0:
            # condensation supply at the least-sealed width (uncapped upper bound)
            r = np.full_like(t_rec, np.nan)
            budget = wall_mass_budget(t_rec, h_rec, w_rec, v_rec, r, cfg, Tb, n_z=300)
            dep_rate_mm = float(np.nanmax(budget.net_thickness) * 1e3)

    rise = water_max - L
    target = overflow_frac * D
    if np.nanmax(rise) >= target:
        order = np.argsort(rise)  # rise ascending; w_eff is the interpolant
        w_eff_overflow = float(np.interp(target, rise[order], w_eff[order]))
        overflow = True
    else:
        w_eff_overflow = float("nan")
        overflow = False

    return CoupledEquilibriumResult(
        w_eff=w_eff, R_eff=R_eff, water_max=water_max, L=L, D=D, delta_w=delta_w,
        w_eff_overflow=w_eff_overflow, overflow=overflow,
        overflow_frac=overflow_frac, deposition_rate=dep_rate_mm,
    )


# ---------------------------------------------------------------------------
# Full coupled stage B (z-dependent liquid solver in the loop)
# ---------------------------------------------------------------------------

@dataclass
class FullEvolutionResult:
    """Fully-coupled stage-B equilibrium using the z-dependent liquid solver."""

    zeta: np.ndarray
    e_ice: np.ndarray
    w_eff_min: np.ndarray
    L: float
    D: float
    water_max_height: float
    overflow: bool
    iterations: int
    converged: bool
    water_max_history: list = field(default_factory=list)

    @property
    def surface(self) -> float:
        return self.L + self.D

    def summary(self) -> str:
        wm = [f"{x - self.L:.1f}" for x in self.water_max_history]
        return (
            f"full coupled stage B:\n"
            f"  iterations: {self.iterations}  converged: {self.converged}  "
            f"overflow: {self.overflow}\n"
            f"  water rise above equilibrium: {self.water_max_history[0]-self.L:.2f} -> "
            f"{self.water_max_height - self.L:.2f} m (D={self.D:.0f})\n"
            f"  final distance below surface: {self.surface - self.water_max_height:.2f} m\n"
            f"  peak ice lining: {np.nanmax(self.e_ice)*1e3:.3e} mm/wall\n"
            f"  water-rise history (m above eq): {wm}"
        )


def evolve_geometry_full(
    cfg: Config,
    wmin: float,
    wmaxmin: float,
    *,
    forcing_model: str = "single-cosine",
    forcing_params: dict | None = None,
    Tb: float = 272.6,
    lookup=None,
    n_z: int = 400,
    n_t: int = 120,
    cycles_per_iter: float = 10.0,
    max_iter: int = 40,
    tol: float = 1e-5,
    rho_ice: float = RHO_ICE,
) -> FullEvolutionResult:
    """Fully-coupled stage B: re-run the z-dependent liquid solver each step.

    The ice lining ``e_ice(zeta)`` (above the water line) narrows the top neck;
    the tapered liquid solver then squeezes the water faster so it climbs
    higher, exposing more wall to condensation. Iterates until the water reaches
    the surface (overflow) or the lining converges.

    ``lookup`` is an optional :class:`GasLookupTable` for the choke cap; if
    ``None`` the deposition is uncapped (upper bound).
    """
    forcing_params = forcing_params or {}
    P = float(cfg.physical.orbital_period)
    L = float(cfg.physical.equilibrium_depth)
    D = L / 10.0
    surface = L + D
    Lv = float(cfg.physical.latent_heat)
    Lf = float(cfg.physical.latent_heat_fusion)
    phys, fp = cfg.physical, cfg.friction
    t_in = np.arange(100, P + 1, 200.0)

    zeta = _make_grid(surface, n_z)
    dep_flux = _condensation_flux(zeta, surface, Tb, Lv)
    ec = vapor_pressure(273.15) / np.sqrt(2.0 * np.pi * _RG)
    bv = Lv / _RG

    e_ice = np.zeros(len(zeta))
    water_max_history: list = []
    converged = overflow = False
    water_max = L
    it = 0
    for it in range(1, max_iter + 1):
        w_in = build_width_series(t_in, wmaxmin, wmin, orbital_period=P,
                                  forcing_model=forcing_model, **forcing_params)
        wt, ht, tt, vt = liquid_dynamics_tapered(
            w_in, t_in, L, cfg, e_ice_zeta=zeta, e_ice_values=e_ice)
        water_max = L + float(np.nanmax(ht))
        water_max_history.append(water_max)
        if (water_max - L) >= 0.999 * D:
            overflow = True
            break

        # subsample the last period
        m = tt >= (tt[-1] - P)
        tc, hc, wc, vc = tt[m], ht[m], wt[m], vt[m]
        dwc = np.gradient(wt, tt)[m]
        if len(tc) > n_t:
            sel = np.linspace(0, len(tc) - 1, n_t).round().astype(int)
            tc, hc, wc, vc, dwc = tc[sel], hc[sel], wc[sel], vc[sel], dwc[sel]
        dt = np.gradient(tc)

        if lookup is not None:
            rc, *_ = lookup.query_vectorized(Tb, D - hc, wc)
            rc = np.where(np.isfinite(rc), rc, np.nanmean(rc[np.isfinite(rc)]) if np.any(np.isfinite(rc)) else 0.77)
        else:
            rc = np.zeros(len(tc))  # uncapped: max inlet flux -> deposit on all exposed wall
        phi_t = (1.0 - rc) * evaporation_rate_simple(ec, bv, Tb)

        # Net ice budget: condensation where exposed (gated inside) minus
        # frictional melt where covered. No hard "ice-free up to water_max"
        # rule -- deposition is already exposure-gated, so always-covered wall
        # (below the tidal low-water line) never accretes, while the tidal
        # water-line band can, narrowing the neck the water flows through.
        dep = _deposition_per_cycle(zeta, dep_flux, tc, hc, wc, phi_t, rc, e_ice, L, dt)
        melt = _friction_melt_per_cycle(zeta, tc, hc, wc, vc, dwc, dt, fp, phys, Lf, L)
        e_new = e_ice + (dep - melt) / rho_ice * cycles_per_iter
        e_new = np.clip(e_new, 0.0, wt.min() / 2.0)
        step = float(np.nanmax(np.abs(e_new - e_ice)))
        e_ice = e_new
        if step < tol:
            converged = True
            break

    w_eff_min = np.clip(np.nanmin(wt) - 2.0 * e_ice, 0.0, None)
    return FullEvolutionResult(
        zeta=zeta, e_ice=e_ice, w_eff_min=w_eff_min, L=L, D=D,
        water_max_height=water_max, overflow=overflow, iterations=it,
        converged=converged, water_max_history=water_max_history,
    )
