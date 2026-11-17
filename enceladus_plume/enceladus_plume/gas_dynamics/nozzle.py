"""Variable-width gas column: 1-D unsteady compressible flow marched to steady state.

Per unit crack length, conserved U = (rho w, rho v w, rho E w) on a finite-volume
grid from the liquid surface (z = 0) to the mouth (z = Hg), with the local gap w(z).
Fluxes: HLL. Sources: area change p dw/dz, gravity, wall friction 2 Cf rho v|v| / w,
and wall exchange S = 2 Ev (kg m^-1 s^-1 per unit z, Ev <= 0: condensation).

Wall law (Ingersoll & Pankine 2010 closure, as in ``wall_budget``): the wall sits at
the saturation temperature of the local gas, Tw = T_sat(p), and the latent heat
q = (4k/(pi d))(Tw - Ts) is conducted to the surface over the distance d below the
mouth and radiated there, 2 sigma (Ts^4 - Te^4) = q.  Ev = -q / (2 Lv) per wall, as in
``physics.find_evap_surface``.  No sublimation
of the lining, no grain nucleation.  The liquid surface evaporates into the bottom
cell at the Hertz-Knudsen rate with the local back pressure.

Energy: adiabatic (gamma_eff) plus wall exchange, so a sonic throat cools the gas.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

SIGMA, KT, TE = 5.67e-8, 2.4, 68.0
A_VP, B_VP = 3.63e12, 6147.0


def p_sat(T):
    return A_VP * np.exp(-B_VP / T)


def T_sat(p):
    return B_VP / np.log(A_VP / np.maximum(p, 1e-30))


def ip_heat_flux(Tw, d):
    """Ingersoll-Pankine conducted/radiated flux q(Tw, d) [W/m^2], vectorized bisection on Ts."""
    Tw = np.asarray(Tw, float); d = np.asarray(d, float)
    c2 = 4.0 * KT / (np.pi * np.maximum(d, 1e-6))
    lo = np.full(np.broadcast(Tw, d).shape, 1.0); hi = np.broadcast_to(Tw, lo.shape).copy()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = 2 * SIGMA * (mid**4 - TE**4) + c2 * (mid - Tw)
        hi = np.where(f > 0, mid, hi); lo = np.where(f > 0, lo, mid)
    Ts = 0.5 * (lo + hi)
    return np.clip(c2 * (Tw - Ts), 0.0, None)


@dataclass
class NozzleResult:
    z: np.ndarray; w: np.ndarray; d: np.ndarray
    rho: np.ndarray; v: np.ndarray; p: np.ndarray; T: np.ndarray; M: np.ndarray
    Ev: np.ndarray            # wall exchange per wall area, kg m^-2 s^-1 (<= 0 condensation)
    flux_out: float           # vapor leaving the mouth, kg s^-1 per m of crack
    evap_in: float            # evaporation from the liquid surface, kg s^-1 per m of crack
    deposit: float            # total wall deposit, kg s^-1 per m of crack (both walls)
    converged: bool; steps: int; residual: float

    @property
    def z_sonic(self):
        i = np.where(np.nan_to_num(self.M) >= 1.0)[0]
        return float(self.z[i[0]]) if len(i) else np.nan

    def seal_rate(self, rho_ice=917.0):
        """Relative narrowing rate d ln w / dt [1/s] from both walls."""
        return 2.0 * (-self.Ev) / rho_ice / self.w


def make_grid(Hg, n, dz_top=0.05):
    """Cell edges from 0 to Hg: top cell dz_top, geometric growth downward."""
    if n * dz_top >= Hg:
        return np.linspace(0.0, Hg, n + 1)
    lo, hi = 1.0, 2.0
    for _ in range(100):                      # sum dz_top q^i = Hg
        q = 0.5 * (lo + hi)
        tot = dz_top * (q**n - 1) / (q - 1)
        lo, hi = (lo, q) if tot > Hg else (q, hi)
    dz = dz_top * q ** np.arange(n)[::-1]
    edges = np.concatenate([[0.0], np.cumsum(dz)]); edges *= Hg / edges[-1]
    return edges


def solve_gas_column(width_fn, Hg, *, Tb=272.6, gamma=1.33, rg=462.83, Lv=2.84e6, g=0.113,
                     Cf=0.002, n=600, dz_top=0.05, t_max=40.0, cfl=0.4, tol=1e-3,
                     check_every=2000, wall=True, verbose=False, t_min=None):
    """width_fn(z) -> gap [m] (z from the liquid surface up). Returns NozzleResult.

    Converged when the instantaneous mass balance evap = deposit + out closes to 2e-3 and the
    deposit and outflow (relative to the evaporation) change by less than ``tol`` between
    checks, after ``t_min`` seconds (default 10 s + Hg/20)."""
    edges = make_grid(Hg, n, dz_top); z = 0.5 * (edges[1:] + edges[:-1]); dz = np.diff(edges)
    w = np.maximum(width_fn(z), 1e-5); w_e = np.maximum(width_fn(edges), 1e-5)
    d = np.maximum(Hg - z, 0.5 * dz)                     # distance below the mouth
    cv = rg / (gamma - 1); cp = gamma * cv
    # tabulate the wall heat flux q(Tw, d) per cell for interpolation in Tw
    Tw_tab = np.linspace(TE + 1.0, Tb + 2.0, 80)
    q_tab = ip_heat_flux(Tw_tab[None, :], d[:, None])      # (n, 80)
    # initial state: saturated vapor at Tb, at rest
    rho = np.full(n, 0.5 * p_sat(Tb) / (rg * Tb)); v = np.zeros(n); T = np.full(n, Tb)
    U = np.stack([rho * w, rho * v * w, rho * (cv * T + 0.5 * v**2) * w])
    ps_b = p_sat(Tb); kin_b = 1.0 / np.sqrt(2 * np.pi * rg * Tb)

    def prim(U):
        rho = np.maximum(U[0] / w, 1e-12); v = U[1] / (rho * w)
        e = np.maximum(U[2] / (rho * w) - 0.5 * v**2, cv * 5.0)
        T = e / cv; p = (gamma - 1) * rho * e
        return rho, v, p, T

    def hll(rL, vL, pL, rR, vR, pR, wf):
        cL = np.sqrt(gamma * pL / rL); cR = np.sqrt(gamma * pR / rR)
        sL = np.minimum(vL - cL, vR - cR); sR = np.maximum(vL + cL, vR + cR)
        EL = pL / (gamma - 1) + 0.5 * rL * vL**2; ER = pR / (gamma - 1) + 0.5 * rR * vR**2
        FL = np.stack([rL * vL, rL * vL**2 + pL, (EL + pL) * vL]) * wf
        FR = np.stack([rR * vR, rR * vR**2 + pR, (ER + pR) * vR]) * wf
        UL = np.stack([rL, rL * vL, EL]) * wf; UR = np.stack([rR, rR * vR, ER]) * wf
        Fm = (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)
        return np.where(sL >= 0, FL, np.where(sR <= 0, FR, Fm))

    t = 0.0; steps = 0; converged = False; res = np.inf; last = None; S_last = np.zeros(n)
    t_min = (10.0 + Hg / 20.0) if t_min is None else t_min   # sound crossings + evaporation turnover
    Eb = 0.0; F_top = 0.0
    flux_hist = []
    while t < t_max:
        rho, v, p, T = prim(U)
        c = np.sqrt(gamma * p / rho)
        dt = cfl * np.min(dz / (np.abs(v) + c))
        # ghost states: bottom = reflective wall, top = near vacuum
        rL = np.concatenate([[rho[0]], rho]); vL = np.concatenate([[-v[0]], v]); pL = np.concatenate([[p[0]], p])
        rR = np.concatenate([rho, [rho[-1] * 1e-3]]); vR = np.concatenate([v, [max(v[-1], c[-1])]]); pR = np.concatenate([p, [p[-1] * 1e-3]])
        F = hll(rL, vL, pL, rR, vR, pR, w_e)              # (3, n+1)
        F_top = float(F[0, -1])                           # mass actually leaving the mouth
        dU = -(F[:, 1:] - F[:, :-1]) / dz
        # sources: area, gravity, friction
        dU[1] += p * (w_e[1:] - w_e[:-1]) / dz - rho * g * w - 2.0 * Cf * rho * v * np.abs(v)
        dU[2] += -rho * g * w * v
        # wall exchange (condensation only), limited to the mass in the cell
        if wall:
            Tw = np.clip(T_sat(p), Tw_tab[0], Tw_tab[-1])
            j = np.clip(np.searchsorted(Tw_tab, Tw) - 1, 0, len(Tw_tab) - 2)
            f = (Tw - Tw_tab[j]) / (Tw_tab[j + 1] - Tw_tab[j])
            q = q_tab[np.arange(n), j] * (1 - f) + q_tab[np.arange(n), j + 1] * f
            Ev = -q / (2.0 * Lv)                           # per wall area
            S = np.maximum(2.0 * Ev, -0.5 * rho * w / dt)   # kg m^-1 s^-1 per unit z
            S_last = S
            dU[0] += S; dU[1] += S * v; dU[2] += S * (cv * T + 0.5 * v**2)
        else:
            Ev = np.zeros(n); S_last = np.zeros(n)
        # liquid-surface evaporation into the bottom cell (Hertz-Knudsen with back pressure)
        Eb = max(ps_b * kin_b - p[0] / np.sqrt(2 * np.pi * rg * T[0]), 0.0) * w[0]
        dU[0, 0] += Eb / dz[0]; dU[2, 0] += Eb * cv * Tb / dz[0]
        U = U + dt * dU
        t += dt; steps += 1
        if steps % check_every == 0:
            rho, v, p, T = prim(U)
            fo = F_top; dep = float(np.sum(-S_last * dz)); ev = max(float(Eb), 1e-30)
            bal = (ev - dep - fo) / ev
            if last is not None:
                res = max(abs(fo - last[0]), abs(dep - last[1])) / ev
                if t > t_min and abs(bal) < 2e-3 and res < tol:
                    converged = True; break
            last = (fo, dep)
            if verbose:
                print(f"  t={t:7.2f}s step {steps:7d} out/evap {fo/ev:.4f} dep/evap {dep/ev:.4f} bal {bal:+.1e} res {res:.1e}", flush=True)
    rho, v, p, T = prim(U); M = v / np.sqrt(gamma * p / rho)
    vac = rho < 1e-8                                       # emptied cells: no meaningful T or M
    T = np.where(vac, np.nan, T); M = np.where(vac, np.nan, M)
    if wall:
        Tw = np.clip(T_sat(p), Tw_tab[0], Tw_tab[-1]); j = np.clip(np.searchsorted(Tw_tab, Tw) - 1, 0, len(Tw_tab) - 2)
        Ev = 0.5 * S_last                                 # realized exchange per wall (mass-limited)
    Eb = max(ps_b * kin_b - p[0] / np.sqrt(2 * np.pi * rg * T[0]), 0.0) * w[0]
    return NozzleResult(z=z, w=w, d=d, rho=rho, v=v, p=p, T=T, M=M, Ev=Ev,
                        flux_out=F_top, evap_in=float(Eb),
                        deposit=float(np.sum(-S_last * dz)), converged=converged, steps=steps, residual=res)
