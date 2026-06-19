"""Predict the phase and strength of the two diurnal mass-flux peaks.

The diurnal water-vapor mass flux of an active (sealed, near-overflow) crack has
two peaks, both carried by the gas evaporation ``w * phi_top(w, depth)`` --- the
overflow liquid flux is negligible (<5%):

* the **widening peak**, near the maximum-crack-width phase (MA ~ 180-215 deg),
  with strength set by ``w_max`` and the gas throughput there; and
* the **approach peak**, near the maximum-water-level phase ``psi_h``
  (~ 270-330 deg), where the water rises close to the surface, the gas column is
  short, condensation losses are small and ``phi_top`` is large.

Which peak is the *main* one flips with parameters: the approach peak overtakes
the widening peak as the sealing drives the water closer to the surface
(``h_max/D -> 1``; see :mod:`enceladus_plume.wall_geometry`). With the single
global phase offset the observations allow, the (approach, widening) pair maps
onto the observed main / secondary peaks near MA 180 deg / 50 deg.

This module extracts the two peaks directly from the total mass-flux curve (gas
evaporation plus the small overflow term), which is accurate, rather than from a
closed-form estimate (the two-point ``w*phi_top`` ratio at the max-width and
max-water phases is only order-of-magnitude, ~x2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate
from .utils import build_width_series


@dataclass
class PeakPrediction:
    """Phases (deg mean anomaly) and strengths of the two mass-flux peaks."""

    phi_widening: float       # deg, widening-peak phase (~max-width)
    A_widening: float         # widening-peak mass flux (kg/m/s per unit length)
    phi_approach: float       # deg, approach-peak phase (~max water level); NaN if absent
    A_approach: float         # approach-peak mass flux; NaN if absent
    psi_h: float              # deg, phase of maximum water level
    hmax_over_D: float        # closeness of approach to the surface (1 = overflow)
    delta_w: float
    w_eff: float

    @property
    def has_two_peaks(self) -> bool:
        return np.isfinite(self.A_approach)

    @property
    def ratio(self) -> float:
        """Approach/widening strength ratio (NaN if no approach peak)."""
        return self.A_approach / self.A_widening if self.has_two_peaks else float("nan")

    @property
    def main_is_approach(self) -> bool:
        return self.has_two_peaks and self.A_approach > self.A_widening

    @property
    def main_phase(self) -> float:
        return self.phi_approach if self.main_is_approach else self.phi_widening

    @property
    def secondary_phase(self) -> float:
        if not self.has_two_peaks:
            return float("nan")
        return self.phi_widening if self.main_is_approach else self.phi_approach

    @property
    def peak_separation(self) -> float:
        """Approach minus widening phase (deg); the model's main/secondary spacing."""
        return self.phi_approach - self.phi_widening if self.has_two_peaks else float("nan")

    def summary(self) -> str:
        if not self.has_two_peaks:
            return (f"single peak (no close approach): widening at "
                    f"MA {self.phi_widening:.0f} deg, h_max/D={self.hmax_over_D:.2f}")
        which = "approach" if self.main_is_approach else "widening"
        return (
            f"two mass-flux peaks (delta_w={self.delta_w*1e3:.1f} mm, "
            f"w_eff={self.w_eff*1e3:.1f} mm, h_max/D={self.hmax_over_D:.2f}):\n"
            f"  widening: MA {self.phi_widening:5.0f} deg   A={self.A_widening:.3e}\n"
            f"  approach: MA {self.phi_approach:5.0f} deg   A={self.A_approach:.3e}\n"
            f"  A_approach/A_widening = {self.ratio:.2f}  (main = {which} peak)\n"
            f"  separation = {self.peak_separation:.0f} deg"
        )


def _refine(MA: np.ndarray, g: np.ndarray, idx: int) -> tuple[float, float]:
    """Parabolic sub-sample peak refinement; only for strict interior maxima."""
    if 0 < idx < len(g) - 1 and g[idx] > g[idx - 1] and g[idx] > g[idx + 1]:
        d = g[idx - 1] - 2 * g[idx] + g[idx + 1]
        if d != 0:
            dx = 0.5 * (g[idx - 1] - g[idx + 1]) / d
            return MA[idx] + dx * (MA[idx + 1] - MA[idx]), g[idx] - 0.25 * (g[idx - 1] - g[idx + 1]) * dx
    return float(MA[idx]), float(g[idx])


def _window_peak(MA, g, lo, hi):
    """Largest value in [lo, hi]; returns (phase, amp, is_interior_local_max)."""
    mask = (MA >= lo) & (MA <= hi)
    if not np.any(mask):
        return float("nan"), float("nan"), False
    idx = np.where(mask)[0][int(np.argmax(g[mask]))]
    interior = 0 < idx < len(g) - 1 and g[idx] > g[idx - 1] and g[idx] > g[idx + 1]
    phase, amp = _refine(MA, g, idx)
    return phase, amp, interior


def predict_peaks(
    cfg: Config,
    delta_w: float,
    w_eff: float,
    lookup,
    Tb: float = 272.6,
    forcing_model: str = "single-cosine",
    forcing_params: Optional[dict] = None,
    widen_window: tuple = (120.0, 215.0),
    approach_window: tuple = (215.0, 358.0),
) -> PeakPrediction:
    """Predict the two diurnal mass-flux peaks for a sealed crack.

    Parameters
    ----------
    cfg : configuration (``cfg.physical.equilibrium_depth`` sets L and D=L/10).
    delta_w : absolute tidal swing ``w_max - w_min`` (m).
    w_eff : effective minimum (sealed) width (m); ``R_eff = 1 + delta_w/w_eff``.
    lookup : a built :class:`~enceladus_plume.gas_dynamics.lookup.GasLookupTable`
        spanning the depth and width ranges visited by the run.
    """
    forcing_params = forcing_params or {}
    P = float(cfg.physical.orbital_period)
    L = float(cfg.physical.equilibrium_depth)
    D = L / 10.0
    t_in = np.arange(100, P + 1, 200.0)

    R = 1.0 + delta_w / w_eff
    w_in = build_width_series(t_in, R, w_eff, orbital_period=P,
                              forcing_model=forcing_model, **forcing_params)
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)

    dlo, dhi = lookup.depth.min(), lookup.depth.max()
    wlo, whi = lookup.delta.min(), lookup.delta.max()
    _, phi, _, _ = lookup.query_vectorized(
        Tb, np.clip(D - h, dlo, dhi), np.clip(w, wlo, whi))
    gas = np.nan_to_num(phi * w)

    # overflow evaporation (small but included): a fraction f_evap = Lf/(Lv+Lf)
    # of the water that exits the top is evaporated as it freezes/sublimates.
    Lv = float(cfg.physical.latent_heat)
    Lf = float(cfg.physical.latent_heat_fusion)
    f_evap = Lf / (Lv + Lf)
    rho_w = float(cfg.physical.liquid_density)
    overflow_dhdt = compute_overflow_rate(t, v, h, w_in, t_in, L, cfg)
    overflow = np.nan_to_num(f_evap * rho_w * w * overflow_dhdt)
    flux = gas + overflow  # total diurnal mass flux (per unit crack length)

    m = t >= (t[-1] - P)
    o = np.argsort(t[m] - t[m][0])
    MA = (t[m][o] - t[m][o][0]) / P * 360.0
    g = flux[m][o]
    hh = h[m][o]

    phi_w, A_w, _ = _window_peak(MA, g, *widen_window)
    phi_a, A_a, real = _window_peak(MA, g, *approach_window)
    if not real:                      # no distinct approach peak (no close approach)
        phi_a, A_a = float("nan"), float("nan")
    psi_h = float(MA[int(np.argmax(hh))])

    return PeakPrediction(
        phi_widening=phi_w, A_widening=A_w, phi_approach=phi_a, A_approach=A_a,
        psi_h=psi_h, hmax_over_D=float(hh.max() / D), delta_w=delta_w, w_eff=w_eff)
