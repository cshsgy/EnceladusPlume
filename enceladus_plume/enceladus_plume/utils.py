"""Utility functions -- crack data processing, width model, I/O.

Ports of SimplifyCrackDynamicsData.m, width_new.m, and assorted helpers.
"""

from __future__ import annotations

import numpy as np

FORCING_SINGLE_COSINE = "single-cosine"
FORCING_DOUBLE_COSINE = "double-cosine"
FORCING_SHIFTED_DOUBLE_COSINE = "shifted-double-cosine"


def _normalize_periodic_profile(
    raw: np.ndarray | float,
    sample_raw: np.ndarray,
) -> np.ndarray | float:
    raw_min = float(np.min(sample_raw))
    raw_max = float(np.max(sample_raw))
    if abs(raw_max - raw_min) < 1.0e-12:
        profile = np.zeros_like(np.asarray(raw, dtype=float))
    else:
        profile = (np.asarray(raw, dtype=float) - raw_min) / (raw_max - raw_min)
    profile = np.clip(profile, 0.0, 1.0)
    if np.isscalar(raw):
        return float(profile)
    return profile


# ---------------------------------------------------------------------------
# Crack width model (analytic, 2022 solver)
# ---------------------------------------------------------------------------

def width_scale(
    phase: np.ndarray | float,
    wmaxmin: float,
    forcing_model: str = FORCING_SINGLE_COSINE,
    phase_offset_deg: float = 0.0,
    second_harmonic_phase_deg: float = 0.0,
    second_harmonic_scale: float = 1.0,
) -> np.ndarray | float:
    """Return the dimensionless width scale for a forcing law.

    ``phase`` is the orbital phase angle in radians. The returned value is
    clipped to remain positive so experimental forcing laws cannot drive the
    crack width negative.
    """
    phase_arr = np.asarray(phase, dtype=float)
    phase_shift = np.deg2rad(phase_offset_deg)
    phase_arr = phase_arr - phase_shift

    if forcing_model == FORCING_SINGLE_COSINE:
        eps = 0.5 * (wmaxmin - 1.0)
        scale = 1.0 + eps * (1.0 - np.cos(phase_arr))
    elif forcing_model == FORCING_DOUBLE_COSINE:
        sample_phase = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False) - phase_shift
        raw_profile = -(2.0 / 3.0) * np.cos(phase_arr) + (1.0 / 6.0) * np.cos(2.0 * phase_arr)
        sample_raw = -(2.0 / 3.0) * np.cos(sample_phase) + (1.0 / 6.0) * np.cos(2.0 * sample_phase)
        profile = _normalize_periodic_profile(raw_profile, sample_raw)
        scale = 1.0 + (wmaxmin - 1.0) * profile
    elif forcing_model == FORCING_SHIFTED_DOUBLE_COSINE:
        harmonic_phase = np.deg2rad(second_harmonic_phase_deg)
        sample_phase = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False) - phase_shift
        raw_profile = (
            -(2.0 / 3.0) * np.cos(phase_arr)
            + second_harmonic_scale * (1.0 / 6.0) * np.cos(2.0 * (phase_arr - harmonic_phase))
        )
        sample_raw = (
            -(2.0 / 3.0) * np.cos(sample_phase)
            + second_harmonic_scale * (1.0 / 6.0) * np.cos(2.0 * (sample_phase - harmonic_phase))
        )
        profile = _normalize_periodic_profile(raw_profile, sample_raw)
        scale = 1.0 + (wmaxmin - 1.0) * profile
    else:
        raise ValueError(
            f"Unknown forcing model {forcing_model!r}; expected "
            f"{FORCING_SINGLE_COSINE!r}, {FORCING_DOUBLE_COSINE!r}, or "
            f"{FORCING_SHIFTED_DOUBLE_COSINE!r}."
        )

    scale = np.maximum(scale, 1.0e-8)
    if np.isscalar(phase):
        return float(scale)
    return scale


def build_width_series(
    t: np.ndarray | float,
    wmaxmin: float,
    wmin: float,
    orbital_period: float = 1.37 * 86400,
    forcing_model: str = FORCING_SINGLE_COSINE,
    phase_offset_deg: float = 0.0,
    second_harmonic_phase_deg: float = 0.0,
    second_harmonic_scale: float = 1.0,
) -> np.ndarray | float:
    """Return crack width for one or more times and a forcing model."""
    phase = 2.0 * np.pi * np.asarray(t, dtype=float) / orbital_period
    widths = wmin * width_scale(
        phase,
        wmaxmin,
        forcing_model=forcing_model,
        phase_offset_deg=phase_offset_deg,
        second_harmonic_phase_deg=second_harmonic_phase_deg,
        second_harmonic_scale=second_harmonic_scale,
    )
    if np.isscalar(t):
        return float(widths)
    return widths


def width_new(z: float, t: float, wmaxmin: float, wmin: float,
              orbital_period: float = 1.37 * 86400,
              forcing_model: str = FORCING_SINGLE_COSINE,
              phase_offset_deg: float = 0.0,
              second_harmonic_phase_deg: float = 0.0,
              second_harmonic_scale: float = 1.0) -> float:
    """Crack width as a function of (z, t).

    The default is the legacy single-cosine model. Experimental forcing laws
    such as the double-cosine variant can be selected via ``forcing_model``.

    Returns width in metres (always >= 1e-8).
    """
    return float(
        build_width_series(
            t,
            wmaxmin,
            wmin,
            orbital_period=orbital_period,
            forcing_model=forcing_model,
            phase_offset_deg=phase_offset_deg,
            second_harmonic_phase_deg=second_harmonic_phase_deg,
            second_harmonic_scale=second_harmonic_scale,
        )
    )


# ---------------------------------------------------------------------------
# Simplify crack dynamics data to one period
# ---------------------------------------------------------------------------

def simplify_crack_dynamics_data(
    t_rec: np.ndarray,
    period: float,
    h_rec: np.ndarray,
    w_rec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the last full orbital period from recorded crack data.

    Port of SimplifyCrackDynamicsData.m.

    Returns (t_rec, h_rec, w_rec) trimmed to one period with small
    padding at both ends.
    """
    n = len(t_rec)
    # Find the start of the last full cycle
    i_start = n - 2
    for i in range(n - 2, 0, -1):
        phase_i = t_rec[i] % period
        phase_ip1 = t_rec[i + 1] % period
        if phase_i > phase_ip1 and (t_rec[-1] - t_rec[i]) >= period:
            i_start = i
            break

    # Find the end of the cycle starting at i_start
    j_end = i_start + 2
    for j in range(i_start + 2, n):
        if (t_rec[j - 1] % period) > (t_rec[j] % period):
            j_end = j
            break
    else:
        j_end = n - 1

    t_out = t_rec[i_start:j_end + 1] % period
    t_out[0] -= period
    t_out[-1] += period
    h_out = h_rec[i_start:j_end + 1]
    w_out = w_rec[i_start:j_end + 1]
    return t_out, h_out, w_out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results(path: str, **arrays) -> None:
    """Save result arrays to a .npz file."""
    np.savez(path, **arrays)


def load_results(path: str) -> dict[str, np.ndarray]:
    """Load result arrays from a .npz file."""
    return dict(np.load(path))
