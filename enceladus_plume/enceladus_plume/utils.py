"""Utility functions -- crack data processing, width model, I/O.

Ports of SimplifyCrackDynamicsData.m, width_new.m, and assorted helpers.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Crack width model (analytic, 2022 solver)
# ---------------------------------------------------------------------------

def width_new(z: float, t: float, wmaxmin: float, wmin: float,
              orbital_period: float = 1.37 * 86400) -> float:
    """Sinusoidal crack width as a function of (z, t).

    w(t) = wmin * [1 + eps * (1 - cos(omega * t))]
    where eps = 0.5 * (wmaxmin - 1).

    Returns width in metres (always >= 1e-8).
    """
    eps = 0.5 * (wmaxmin - 1.0)
    omega = 2.0 * np.pi / orbital_period
    w = (1.0 + eps * (1.0 - np.cos(omega * t))) * wmin
    return max(w, 1e-8)


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
