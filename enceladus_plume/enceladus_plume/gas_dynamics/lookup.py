"""Trilinear interpolation from pre-computed gas dynamics lookup table.

Port of interp_r_function.m (2023).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


def _clean_outliers(arr: np.ndarray, size: int = 3, threshold: float = 0.05,
                    max_passes: int = 8) -> np.ndarray:
    """Replace isolated outlier points with median-filtered values.

    A point is an outlier if it deviates from the local median by more than
    *threshold* (absolute).  The median filter window is *size* along each
    spatial axis (width, depth) and 1 along the Tb axis (which typically
    has only 2 points).

    The pass is **iterated** until no outliers remain (or *max_passes*). A
    single pass leaves residual glitches where several bad points cluster (the
    gas solver scatters near the choke boundary at small widths); each further
    pass cleans points whose neighbours were themselves fixed, and the residual
    converges to zero in a few passes.
    """
    kernel = [size, size, 1] if arr.ndim == 3 else size
    out = arr.copy()
    first_frac = 0.0
    passes = 0
    for passes in range(1, max_passes + 1):
        med = median_filter(out, size=kernel, mode="nearest")
        bad = np.abs(out - med) > threshold
        n_bad = int(np.sum(bad))
        if passes == 1:
            first_frac = 100.0 * n_bad / arr.size
        if n_bad == 0:
            passes -= 1
            break
        out[bad] = med[bad]
    if first_frac > 0:
        logger.info("  cleaned outliers (%.2f%% initially, converged in %d pass(es))",
                     first_frac, passes)
    return out


class GasLookupTable:
    """Wraps a pre-computed gas dynamics .npz file and provides fast
    interpolation of (r, phi, rho, phi0) as functions of (T, depth, width).
    """

    def __init__(self, path: str = "r_rec.npz", clean: bool = True):
        data = np.load(path)
        self.Tb = data["Tb"]
        self.depth = data["depth"]
        self.delta = data["delta"]

        r_rec = data["r_rec"]
        phi_rec = data["phi_rec"]
        rho_rec = data["rho_rec"]
        phi0_rec = data["phi0_rec"]

        if clean:
            logger.info("Cleaning lookup table outliers ...")
            phi_rec = _clean_outliers(phi_rec)
            rho_rec = _clean_outliers(rho_rec)
            phi0_rec = _clean_outliers(phi0_rec)
            r_rec = _clean_outliers(r_rec)

        points = (self.delta, self.depth, self.Tb)
        self._interp_r = RegularGridInterpolator(
            points, r_rec, bounds_error=False, fill_value=np.nan)
        self._interp_phi = RegularGridInterpolator(
            points, phi_rec, bounds_error=False, fill_value=np.nan)
        self._interp_rho = RegularGridInterpolator(
            points, rho_rec, bounds_error=False, fill_value=np.nan)
        self._interp_phi0 = RegularGridInterpolator(
            points, phi0_rec, bounds_error=False, fill_value=np.nan)

    def query(self, T: float, d: float, w: float
              ) -> tuple[float, float, float, float]:
        """Interpolate (r, phi, rho, phi0) for given (T, depth, width).

        Returns NaN for out-of-range queries.
        """
        pt = np.array([[w, d, T]])
        r = float(self._interp_r(pt))
        phi = float(self._interp_phi(pt))
        rho = float(self._interp_rho(pt))
        phi0 = float(self._interp_phi0(pt))
        return r, phi, rho, phi0

    def query_vectorized(
        self,
        Tb: float,
        depths: np.ndarray,
        widths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch-interpolate over arrays of (depth, width) at fixed Tb.

        Parameters
        ----------
        Tb     : base temperature (K), scalar broadcast to all points.
        depths : 1-D array of water-column depths (m).
        widths : 1-D array of crack widths (m), same length as *depths*.

        Returns
        -------
        r, phi, rho, phi0 : 1-D arrays, same length as inputs.
        """
        depths = np.asarray(depths, dtype=float)
        widths = np.asarray(widths, dtype=float)
        n = len(depths)
        pts = np.column_stack([widths, depths, np.full(n, Tb)])
        r = self._interp_r(pts)
        phi = self._interp_phi(pts)
        rho = self._interp_rho(pts)
        phi0 = self._interp_phi0(pts)
        return r, phi, rho, phi0


def interp_r_function(
    T: float, d: float, w: float,
    lookup: Optional[GasLookupTable] = None,
    path: str = "r_rec.npz",
) -> tuple[float, float, float, float]:
    """Convenience wrapper matching the MATLAB function signature.

    If a ``GasLookupTable`` instance is provided it is reused; otherwise a
    new one is created from *path* on every call (slow -- prefer passing
    the object).

    Returns (r, phi, rho, phi0).
    """
    if lookup is None:
        lookup = GasLookupTable(path)
    return lookup.query(T, d, w)
