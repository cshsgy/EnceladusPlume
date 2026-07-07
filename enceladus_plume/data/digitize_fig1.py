#!/usr/bin/env python3
"""Digitize the observed diurnal plume profile from Figure 1.

The manuscript Fig 1 (``writing/manuscript/Figures/Figure_1.pdf``) reproduces
the slab-density-vs-mean-anomaly scatter of Ingersoll et al. (2020, Icarus 344,
113345). That PDF is a flat raster (no vector paths or text), and no tabular
data is available in the repository, so the target profile for the diurnal fit
is digitized here directly from the figure.

Method
------
1. Rasterize Figure_1.pdf at a fixed resolution (pymupdf, Matrix(6, 6)).
2. Calibrate the axes from the numeric label positions (constants below were
   read from the deterministic Matrix(6,6) render; residuals < 3 px).
3. Classify data pixels = coloured (saturation > 0.25) or dark (near-black
   letter glyphs) pixels inside the plot frame, excluding the legend colour-bar
   box (which contains a near-black 2017 segment that would otherwise
   contaminate the trough).
4. Map data pixels to (mean anomaly, slab density) and bin in 15 deg bins,
   taking the median per bin (robust to the dense, multi-year scatter) with
   sigma = IQR / 1.349.

The result is written to ``observed_diurnal.csv``. This is a dev/one-time tool;
it requires ``pymupdf`` (not a runtime dependency of the package).

Reproduce:  python enceladus_plume/data/digitize_fig1.py
"""
from __future__ import annotations

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDF = os.path.normpath(os.path.join(
    _HERE, "..", "..", "writing", "manuscript", "Figures", "Figure_1.pdf"))
_OUT = os.path.join(_HERE, "observed_diurnal.csv")

# --- calibration for the Matrix(6,6) render (deterministic) ----------------
# plot frame (px) and numeric-label anchor positions -> data coordinates.
_FRAME = dict(L=191, R=1572, T=33, B=1510)
_XCOLS = np.array([191.0, 579.0, 960.0, 1344.0]); _XVALS = np.array([0, 100, 200, 300.0])
_YROWS = np.array([58.0, 335.0, 630.0, 925.0, 1220.0, 1495.0])
_YVALS = np.array([500, 400, 300, 200, 100, 0.0])
# legend colour-bar bounding box (px), incl. the near-black 2017 segment.
_LEGEND = dict(r0=1285, r1=1470, c0=267, c1=1168)
_BIN_DEG = 15.0


def digitize(save: bool = True):
    import fitz  # pymupdf
    pix = fitz.open(_PDF)[0].get_pixmap(matrix=fitz.Matrix(6, 6))
    im = (np.frombuffer(pix.samples, np.uint8)
          .reshape(pix.height, pix.width, pix.n)[..., :3].astype(float) / 255.0)
    H, W, _ = im.shape

    axs, axi = np.polyfit(_XCOLS, _XVALS, 1)
    ays, ayi = np.polyfit(_YROWS, _YVALS, 1)
    ma_of = lambda c: axs * c + axi
    sl_of = lambda r: ays * r + ayi

    m = 10
    interior = np.zeros((H, W), bool)
    interior[_FRAME["T"] + m:_FRAME["B"] - m, _FRAME["L"] + m:_FRAME["R"] - m] = True
    sat = im.max(-1) - im.min(-1)
    data = interior & ((sat > 0.25) | (im < 0.4).all(-1))
    data[_LEGEND["r0"]:_LEGEND["r1"], _LEGEND["c0"]:_LEGEND["c1"]] = False

    ys, xs = np.where(data)
    ma, sl = ma_of(xs), sl_of(ys)
    keep = (ma >= 0) & (ma <= 360) & (sl >= 0) & (sl <= 500)
    ma, sl = ma[keep], sl[keep]

    edges = np.arange(0, 361, _BIN_DEG)
    cen = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(len(cen), np.nan)
    sig = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        v = sl[(ma >= edges[i]) & (ma < edges[i + 1])]
        if v.size >= 20:
            med[i] = np.median(v)
            sig[i] = max((np.percentile(v, 75) - np.percentile(v, 25)) / 1.349, 5.0)
    ok = np.isfinite(med)
    table = np.column_stack([cen[ok], med[ok], sig[ok]])
    if save:
        np.savetxt(_OUT, table, delimiter=",",
                   header="mean_anomaly_deg,slab_density_kg_km,sigma", comments="")
        print(f"wrote {_OUT} ({len(table)} points)")
    return table


if __name__ == "__main__":
    digitize()
