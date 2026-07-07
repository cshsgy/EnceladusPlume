# Observed data

## `observed_diurnal.csv`

Target profile for the diurnal fit: **slab density vs mean anomaly**, columns
`mean_anomaly_deg, slab_density_kg_km, sigma` (24 rows, 15° bins).

**Source.** Ingersoll, Ewald & Trumbo (2020), *Time variability of the Enceladus
plumes*, Icarus **344**, 113345 — the slab-density-vs-mean-anomaly scatter
reproduced as Fig 1 of this manuscript (`writing/manuscript/Figures/Figure_1.pdf`).

**Acquisition method — digitized, not tabular.** The manuscript Fig 1 PDF is a
flat raster (no vector paths or text) and no tabular data was available, so this
profile was **digitized from the figure** by `digitize_fig1.py`: the scatter is
rasterized, the axes calibrated from the numeric labels (residuals < 3 px), the
legend colour-bar masked, and the data pixels binned in 15° bins taking the
**median** (robust to the dense multi-year scatter) with `sigma = IQR / 1.349`.

**Caveats.** These are digitized central-trend values, not the underlying
measurements: the per-bin median tracks the densest part of a wide, multi-year
scatter, `sigma` is the spread of that scatter (not a measurement error), and the
absolute slab density maps to plume emission only up to an unknown proportionality
(a free amplitude in the fit). Use for shape fitting (peak phases, peak ratio,
trough depth), not as calibrated measurements.

Regenerate: `python enceladus_plume/data/digitize_fig1.py` (requires `pymupdf`).
Visual check: `fig1_digitization_check.png` (digitized median over the scatter).
