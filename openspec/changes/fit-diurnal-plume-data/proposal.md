## Why

We have shown that the two-peak diurnal emission is a self-sealing attractor and we predict the *phase* and *relative strength* of the two peaks, but we have never quantitatively **fit** the observed diurnal profile. Fitting closes the loop: it turns the forward model into an inversion of the Cassini observations for the physical crack parameters (absolute tidal swing `Δw` and effective source depth `L`), giving the paper a concrete, data-anchored result instead of an order-of-magnitude comparison.

## What Changes

- **Obtain the observed diurnal profile.** The target is the Ingersoll et al. (2020) slab-density-vs-mean-anomaly data shown in Fig 1. The repo copy (`Figures/Figure_1.pdf`) is a **flat raster with no vector paths or text** (confirmed: 0 drawings, 0 text objects), so the data is not directly extractable. Acquire it by either (a) locating the tabular/supplementary data from the source paper, or (b) digitizing a binned mean curve from the rasterized scatter via axis-calibrated pixel analysis. Persist the resulting `(mean_anomaly, slab_density[, sigma])` table in the repo. **If neither path yields usable data, stop and report to the user.**
- **Add a fitting routine.** Fit the model total mass-flux-vs-mean-anomaly curve (built from the existing `_flux_curve` / gas-lookup machinery) to the observed profile over a small set of free parameters, using least squares.
- **Report the fit.** Best-fit parameters with uncertainties, goodness-of-fit, and a data-vs-model overlay figure.
- **Leave a paper placeholder.** Add a clearly-marked placeholder subsection in `main.tex` to discuss the implications of the fitted parameters (what the inferred `Δw` and `L` say about the source), to be filled once the fit is reviewed.

## Capabilities

### New Capabilities
- `diurnal-fit`: acquiring the observed diurnal emission profile, fitting the forward model to it over a defined free-parameter set, and reporting best-fit parameters, uncertainties, and a data-vs-model overlay.

### Modified Capabilities
<!-- None. The forward flux model (liquid/gas/overflow) is used as-is; this change consumes it, it does not alter its requirements. -->

## Impact

- **New code**: a fitting module / CLI (e.g. `enceladus_plume/run_fit.py`) built on `peaks._flux_curve`, the gas `GasLookupTable`, and `scipy.optimize`.
- **New data**: a small committed data file with the observed `(mean_anomaly, slab_density)` table and its provenance (source + acquisition method).
- **New figure**: a data-vs-model overlay added via `make_figures.py`.
- **Manuscript**: a placeholder subsection in `writing/manuscript/main.tex`.
- **Dependencies**: `scipy.optimize` (already a dependency); `pymupdf` only if raster digitization is used (dev-only, not a runtime dep).
- **Key risk**: data availability — the only in-repo copy is a raster scatter. Mitigation is digitization of a binned curve; accuracy of the fit is bounded by this and must be stated.
