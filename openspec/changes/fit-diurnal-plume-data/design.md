## Context

The forward model already produces a total mass-flux-versus-mean-anomaly curve for a sealed, near-overflow crack (gas evaporation `w·phi_top(w, D−h)` plus the buffered overflow term), exposed through `peaks._flux_curve`. What is missing is an inversion: fitting that curve to the observed diurnal profile to infer the physical crack parameters. The observed target is the Ingersoll et al. (2020) slab-density-vs-mean-anomaly data (repo Fig 1), which exists here only as a flat raster scatter (0 vector paths / 0 text), so a usable target curve must first be produced.

## Goals / Non-Goals

**Goals:**
- Produce a machine-readable observed profile `(MA, slab_density[, sigma])`, with provenance.
- Fit the forward model to it over a small, physically-motivated free-parameter set and report best-fit values, uncertainties, and a data-vs-model overlay.
- Add a marked manuscript placeholder for the implications discussion.

**Non-Goals:**
- Changing the forward flux model (used as-is).
- A precise instrument-level inversion — the fit is a first demonstration whose accuracy is bounded by the data-acquisition method and stated as such.
- Filling in the final scientific interpretation text (only a placeholder is added now).

## Decisions

**1. Data acquisition — tabular first, digitized binned curve as fallback.**
Attempt to obtain the source paper's tabular/supplementary data. If unavailable, digitize a binned curve from the rasterized Fig 1: rasterize at high DPI (pymupdf), calibrate the axes from tick pixel positions, classify non-background/non-axis pixels as data points, map to `(MA, slab_density)`, and bin in mean anomaly (~15° bins) taking the **median** per bin (robust to the dense scatter and overlapping year-group symbols) with the inter-quartile spread as `sigma`. All year-groups are combined (per-colour separation is unreliable at this resolution). The digitized nature and its uncertainty are recorded. *Alternative considered:* fit individual scatter points — rejected (colour/symbol de-blending from a raster is unreliable and adds no real information over a binned curve). If nothing works, stop and report.

**2. Free parameters — fit on the attractor: `{Δw, L, φ0, A}`.**
The crack is placed on the self-sealing attractor, so the effective width `w_eff` is not free but set by the overflow seal depth `w_eff*(Δw, L)`. Free parameters are the absolute tidal swing `Δw`, source depth `L`, a single global mean-anomaly phase offset `φ0`, and an amplitude scale `A` (emission ↔ slab-density proportionality, a nuisance parameter). *Alternative considered:* also free `w_eff` — rejected because `Δw` and `w_eff` are degenerate in setting the peak ratio, and fitting on-attractor is the physically consistent choice given the paper's thesis. To keep each objective evaluation cheap, precompute `w_eff*(Δw, L)` on a coarse grid with `evolve_geometry_coupled` and interpolate during optimization; each evaluation is then a single `_flux_curve` call.

**3. Objective & optimizer — weighted least squares, global then local.**
Minimize `Σ [(A·model(MA_i; Δw,L,φ0) − obs_i)/sigma_i]²`. Because the objective is multimodal in `φ0`, use a global pass (`scipy.optimize.differential_evolution`) followed by a local refine (`least_squares`); parameter uncertainties from the covariance (Jacobianᵀ·Jacobian) at the optimum, scaled by the reduced chi-square. Report chi²/dof.

**4. Reuse & reproducibility.**
Build the model curve with `peaks._flux_curve` + a `GasLookupTable` (reuse the persistent lookup used by `make_figures`, regenerating if absent). Ship the fit as a CLI (`run_fit.py`) that writes the params report and the overlay figure; add the overlay to `make_figures.py` for reproducibility.

**5. Manuscript placeholder.**
Add a new subsection near "Comparison with the observed emission" in `main.tex`, marked with an explicit placeholder comment, referencing the overlay figure and an inferred-parameter table, for the implications discussion to be completed after review.

## Risks / Trade-offs

- **Digitization accuracy** → fit parameters carry a systematic uncertainty beyond the formal covariance. Mitigation: use robust binned medians, report the acquisition method explicitly, and frame the fit as a first inversion, not a measurement.
- **Amplitude scale `A` is an unconstrained nuisance** (slab-density↔emission proportionality unknown) → only the *shape* (peak phases, peak ratio, trough depth) constrains `{Δw, L, φ0}`. This is acceptable and stated; absolute emission is already checked separately in the paper.
- **Parameter degeneracy (`Δw` vs `w_eff`)** → mitigated by the on-attractor constraint.
- **Multimodality in `φ0`** → mitigated by the global optimizer pass.
- **Attractor precompute cost** → mitigated by a coarse grid + interpolation rather than re-solving inside the optimizer.

## Open Questions

- Which observed subset to fit — all year-groups combined (default) vs a single representative epoch? Default is combined; revisit if the epochs differ systematically.
- Whether the source paper exposes tabular data at all (resolved during step 1; falls back to digitization otherwise).
