## 1. Acquire the observed diurnal profile

- [x] 1.1 Attempt to obtain tabular/supplementary data for the Ingersoll et al. (2020) slab-density-vs-mean-anomaly profile (source paper / data repository)
- [x] 1.2 If tabular data is unavailable, digitize a binned curve from `Figures/Figure_1.pdf`: rasterize at high DPI, calibrate axes from tick positions, classify data pixels, map to `(MA, slab_density)`, bin in ~15° mean-anomaly bins taking the median and IQR spread
- [x] 1.3 Persist the target profile as a committed data file `(mean_anomaly_deg, slab_density, sigma)` with a provenance note (source + acquisition method + approximate/digitized flag)
- [x] 1.4 If no usable profile can be produced, stop and report to the user (do not fabricate data)

## 2. Set up the fit

- [x] 2.1 Precompute the attractor seal depth `w_eff*(Δw, L)` on a coarse grid via `evolve_geometry_coupled` and build an interpolator
- [x] 2.2 Wrap `peaks._flux_curve` (+ a `GasLookupTable`, reusing the persistent lookup) into a model function `model(MA; Δw, L, φ0)` on the attractor
- [x] 2.3 Define free parameters `{Δw, L, φ0, A}` with physical bounds and the weighted least-squares objective against the target profile

## 3. Fit and report

- [x] 3.1 Run a global optimization pass (`differential_evolution`) then a local refine (`least_squares`); extract best-fit parameters
- [x] 3.2 Estimate parameter uncertainties from the covariance at the optimum and compute chi²/dof
- [x] 3.3 Expose the fit as a CLI `run_fit.py` that prints the parameter report and writes the results
- [x] 3.4 Add a data-vs-model overlay figure (data points with error bars + best-fit model curve) to `make_figures.py` and generate it

## 4. Manuscript placeholder

- [x] 4.1 Add a clearly-marked placeholder subsection to `writing/manuscript/main.tex` near the observed-emission comparison, referencing the overlay figure and an inferred-parameter (`Δw`, `L`) table, for the implications discussion
- [x] 4.2 Recompile `main.pdf` and confirm the placeholder and figure resolve

## 5. Verify

- [x] 5.1 Report best-fit `{Δw, L, φ0, A}`, uncertainties, and chi²/dof to the user, and state the acquisition method and its accuracy caveat
- [x] 5.2 Check that the best-fit implied source depth and swing are physically plausible and consistent with the attractor/peak-ratio story
