## ADDED Requirements

### Requirement: Observed diurnal profile acquisition
The system SHALL obtain the observed diurnal emission profile (slab density versus mean anomaly, Ingersoll et al. 2020) as a machine-readable table of `(mean_anomaly_deg, slab_density)` values, persisted in the repository together with a note recording its source and acquisition method. If a usable profile cannot be obtained by any available method, the system SHALL stop and report this to the user rather than fabricate data.

#### Scenario: Tabular data is available
- **WHEN** the source paper's tabular or supplementary data can be obtained
- **THEN** it is stored as the target profile with provenance recorded

#### Scenario: Only the raster figure is available
- **WHEN** no tabular data can be obtained and only the rasterized Fig 1 scatter exists
- **THEN** a binned mean curve is digitized from the raster via axis-calibrated pixel analysis, stored as the target profile, and its approximate (digitized) nature is recorded

#### Scenario: No data obtainable
- **WHEN** neither tabular data nor a digitizable figure yields a usable profile
- **THEN** the system reports the failure to the user and does not invent data

### Requirement: Forward-model fit to the observed profile
The system SHALL fit the model total mass-flux-versus-mean-anomaly curve, computed from the existing forward model (liquid dynamics, gas lookup, buffered overflow), to the observed profile by least squares over a defined set of free parameters: the absolute tidal swing `Δw`, the effective source depth `L`, a global mean-anomaly phase offset, and an amplitude scale relating emission to slab density. All other model constants SHALL remain fixed at their audited values.

#### Scenario: Fit converges
- **WHEN** the fitting routine is run against a valid target profile
- **THEN** it returns best-fit values for each free parameter within the physically allowed bounds

#### Scenario: Model curve is phase-aligned and scaled
- **WHEN** the model curve is compared to the observed profile
- **THEN** the single global phase offset and amplitude scale are applied consistently to the whole curve (not per-peak)

### Requirement: Fit reporting
The system SHALL report the best-fit parameters with uncertainty estimates and a goodness-of-fit metric, and SHALL produce a data-versus-model overlay figure suitable for the manuscript.

#### Scenario: Report produced
- **WHEN** the fit completes
- **THEN** best-fit parameters, uncertainties, a goodness-of-fit metric, and an overlay figure are emitted

### Requirement: Manuscript placeholder for implications
The system SHALL add a clearly-marked placeholder subsection to the manuscript for discussing the implications of the fitted parameters, so the discussion can be completed after the fit is reviewed.

#### Scenario: Placeholder added
- **WHEN** the fit is reported
- **THEN** a marked placeholder subsection exists in `main.tex` referencing the fit and the overlay figure
