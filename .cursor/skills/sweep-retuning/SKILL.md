---
name: sweep-retuning
description: Analyze the latest Enceladus plume sweep outputs, compare the simulated profile shape against a target observational profile, and update the next `run_config` sweep accordingly. Use when the user wants to tune `wmin` and `wmaxmin`, narrow or expand the sweep window, or iteratively match a target peak/trough profile.
---

# Sweep Retuning

Use this skill after at least one run of `enceladus_plume/run_pipeline.py`.

Current repository findings:

- the existing `results_zoom/` single-cosine runs tend to place the weak early peak near `0 deg` and the major peak near `180 deg`
- that default family therefore misses the desired moderate first peak near `50 deg`
- the experimental double-cosine forcing should be used in its normalized implementation so it preserves the same `wmin` to `wmax` bounds as the legacy forcing
- the literal unnormalized double-cosine formula can drive widths too close to zero at high `wmaxmin`, which makes the liquid solve numerically unstable
- in the initial normalized double-cosine probe, the later secondary peak moved earlier (roughly `264-285 deg` instead of `301-315 deg` for comparable cases), but the moderate first peak near `50 deg` was still not recovered in the tested range
- in the first shifted-double-cosine probe with `second_harmonic_phase_deg = 45`, the major peak moved later to about `201.5 deg`, but the model still did not create a distinct moderate peak near `50 deg`

The goal is not exact unit matching. Match the **shape** of the target profile:

- a moderate first peak near `50 deg`
- a stronger major peak later in the cycle
- the phase separation between the first and major peaks matters more than their absolute positions
- a steep decline after the major peak
- a low trough around `280-320 deg`
- a small recovery toward `340-360 deg`

Treat the pipeline's orbital `Phase` as mean anomaly scaled to degrees:

`mean_anomaly_deg = phase * 360`

Because the phase can be shifted, prioritize:

- relative spacing between the first and major peaks
- relative width and height of the two peaks
- the shape of the post-peak drop and trough

Do not reject a candidate only because both peaks are shifted together in absolute phase.

## Inputs To Inspect

Start from the selected output directory, usually `results/` or `results_zoom/`.

Inspect:

- `summary.npz`
- `plot_massflux_grid.png`
- `plot_summary_heatmap.png`
- any promising `wmin*_ratio*.npz` files

Within each case, focus on `mass_flux` first. Use `phi_top` only as a secondary diagnostic.

## Retuning Workflow

1. Identify the current best-looking case from `plot_massflux_grid.png`.
2. Compare that case's `mass_flux` curve against the target shape.
3. Describe the mismatch in plain language:
   - first peak missing, too weak, too strong, too broad, or too narrow
   - major peak too weak or too strong
   - peak separation too small or too large
   - first peak to major peak amplitude ratio off
   - post-peak decline too shallow or too steep
   - trough too shallow or too deep
4. Update the next sweep by editing only `sweep.wmin` and `sweep.wmaxmin` unless the user asks for solver or lookup changes.
5. Keep the next sweep centered around the most promising cases rather than jumping to a completely different region.

If no single-cosine sweep can recover the observed two-peak structure, consider changing the forcing law before expanding the sweep further.

## Default Retuning Rules

Use these heuristics unless the latest results clearly suggest otherwise:

- If the best peak occurs at too small a mean anomaly, try **larger** `wmaxmin`.
- If the best peak occurs at too large a mean anomaly, try **smaller** `wmaxmin`.
- If the main peak is too weak or the whole curve is too flat, include **smaller** `wmin`.
- If the curve is too strong everywhere or the baseline is too high, include **larger** `wmin`.
- If the first peak is missing but the major peak looks plausible, first densify the local sweep around the best case.
- If the first peak remains missing across that local sweep, try a different forcing law rather than only pushing `wmin` and `wmaxmin` farther.
- If the first peak is present but too weak, add nearby smaller `wmin` values while keeping the best `wmaxmin` region.
- If the first peak is too strong relative to the major peak, add nearby larger `wmin` values.
- If the first-to-major peak spacing is wrong but the overall shape is promising, prioritize local retuning that preserves amplitude while shifting relative timing.
- If both peaks move together under retuning, compare them after allowing a global phase shift and focus on the residual spacing error.
- If a second-harmonic phase shift mostly moves the major peak but still fails to create the early moderate peak, do not keep tuning that shift blindly; consider changing harmonic weights or adding a richer forcing family.
- If only one corner of the sweep looks promising, create a denser local sweep around that corner.
- If the optimum appears between sampled values, insert intermediate values instead of only expanding outward.

## Forcing-Law Option

The default interpretation of the sweep assumes a single-cosine width forcing. If that family cannot produce the moderate first peak plus stronger second peak, try a double-cosine option motivated by tidally heated forcing.

Suggested alternative:

```text
delta(t) = delta_min * (
    (R_delta + 1) / 3
    - 2 * (R_delta - 1) / 3 * cos(2 * pi * t / P)
    + (R_delta - 1) / 6 * cos(4 * pi * t / P)
)
```

Use this option when:

- the major peak can be matched but the earlier moderate peak is persistently absent
- the simulated curve only produces one broad maximum instead of two distinct peaks
- the relative peak spacing is wrong even after trying a local sweep and allowing a global phase shift

When trying this option:

- keep the sweep centered on the previous best `wmin` and `wmaxmin` region
- change only the forcing law first, so the comparison is interpretable
- use the normalized code-path, not the literal raw formula, so `wmin` and `wmaxmin` remain comparable to single-cosine runs
- use a separate output directory or clearly labeled follow-up config
- compare single-cosine and double-cosine runs by peak spacing, first-to-major amplitude ratio, and post-major decline shape

## How To Edit The Next Sweep

Prefer small, interpretable adjustments:

- keep 1 to 3 previous good values
- add 2 to 4 new neighboring values around the apparent optimum
- remove values that are clearly far from the target shape
- keep output directories separate if preserving old results matters
- if testing a new forcing law, hold the sweep region nearly fixed for one comparison run

Good pattern:

- broad sweep -> identify promising region -> zoom sweep -> compare again

Avoid:

- changing `wmin` and `wmaxmin` so aggressively that the comparison becomes hard to interpret
- changing solver tolerances during shape-matching unless there is a numerical problem

## Recommended Response Format

When using this skill, summarize the retuning decision with:

1. Best current candidate case(s)
2. Main mismatch versus target profile
3. Proposed new `wmin` list
4. Proposed new `wmaxmin` list
5. Whether to keep single-cosine forcing or try double-cosine forcing
6. Which config file was updated
7. Whether the next run should reuse `skip_liquid` or rebuild the lookup table

## Practical Notes

- `plot_massflux_grid.png` is the fastest way to compare shapes across the sweep.
- `summary.npz` is useful for ranking candidates by average flux and overflow fraction, but shape matching should be decided from the timeseries.
- For this target, matching the two-peak structure and their phase separation is more important than matching absolute phase.
- If changing the forcing law, prefer a controlled A/B comparison before also moving the sweep window.
- `run_config_zoom.yaml` is the right place for targeted follow-up sweeps.
- Keep generated `results/` artifacts and caches out of git unless the user explicitly requests otherwise.

## Example Requests

- "Analyze the latest sweep and retune the zoom config toward the observed profile."
- "Find the case whose mass-flux curve most resembles the target and update the next sweep."
- "Narrow the sweep around the best peak near 200 degrees."
- "Try a double-cosine forcing because the first peak is still missing."
