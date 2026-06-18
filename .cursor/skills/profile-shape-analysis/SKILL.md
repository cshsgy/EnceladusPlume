---
name: profile-shape-analysis
description: Analyze pipeline output directories for two-peak plume-profile shape, including first-peak location, major-peak location, peak separation, and first-to-major amplitude ratio. Use when comparing `results*/` directories, evaluating single-cosine versus double-cosine forcing, or summarizing how close a sweep is to the target observational profile.
---

# Profile Shape Analysis

Use this skill to compare plume-profile shape across pipeline output directories.

## Quick Start

Run:

```bash
MPLCONFIGDIR="enceladus_plume/.mplconfig" \
XDG_CACHE_HOME="enceladus_plume/.cache" \
enceladus_plume/.venv/bin/python enceladus_plume/analyze_profile_shapes.py enceladus_plume/results_zoom
```

Swap the output directory for any run you want to inspect, such as `enceladus_plume/results_zoom_double_cosine_probe`.

## What The Script Reports

The script ranks cases by two-peak resemblance and prints:

- `first_peak_deg`
- `major_peak_deg`
- `peak_sep_deg`
- `first_to_major_ratio`

Use it as a fast filter before inspecting plots.

## Interpretation Rules

- Good candidates should have a moderate first peak and a stronger later major peak.
- Relative phase spacing matters more than absolute phase, because a global shift is acceptable.
- If the first peak is reported near `0 deg` for most single-cosine cases, that is evidence the forcing family is misaligned with the target shape.
- If double-cosine runs improve peak spacing while preserving a plausible first-to-major ratio, prefer retuning within that forcing family.

## Repository Findings

- Existing `results_zoom/` single-cosine cases cluster around a weak first peak near `0 deg` and a major peak near `180 deg`.
- The normalized double-cosine implementation was added to keep width bounds comparable to single-cosine sweeps.
- The first normalized double-cosine probe shifted the later secondary peak earlier for several tested cases, but it still did not produce a clear moderate peak near `50 deg`.
- A `45 deg` shifted-double-cosine probe moved the major peak later to about `201.5 deg`, but still failed to generate a distinct moderate first peak near `50 deg`.

## Example Requests

- "Compare `results_zoom` and the double-cosine probe by peak spacing."
- "Find the best current two-peak match in the latest output directory."
- "Tell me whether the new forcing moved the first peak away from `0 deg`."
