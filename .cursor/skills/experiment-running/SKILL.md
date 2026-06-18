---
name: experiment-running
description: Run and review Enceladus plume parameter-sweep experiments driven by `enceladus_plume/run_pipeline.py`. Use when the user asks to launch the pipeline, tweak `run_config` sweep parameters, reuse cached liquid dynamics, rebuild the gas lookup table, or inspect generated `.npz` and plot outputs.
---

# Experiment Running

Use this skill for the repository's experiment workflow centered on `enceladus_plume/run_pipeline.py`.

## Quick Start

1. Work from `enceladus_plume/`.
2. Prefer the repo virtualenv:
   `source .venv/bin/activate`
3. Run one of:
   ```bash
   ./run_pipeline.sh
   ```

   ```bash
   python run_pipeline.py --config config/run_config.yaml -v
   ```

4. For the zoomed sweep:
   ```bash
   python run_pipeline.py --config config/run_config_zoom.yaml -v
   ```

## Pipeline Stages

`run_pipeline.py` executes six stages:

1. Run or load cached liquid dynamics for each `(wmin, wmaxmin)` pair.
2. Compute global depth and width ranges from the liquid results.
3. Build or reuse `lookup_table.npz`.
4. Interpolate gas quantities, compute overflow evaporation, and save one `.npz` per case.
5. Save `summary.npz`.
6. Generate plot grids and heatmaps as `.png`.

When summarizing a run, report which stage failed or completed.

## Config Knobs

Read the selected config before editing or running it. The keys that most often matter are:

- `sweep.wmin`: minimum slot-width values to sweep.
- `sweep.wmaxmin`: width-ratio values to sweep.
- `liquid_dynamics.*`: solver accuracy and time-stepping controls.
- `lookup_table.n_grid`: lookup resolution.
- `lookup_table.jobs`: parallel worker count.
- `lookup_table.Tb`: basal temperature samples.
- `skip_liquid`: reuse `output.directory/liquid_cache.pkl` if it already exists.
- `force_rebuild_lookup`: rebuild `lookup_table.npz` even if present.
- `output.directory`: destination for all run artifacts.

Default configs already in the repo:

- `config/run_config.yaml`: main sweep into `results/`.
- `config/run_config_zoom.yaml`: zoomed sweep into `results_zoom/`.

## Execution Workflow

Follow this sequence:

1. Read the target config and confirm the intended sweep and output directory.
2. If the user wants parameter changes, edit only the requested config values.
3. Check whether the target output directory already contains reusable artifacts such as `liquid_cache.pkl` or `lookup_table.npz`.
4. Run the pipeline with `-v` unless the user prefers quieter logs.
5. Watch the logs for per-case liquid timing, lookup-table reuse/build, and final artifact paths.
6. After completion, inspect the output directory and summarize the main artifacts.

## Reviewing Outputs

Expect these outputs inside `output.directory`:

- `liquid_cache.pkl`: cached liquid-stage results for reuse; keep local, do not commit.
- `lookup_table.npz`: gas lookup table.
- `summary.npz`: aggregate sweep metrics.
- `wmin*_ratio*.npz`: per-case timeseries and derived fields.
- `plot_liquid_grid.png`
- `plot_massflux_grid.png`
- `plot_summary_heatmap.png`
- `plot_phitop_grid.png`

When reviewing a completed run, mention:

- the output directory used
- number of sweep cases
- whether liquid dynamics were recomputed or loaded from cache
- whether the lookup table was rebuilt or reused
- whether summary and plot artifacts were written

## Notes

- `run_pipeline.sh` always activates `.venv` and uses `config/run_config.yaml`.
- `run_config_zoom.yaml` is set up to reuse liquid results with `skip_liquid: true` and force a lookup rebuild.
- The pipeline writes generated data under `results/`-style directories; avoid committing caches or large generated binaries unless the user explicitly asks.

## Example Requests

- "Run the default plume experiment."
- "Use the zoom config and tell me whether it reused cached liquid dynamics."
- "Increase the `wmaxmin` sweep, rerun, and summarize the new outputs."
- "Check whether the latest experiment produced `summary.npz` and the plot heatmap."
