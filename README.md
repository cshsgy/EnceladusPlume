# Enceladus Plume

A model of the time-variable water-vapor plume of Enceladus.

Liquid water in the tidally-flexing south-polar cracks rises and falls as the
walls widen and narrow over the 33-hour diurnal cycle. As the walls are
periodically covered and uncovered, sublimation and condensation drive a
vapor flow out of the crack. The model couples three physical processes —
**liquid-column dynamics**, **gas-column dynamics**, and **wall heat
diffusion** — to compute the water level and the water-vapor mass flux, and to
explain the observed two-peak diurnal plume profile (main peak near mean
anomaly 180°, secondary peak near 50°).

The accompanying manuscript is *"Aerosols in the vacuum: modelling the time
variability of the plume of Enceladus"* — see [`writing/`](writing/).

## Repository layout

```
enceladus_plume/     Python solver (current version)
cpp/                 C++ performance core + pybind11 bindings (optional, accelerates hot loops)
writing/
  manuscript/        Working LaTeX manuscript (current version)
  reference.tex      Source of the previous draft
  Enceladus_Draft.pdf  Rendered previous draft
```

> The original MATLAB implementation has been retired. It remains available in
> the git history (removed after the Python port reached parity).

## Physics modules (`enceladus_plume/enceladus_plume/`)

| Module | Role |
|--------|------|
| `liquid_dynamics/` | Water level + entrance velocity solver (adaptive RK, friction integral) |
| `gas_dynamics/`    | Gas-column solver and pre-computed lookup-table interpolation |
| `heat_diffusion/`  | 1-D wall thermal-conduction layer |
| `physics.py`       | Shared thermodynamics (vapor pressure, evaporation, surface temperature) |
| `friction.py`      | Reynolds-dependent friction factors |
| `config.py`        | Typed dataclass config loaded from YAML |
| `workflows/`       | `full_solver` (2022 coupled) and `modular_solver` (2023 lookup-based) |

## Quick start (Python)

```bash
cd enceladus_plume
pip install -r requirements.txt        # or: pip install -e .
pytest -q                              # run the test suite (use -m "not slow" to skip slow tests)
```

Run the parameter-sweep pipeline:

```bash
./run_pipeline.sh                                  # default sweep -> results/
python run_pipeline.py --config config/run_config_zoom.yaml -v   # zoomed sweep -> results_zoom/
```

The pipeline runs the liquid dynamics for each `(wmin, width-ratio)` case,
builds or reuses the gas lookup table, interpolates the gas quantities,
and writes per-case `.npz` files plus plot grids and heatmaps. See
`enceladus_plume/README.md` for the full module reference and CLI options.

## C++ performance core (`cpp/`)

The numerically heavy inner loops — the liquid RK integrator with its friction
integral, and the gas lookup-table generation (RK4 column solve inside a
bisection) — dominate runtime. `cpp/` provides a C++ implementation of these
loops, exposed to Python via [pybind11](https://github.com/pybind/pybind11).

The Python package imports the compiled extension when it is available and
**falls back to the pure-Python implementation otherwise**, so the C++ core is
an optional accelerator, never a hard dependency. See [`cpp/README.md`](cpp/README.md)
for build instructions.
