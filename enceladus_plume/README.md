# Enceladus Plume Solver (Python)

Python port of the Enceladus south-polar plume MATLAB solver. Models water
vapor plume dynamics through tidally-flexing ice cracks on Enceladus.

## Features

- **Full coupled solver** (2022 paper version): crack liquid dynamics, gas
  dynamics with wall heat diffusion, adaptive RKF45 time-stepping.
- **Modular solver** (2023 revision): separate liquid dynamics and
  pre-computed gas dynamics lookup table for faster parameter sweeps.
- YAML-based configuration for all physical and numerical parameters.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full solver (2022-style)

```bash
python run_full.py --config config/default.yaml --wmin 0.01 --wmaxmin 2.0 --depth 20000
```

### Modular solver (2023-style)

```bash
python run_modular.py --config config/default.yaml --times times.txt --slips slips.txt
```

### Configuration

All physical constants and numerical parameters live in `config/default.yaml`.
Pass a custom YAML file via `--config` to override any value.

## Project Structure

```
config/default.yaml          - Default configuration
enceladus_plume/
  config.py                  - Configuration loader
  physics.py                 - Shared thermodynamic functions
  liquid_dynamics/
    solver.py                - RKF45 liquid level + velocity solver
    helpers.py               - Velocity profile, friction, auxiliary integrals
  gas_dynamics/
    solver.py                - Gas column solver (RK4 density + bisection)
    interpolator.py          - Pre-compute gas dynamics lookup table
    lookup.py                - Trilinear interpolation from lookup table
  heat_diffusion/
    solver.py                - 1-D wall heat diffusion
  workflows/
    full_solver.py           - 2022 coupled crack+gas+heat workflow
    modular_solver.py        - 2023 liquid+gas-lookup workflow
  utils.py                   - Crack data utilities, width model, I/O
tests/                       - Unit tests
run_full.py                  - CLI for the full solver
run_modular.py               - CLI for the modular solver
```
