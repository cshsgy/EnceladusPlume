# Enceladus C++ performance core

C++ implementations of the Enceladus plume solver's numerically heavy inner
loops, exposed to Python via [pybind11](https://github.com/pybind/pybind11) as
the extension module `_enceladus_core`.

The Python package ([`enceladus_plume`](../enceladus_plume)) loads this
extension through `enceladus_plume._native` when it is present and **falls back
to pure Python otherwise** — the C++ core is an optional accelerator, never a
hard dependency.

## Layout

```
cpp/
  CMakeLists.txt          Build (fetches pybind11 via FetchContent)
  include/enceladus/      Header-only C++ implementations
    friction.hpp          Darcy/Fanning friction models
    physics.hpp           Vapor pressure, evaporation, surface-temperature bisection
    gas.hpp               Gas-column RK4 + bisection for the inlet ratio r
    liquid.hpp            Liquid-column derivative (velocity profile + friction integral)
  src/bindings.cpp        pybind11 module definition
```

## Building

Build against the **same interpreter** that will import the package (the one
that runs `pytest` / `run_pipeline.py`):

```bash
cmake -S cpp -B cpp/build \
      -DPYBIND11_FINDPYTHON=ON \
      -DPython_EXECUTABLE="$(command -v python3)"
cmake --build cpp/build -j
```

This writes `_enceladus_core.<abi>.so` directly into
`enceladus_plume/enceladus_plume/`, so the next `import enceladus_plume` picks
it up. Verify with:

```bash
cd enceladus_plume
python -c "from enceladus_plume._native import HAVE_NATIVE; print(HAVE_NATIVE)"
pytest tests/test_native_parity.py -q   # C++ vs Python parity
```

> This box has several Python interpreters. If CMake picks the wrong one, pass
> `-DPython_EXECUTABLE=...` explicitly and keep `-DPYBIND11_FINDPYTHON=ON`.

## Status / roadmap

Ported loops are wired into the Python solvers (`liquid_dynamics/solver.py::_derivative`
and `gas_dynamics/interpolator.py::_solve_single`) behind `HAVE_NATIVE`, with a
pure-Python fallback, and covered by parity tests against the Python reference.

- [x] `friction.hpp` — friction-factor models (parity-tested)
- [x] `liquid.hpp` — liquid-column derivative: velocity profile + friction integral + inertial term (parity-tested)
- [x] `gas.hpp` — gas-column RK4 + bisection lookup-table generation (parity-tested; ~10× faster than Python on `solve_r_function`)
- [ ] Heat-diffusion stencil (lower priority; small fraction of runtime)

Each new port should: keep the C++ numerically identical to the Python
reference, add a binding in `src/bindings.cpp`, wire the Python caller behind
`HAVE_NATIVE`, and extend `tests/test_native_parity.py`.
