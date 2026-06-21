# Physical constants audit

Audit of every physical constant in the solver against literature / intended
values and for cross-module consistency. Most are correct; four issues are
flagged below (two are clear bugs, two need an author decision).

## Inventory

| Constant | Code value | Literature / intended | Status |
|----------|-----------|-----------------------|--------|
| Gravity `g` | 0.113 m/s² | Enceladus surface ≈ 0.113 | OK |
| Orbital/rotation period `P` | 118 800 s (33 h) | 32.9 h | OK |
| Stefan–Boltzmann `σ` | 5.67e-8 | 5.670e-8 | OK |
| Liquid density `ρ_w` | 1000 | water | OK |
| Liquid viscosity `μ` | 1.8e-3 Pa·s | water @273 K ≈ 1.79e-3 | OK |
| Vapor viscosity `μ_v` | 8.0e-6 Pa·s | water vapor ≈ 8–9e-6 | OK |
| Latent heat fusion `L_f` | 3.34e5 J/kg | water | OK |
| Vapor pressure `A`, `B` | 3.63e12 Pa, 6147 K | Nakajima & Ingersoll (2016) | OK |
| Heat-capacity ratio `γ` | 1.33 | water vapor | OK |
| Laminar const `C_lam` | 96 | parallel plates | OK |
| Exit Mach `M` | 1.6 | Dong et al. (2011), 1.4–1.8 | OK |
| Sky/equilibrium temp `T_e` | 68 K | S. polar effective T | OK (check) |
| Thermal diffusivity `κ` | 1.0e-6 m²/s | ice ≈ 1.1e-6 | OK |
| **Vapor gas constant `R`** | **8.341/0.018 = 463.4** | **8.314/0.018 = 461.9** | **BUG (typo)** |
| **Latent heat sublimation `L_v`** | **2.8e6 (gas) / 2.84e6 (rest)** | **≈ 2.83–2.84e6** | **BUG (inconsistent)** |
| **Thermal conductivity `k`** | **2.4 (gas/budget) / 3.0 (full)** | **ice ≈ 2.4–3.3** | **DECISION (inconsistent)** |
| **Wall friction `C_f` / model** | **churchill (var.); const 0.004 liquid / 0.002 gas** | **paper: const 0.002** | **DECISION (paper mismatch)** |

## Resolution (2026-06)

- **#1 FIXED** — `8.341 → 8.314` in `config.py`, `interpolator.py`, `gas_dynamics/solver.py`, `wall_budget.py`, and `cpp/.../gas.hpp` (rebuilt). Native parity holds.
- **#2 FIXED** — gas-solver `lv` default `2.8e6 → 2.84e6` (`interpolator.py`, `gas.hpp`, `bindings.cpp`); now consistent with config everywhere.
- **#3 FIXED** — `config.thermal_conductivity 3.0 → 2.4`, matching the gas/wall-budget `k`.
- **#4 RESOLVED (liquid only)** — the inconsistency was the **liquid** friction:
  the Methods stated a constant `C_f=0.002` but the code uses Churchill (1977),
  Reynolds-dependent. The Methods text was rewritten to describe Churchill (+
  `churchill1977` in the bib). The **gas** column was never the inconsistency —
  its constant `C_f=0.002` already matched the paper. Switching the gas to
  Churchill was tested and **reverted**: it chokes the gas column for essentially
  the whole cycle (the low-Reynolds startup region gives runaway laminar friction
  `C_f=C_lam/(8Re)\to\infty`), so the plume — and the two-peak structure —
  vanishes. The gas therefore keeps constant `C_f=0.002`.

Affected figure `peak_predictor.pdf` was regenerated for the corrected `R`/`L_v`
(gas friction unchanged); `wall_seal_regime.pdf` is pure-liquid and unaffected.

## Issues (original)

### 1. Vapor gas constant: `8.341` should be `8.314`  *(clear typo, ~0.32%)*
The universal gas constant is **8.314** J/(mol·K), not 8.341 — a digit
transposition. It propagates everywhere `R` is used (sound speed `√(1.33 R T)`,
vapor density, the Hertz–Knudsen prefactor, the choke condition).
Locations: `config.py:gas_constant_vapor`, `gas_dynamics/interpolator.py:_RG`,
`gas_dynamics/solver.py` (×2), `wall_budget.py:_RG`,
`cpp/include/enceladus/gas.hpp:detail::RG`.
**Fix:** replace `8.341` → `8.314` everywhere (rebuild C++).

### 2. Latent heat of sublimation: gas lookup uses `2.8e6`, everything else `2.84e6`  *(~1.4% inconsistency)*
The gas-column solver builds the lookup (and hence `phi_top`, the main mass-flux
peak) with `lv = 2.8e6`, while `config.latent_heat = 2.84e6` is what
`wall_budget`, `wall_geometry`, `peaks`, and `full_solver` use (and what the
paper quotes). The correct value is ≈ 2.83–2.84e6, so the gas solver's `2.8e6`
is the outlier.
Locations: `gas_dynamics/interpolator.py:39 (lv=2.8e6)`,
`cpp/include/enceladus/gas.hpp` (lv default).
**Fix:** make the gas solver use `2.84e6` (ideally read from config), rebuild C++.

### 3. Thermal conductivity: `2.4` (gas/wall budget) vs `3.0` (full solver)  *(decision)*
The wall surface-temperature balance that sets condensation — i.e. the entire
healing/attractor mechanism — uses `k = 2.4` (`interpolator.kt`, `wall_budget._KT`),
while the 2022 full solver uses `config.thermal_conductivity = 3.0`. Ice
conductivity is ~2.4 (270 K) to ~3.3 (low T); both are defensible, but they
should be **one value**.
**Decision needed:** pick the intended `k` and make all modules use it.

### 4. Wall friction: paper describes constant `C_f = 0.002`, code defaults to Churchill  *(decision)*
The Methods state `f = C_f ρ_w v²` with `C_f = 0.002`, but the default
`friction.liquid_model = "churchill"` (Reynolds-dependent), and the constant
fallback for the liquid is `0.004` (≠ 0.002). All liquid-dynamics results in
this work used Churchill, not the paper's constant `0.002`. `gas_Cf_constant`
does match the paper (0.002).
**Decision needed:** either update the Methods to describe the Churchill model,
or set `liquid_model = "constant"`, `liquid_Cf_constant = 0.002` to match the
paper. This affects the water-rise amplitude (and hence the attractor/peak
numbers).

## Impact
Issues 1–2 are small (≲1.5%) and change absolute amplitudes and the choke
boundary slightly; they do not change any qualitative conclusion (the attractor,
the no-threshold result, the peak phases). Issue 4 (friction model) can change
the water-rise amplitude more substantially and should be settled before the
quantitative numbers are quoted.
