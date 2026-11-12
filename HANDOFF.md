# Project handoff: EnceladusPlume paper and code

Status as of 2026-11-12, written before moving to a new machine. It carries over
the working knowledge that lived in the local Claude memory
(`~/.claude/projects/-home-sam2-dev-EnceladusPlume/memory/`), which does not
migrate with the repositories.

## Repositories

| Repo | Local path | Remote | Role |
|---|---|---|---|
| Code | `~/dev/EnceladusPlume` | `github.com/cshsgy/EnceladusPlume` | Python solver, C++ core, analysis, fit caches |
| Paper | `~/dev/enceladus_plume_paper` | `github.com/cshsgy/enceladus_plume_paper` (Overleaf-linked) | Manuscript `main.tex`, SI `si.tex`, figures |
| Proposal | `~/dev/enceladus_proposal` | `github.com/cshsgy/enceladus_proposal` (Overleaf-linked) | NPP/ICAR proposal (on hold) |

- `EnceladusPlume/writing/` is an untracked, stale copy of an early manuscript.
  Never edit it; the manuscript lives only in the paper repo.
- Overleaf does not pull from GitHub automatically. After pushing, use
  Overleaf → GitHub → "Pull GitHub changes into Overleaf".
- The advisor's comments arrive as PDF annotations in
  `enceladus_plume_paper/feedbacks/` (gitignored). Extract them with pymupdf
  `page.annots()`; line numbers refer to the compiled PDF's line numbering.

## Paper status

- Title: "A self-sealing feedback keeps Enceladus's ocean vents open", for JGR: Planets.
- Authors: Sihe Chen, Andrew P. Ingersoll, Apurva Oza. It is NOT co-authored with Kang.
- The advisor's 18 comments from 9-04-26 are all addressed, and so is an
  internal read-through.
- Remaining items for the author: compile on Overleaf (the local build is XeTeX:
  main 19 pp, SI 9 pp), reply to the advisor, and replace the Zenodo DOI
  (`main.tex` Open Research section, currently 10.5281/zenodo.22111053) with a new
  archive of the final code at submission.

### Adopted physical model (the story of the paper)

- **Question:** what keeps the tiger-stripe cracks open against condensation on
  their cold walls, and why does the plume peak twice per orbit?
- **Attractor:** wall condensation narrows the crack. Friction makes the
  tidally pumped water rise grow steeply as the width falls, so the crack narrows
  until the water first reaches the surface. That width is the closure width δ*,
  found by bisection (`wall_geometry.closure_width`). Liquid at the mouth then
  stops further narrowing.
- **Two peaks from a single-cosine tide:**
  - The broad *wide-crack peak* is the observed secondary, near MA 40°.
  - The sharp *surface-approach peak*, when the water is within ~1 m of the
    surface and the vapor column is shortest, is the observed main peak, near
    MA 185°.
  - No second tidal harmonic is needed.
- **Adopted fit** (`results/diurnal_fit_free_single_sc_fine_d1.json`):

  | Quantity | Value |
  |---|---|
  | Δδ | 5.4 mm |
  | L | 1.55 km (D = L/10, so ice shell ≈ 1.7 km along the stripes) |
  | δ* | 4.5 mm |
  | σφ | 16° |
  | φ0 | 228° |
  | χ² | 41 for 24 points and 5 parameters (reduced 2.1) |
  | Cycle-mean emission | ~160 kg/s over 500 km |

- **MCMC** (`results/freefit_diagnostics/posterior_single_d1.npz`): Δδ = 5.30
  +0.10/−0.08 mm, L = 1.59 ± 0.04 km, σφ = 14 ± 1°. Δδ and L are anticorrelated
  (r = −0.34).
- **Digitization sensitivity** (SI Text S4): L = 1.58 +0.11/−0.16 km.
- **Depth profile:** the minimum is χ² = 40 at 1.55 km. There is a secondary
  minimum near 8 km at Δχ² ≈ +76. The text acknowledges it.
- **Second harmonic (SI Text S5):**
  - Adding α near the adopted fit gives χ² 40 → 36.5 at α = 0.03, which is not
    significant.
  - A separate strong-2f family (α ≈ 0.84, Δδ ≈ 57 mm) reaches only χ² = 50.
  - At fixed L = 3/5/8/12 km that family gives χ² 50/69/94/139.
- **Tidal phase:** φ0 puts the narrowest aperture at MA ≈ 228°, about 50° after
  apocenter. So the brightest emission marks the crack closing, not opening. This
  is written as a testable departure from the opening-at-apocenter picture.
- **Spill/overflow:** zero on the attractor by construction. It is stated as
  unquantifiable and outside the range of validity, and is never used as a
  constraint. τ (the reservoir residence time) is irrelevant.
- **Wall deposit:**
  - The deposition rate is 14 mm/orbit just below the mouth, which is the
    radiative limit, and falls roughly as 1/d.
  - Integrated, it is ~44 kg/m/wall/orbit, or ~370 kg/s for 500 km. This matches
    evaporation minus vent flux.
  - A thin lid fails under Δp ≤ 611 Pa when t ≲ 0.09 mm. Part of the ice melts in
    the liquid, which gives a return path.
  - The upper-bound column-averaged narrowing is ~63 µm/orbit. The time to reach
    δ* is ~4 months from 1 cm, ~5 yr from 10 cm and ~50 yr from 1 m.
- **Wall condensation law** (SI Text S2): latent heat is conducted to the cold
  surface over a path ~ depth d below the mouth (Ingersoll & Pankine 2010 with
  horizontal distance → depth) and radiated. E = −σ(Ts⁴ − Te⁴)/L_v per wall, with
  Ts the surface temperature.

### Superseded results (do not resurrect)

- The old "backflow" capped-column fit had Δδ 7 mm, L 8 km, α 0.82, φ2 123°,
  χ²/dof 2.87, plus "mode A/B" and a peak attributed to overflow. It was wrong for
  two reasons:
  - The surface barrier damped v0, so the displaced water went back down instead
    of spilling.
  - The 10-m cap layer suppressed the approach peak about 3×.
- Figures in `enceladus_plume_paper/Figures/superseded/` belong to that version.
- The intermediate double-cosine free-lip fits (Δδ 23–57 mm, L ~2 km) were either
  not self-consistent (interpolated or single-cosine δ*) or are the worse 2f family.

## Code status and how to reproduce

- **Key settings for the adopted model:**
  - `liquid_dynamics.surface_barrier = "free"` (the default is still "backflow").
  - `liquid_dynamics.barrier_delta = 1.0` m (the default is 10).
  - Fine gas table `results/gas_lut_fine.npz`.
  - `run_fit` exposes `BARRIER_MODE`, `BARRIER_DELTA_M`, `--barrier`,
    `--barrier-delta`.
- **Closure width:** `wall_geometry.closure_width` (bisection). The fit is
  needle-sharp in δ*: ±0.25 mm changes χ² several-fold. Always recompute δ* or use
  the fine grid; never use a coarse interpolation.
- **Committed caches** in `enceladus_plume/results/`:
  - `diurnal_fit*.json`
  - `gas_lut.npz` and `gas_lut_fine.npz`
  - `weff_grid_v2.npz`
  - `weff_grid_single_d1.npz`, the 20×20 bisection grid for the adopted model
  - `freefit_diagnostics/`, holding eval logs, posterior, sensitivity and depth
    profiles
- **Analysis scripts** in `enceladus_plume/analysis/` (run from
  `enceladus_plume/` with `OMP_NUM_THREADS=1 PYTHONPATH=.`):

  | Script | Purpose |
  |---|---|
  | `fit_selfconsistent.py` | Parallel Sobol + Nelder–Mead fit with δ* recomputed per trial. Environment variables `ENC_LUT` and `ENC_BARRIER_DELTA`; flags `--single`, `--tag`. |
  | `single_d1_common.py` | Shared setup for the adopted model |
  | `run_mcmc_single_d1.py` | MCMC posterior |
  | `sensitivity_single_d1.py`, `sens_fig_single_d1.py` | SI Text S4 and Fig S1 |
  | `fig_sweeps_single_d1.py` | Fig 5, the Δδ and L sweeps |
  | `fig_sweeps_2f_d1.py` | SI Fig S2, the α and φ2 sweeps |
  | `fig_selfsealing_d1.py` | Fig 4 |
  | `profile_depth_single.py`, `profile_depth_sigma_single.py`, `profile_depth_double.py` | Depth profiles |
  | `free_lip_decompose.py`, `overflow_partition.py`, `overflow_tau_sensitivity.py` | Diagnostics |

  `make_schema_fig.py` (in the package dir) makes Fig 3 and takes
  `--simple --barrier free --barrier-delta 1`.
- **Figures** are written straight into `~/dev/enceladus_plume_paper/Figures`, or
  wherever `ENC_PAPER_FIGS` points.
- **Tests:** `pytest tests` gives 63 passing in about 9 minutes.

## Environment notes (old machine; re-check on the new one)

- **Python:** the old machine had several interpreters. `/opt/tensorflow/bin/python3`
  (3.12) runs the code and tests but cannot pip-install. `/usr/bin/python3` has
  pypdf and pymupdf (user site). `/opt/anaconda3` was 3.10.
- **C++ core:** build it against the interpreter that runs the code:
  `cmake -S cpp -B cpp/build -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=<python>`,
  then `cmake --build cpp/build -j`. The `.so` lands in
  `enceladus_plume/enceladus_plume/` and is imported as
  `enceladus_plume._enceladus_core`. Verify with `tests/test_native_parity.py`.
  There is a pure-Python fallback.
- **TeX:** the old machine had no TeX Live. A static `~/bin/tectonic` (v0.15) was
  used: `tectonic -X compile main.tex`. It is XeTeX, so the Times font falls back
  and the page count runs about one page long; check the length on Overleaf.
  BibTeX apacite "others" warnings are harmless.
- **CPUs:** 160 cores. Fits and MCMC were parallelized with joblib or
  multiprocessing at about 60 workers. One self-consistent likelihood evaluation
  takes about 1 min (bisection); MCMC on the grid took about 3.5 h.

## Working preferences (from the author)

- Talk in Chinese and write the paper in English. Apply edits that can be executed
  directly. Discuss judgment calls such as wording choices, physics framing and
  scope before changing anything.
- **Settled terminology:**
  - The width is δ, with closure width δ*. The old δ_eff / w notation is retired.
  - Δδ is the "tidal width amplitude", never "swing" or "perturbation".
  - Phases are the "opening/closing phase".
  - The peaks are the "wide-crack peak" and the "surface-approach peak".
  - Use "second harmonic" or "semidiurnal term", never "double cosine".
  - Use "self-sealing", meaning narrowing but never closing.
  - American spelling.
- **AGU limits:** abstract ≤ 250 words (now 234), PLS ≤ 200 (now 193), key points
  ≤ 140 characters each.
- Commit messages end with a `Co-Authored-By:` line naming the AI model used.

## Proposal (on hold)

- NPP opportunity 0044-NPP-NOV26-ABProg-Astrobio (ICAR: Investigating Ocean Worlds).
  The deadline was 2026-11-01. The proposal was last synced with the paper's
  single-cosine result.
- **Known issues when resumed:**
  - The local build is 16 pages, over the 15-page cap.
  - The AI-use statement is still a red placeholder.
  - The bib entry `chen2026` still lists Kang as a co-author; remove it.
- The host team proposal is `~/dev/kwy_proposal.pdf`. Its research plan pages are
  scanned images; render them with pymupdf to read them.
