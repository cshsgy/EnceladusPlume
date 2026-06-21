# Manuscript

New working version of the Enceladus plume paper:
*"Aerosols in the vacuum: modelling the time variability of the plume of Enceladus."*

Built with the AGU journal LaTeX template (`agujournal2019`). The entry point is
[`main.tex`](main.tex), seeded from the previous draft (`../reference.tex`, also
rendered in `../Enceladus_Draft.pdf`).

## Assets (in place)

The files needed to compile are now in this folder (from the original draft
bundle):

| Asset | Used by | Notes |
|-------|---------|-------|
| `agujournal2019.cls`, `trackchanges.sty` | `\documentclass` / preamble | AGU class + track-changes package. `apacite` is provided by a standard TeX distribution. |
| `enceladus.bib` | `\bibliography{enceladus}` | 31 references; covers every real citation in `main.tex`. |
| `Figures/Figure_1.pdf`, `Figure_2.pdf` | `\includegraphics` | Observation + schematic (kept). Figures 3–10 of the first draft were removed (misused constants). |
| `Figures/wall_seal_regime.pdf`, `peak_predictor.pdf` | `\includegraphics` | New figures generated from the solver. |

## Building

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

(Or `latexmk -pdf main.tex` once the assets above are present.)
