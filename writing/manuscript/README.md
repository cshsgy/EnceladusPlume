# Manuscript

New working version of the Enceladus plume paper:
*"Aerosols in the vacuum: modelling the time variability of the plume of Enceladus."*

Built with the AGU journal LaTeX template (`agujournal2019`). The entry point is
[`main.tex`](main.tex), seeded from the previous draft (`../reference.tex`, also
rendered in `../Enceladus_Draft.pdf`).

## Required assets

`main.tex` depends on a few files that are **not yet in this folder** and must be
added before it will compile:

| Asset | Used by | Notes |
|-------|---------|-------|
| `agujournal2019.cls` (+ `apacite`, `agujournal2019.bst`) | `\documentclass` / bibliography | Ships with the [AGU LaTeX template](https://www.agu.org/publish-with-agu/publish/author-resources/latex). Drop the class/style files here or install them in your TeX tree. |
| `enceladus.bib` | `\bibliography{enceladus}` | BibTeX database of the references cited in the draft. |
| `Figures/Figure_1.pdf` … `Figures/Figure_10.pdf` | `\includegraphics` | The 10 figures. See `Figures/README.md`. |

## Building

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

(Or `latexmk -pdf main.tex` once the assets above are present.)
