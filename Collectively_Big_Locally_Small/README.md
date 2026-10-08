# Collectively Big, Locally Small — LaTeX sources

This directory holds the LaTeX project for the survey, rebuilt from `recovered_main.pdf`. The recovered PDF is the reference and is not modified.

## Build

```sh
python3 scripts/make_figures.py   # optional: regenerates figures/*.pdf
latexmk -pdf main.tex             # 52 pages, the same as recovered_main.pdf
```

You need TeX Live with `acmart` (v2.03 or later), `algorithm2e`, `pifont`, `makecell` and TikZ. You also need Python 3 with numpy, scipy and matplotlib 3.11. The figures are committed, so the Python step is optional.

## Files

| Path | Contents |
|---|---|
| `main.tex` | Full text, equations, tables, algorithm and TikZ diagrams |
| `references.bib` | 329 BibTeX entries; keys are `firstauthorYEARfirstword` |
| `ACM-Reference-Format-recovered.bst` | ACM style with one change: `n.dashify` emits `-` rather than `--`, so page ranges print as they do in the original |
| `data/corpus.csv` | The 290 coded works of Appendix A (Table 7) |
| `scripts/` | Python code behind every data figure and corpus count |
| `figures/` | Figures included by `main.tex` |
| `figures/original/` | Vector figures extracted from `recovered_main.pdf`, kept for reference |
| `figures/extracted/` | Raster images from `pdfimages` (masks and shading only; the paper has no photographs) |
| `figures/regenerated/` | Script output that is not byte-identical to the original and so is not included in the paper |

## Figure provenance

| Figure | Source | Notes |
|---|---|---|
| 1 Structure | TikZ in `main.tex` | Redrawn from PDF coordinates |
| 2 Deflation | `scripts/fig_deflation.py` | Pixel-identical. Riley n* = 2707, n_eff = 414 |
| 3 Scarcity classes | TikZ | |
| 4 Provenance channels | TikZ | |
| 5 Potential matrix | `scripts/fig_matrix.py` | Pixel-identical |
| 6 Corpus | `scripts/fig_corpus.py` | Pixel-identical, computed from `data/corpus.csv` |
| 7 Gap map | `scripts/fig_gapmap.py` | Pixel-identical, computed from `data/corpus.csv` |
| 8 Risk / ESS gain | `scripts/fig_risk.py`, `scripts/fig_ess_gain.py` | Pixel-identical. Parameters: σ²=1, τ²=0.02, K=12, n_j=200, C=1, δ=1e-5, n_k=50 |
| 9 Gossip | `figures/fig_gossip.pdf` is the original vector figure | `scripts/fig_gossip.py` reproduces the published spectral gaps (λ = 0.977, 0.964, 0.937, 0.933). The random graph seeds behind the plotted curves cannot be recovered, so its output is kept in `figures/regenerated/` |
| 10 GHB architecture | TikZ | |

No diagram is a raster image.

`scripts/corpus.py` reproduces every count in Sections 5–7 from the CSV: 290 works, 204 since 2019, 102 pooling works, 48 works with a secondary channel, 33 works with channel "none", and so on. Run `python3 scripts/corpus.py`.

## Fidelity to `recovered_main.pdf`

- **Pages:** 52 in both.
- **Text:** a word diff of `pdftotext -raw` output gives 31,654 words on each side, with a similarity ratio of 0.99981. The only difference is the extraction order of the equation number "(2)" relative to the Eq. (2) tokens, which is not a textual change.
- **Line and page breaks:** identical on every page.
- **Residual vertical offsets, all under 2 pt:**
  - p. 5: Eq. (2) underbraces, +0.56 pt
  - p. 10: −0.35 pt
  - p. 11: P6 list item, +1.66 pt, from list glue
  - p. 26: −0.35 pt
  - p. 27: Eq. (6) bracket, horizontal shift of ±0.83 pt
