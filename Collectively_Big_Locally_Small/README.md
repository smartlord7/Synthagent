# Collectively Big, Locally Small — LaTeX sources

This is the LaTeX project for the survey *Collectively Big, Locally Small: Formalizing Small Data in Healthcare and Learning Privately Across Siloed Hospitals*. It also contains its online supplement and the code behind every figure and number.

The project was first rebuilt 1:1 from `recovered_main.pdf`, which is kept unmodified. That exact reconstruction is commit `425ec47` (see [Recovery](#recovery-from-recovered_mainpdf)). Later commits revise the paper for submission (see [Changes since the recovery](#changes-since-the-recovery)).

## Build

```sh
latexmk -pdf main.tex                         # the paper (52 pages)
cd supplement && latexmk -pdf supplement.tex  # the online supplement (26 pages); build main.tex first
```

You need TeX Live with `acmart` (v2.03 or later), `algorithm2e`, `pifont`, `makecell`, `xr` and TikZ. All figures and generated numbers are committed, so no Python is needed to build. To regenerate them you need Python 3 with numpy, scipy, matplotlib 3.11 and requests:

```sh
python3 scripts/make_figures.py      # Figures 2 and 6-10 of the paper
python3 simulations/run_all.py       # Simulations 1-7 (about 2 minutes)
python3 simulations/make_numbers.py  # supplement/results/sim_numbers.tex, s7_table.tex
# systematic search: queries the databases (slow), then screens, codes and counts
python3 prisma/search.py && python3 prisma/screen.py && python3 prisma/fetch_corpus_abstracts.py \
  && python3 prisma/classify.py && python3 prisma/verify.py && python3 prisma/sensitivity.py \
  && python3 prisma/flow.py && python3 prisma/fig_fieldmap.py
```

## Layout

| Path | Contents |
|---|---|
| `main.tex` | The paper |
| `references.bib` | 330 BibTeX entries |
| `ACM-Reference-Format-recovered.bst` | ACM style with one change: `n.dashify` emits `-` rather than `--`, matching the original |
| `figures/` | Figures of the paper; `fig_prisma.tex` is the TikZ PRISMA 2020 flow diagram |
| `data/corpus.csv` | The 290 hand-coded works |
| `scripts/` | Code for the corpus counts and the data figures of the paper |
| `prisma/` | The systematic search (see below) |
| `simulations/` | The simulation study of the blueprint (see below) |
| `supplement/` | Online supplement: `supplement.tex`, its figures, generated results and the corpus table |

## Systematic search (`prisma/`)

Section 5.4 of the paper runs the protocol query on PubMed (E-utilities), arXiv (its API), and medRxiv and arXiv via Europe PMC, for 2010 to 8 October 2026.

| Script | Does |
|---|---|
| `search.py` | Runs the queries and saves the raw records (`data/raw_*.jsonl.gz`) and `data/search_log.json` (exact query strings, hit counts) |
| `screen.py` | Deduplicates; screens titles and abstracts with explicit rules for I1, I2, X1–X3; writes `data/flow_search.json` |
| `verify.py` | Precision and NPV of the rules against the single-reviewer judgments in `data/verify_judgments.json` (200 records) |
| `fetch_corpus_abstracts.py` | Looks up every corpus work in Europe PMC (indexing status, abstract) |
| `classify.py` | Dictionary coder for channels, classes and modality; validates it against the hand coding; computes the recall of the search against the corpus and the field-level map |
| `sensitivity.py` | Recall of broadened query variants on two random halves of the corpus |
| `flow.py` | Writes every reported number as LaTeX macros (`prisma_numbers.tex`, `sensitivity_table.tex`), which `main.tex` and `supplement.tex` input |
| `fig_fieldmap.py` | Supplementary figure: corpus gap map next to the field-level map |
| `import_exports.py` | Adds RIS, BibTeX or CSV exports from Scopus, Web of Science, IEEE Xplore or ACM DL |

The subscription databases were not searched. To add them, run the query in each, export the results, and run `python3 prisma/import_exports.py scopus export.ris`. Then add the source name to `SOURCES` in `screen.py` and rerun the chain above.

`data/excluded_background.csv` gives the exclusion reason (X1/X2/X3) for each of the 39 identified works that are not in the coded corpus.

## Simulations (`simulations/`)

`ghb.py` implements plug-in GHB: a DerSimonian–Laird or half-Cauchy-posterior estimate of τ², an inverse-variance leave-one-out mean by gossip, and shrinkage. It also implements the baselines. The `exp1`–`exp7` scripts are Simulations 1–7 of the supplement:
1. Closed forms
2. Heterogeneity
3. Equity and the number of hospitals
4. Gossip rounds
5. Joint release
6. GMRF prior
7. Logistic-regression heads

Each script writes a figure to `supplement/figures/` and its numbers to `supplement/results/`. The closed forms are in `scripts/blueprint.py`.

## Changes since the recovery

- **Corpus count.** The recovered text said the bibliography had 335 entries, 45 of them excluded background. The compiled bibliography has 329, and 39 of them are not in the corpus. The text now reads "329 identified, 39 excluded (X1 18, X2 5, X3 16)".
- **Systematic search.** The §5.4 protocol has been executed and reported with a PRISMA 2020 flow diagram (new Figure 6). The author note is removed, and the threats to validity, abstract, gap discussion (§7), observation 6 and agenda are updated with what the search confirms and what it does not.
- **Appendix A.** The corpus table moved to the online supplement. Three works cited only there are kept in the bibliography with `\nocite`.
- **§8.**
  - The claim that GHB is never worse than local estimation is now limited to known τ² and M.
  - The caption of Table 6 now says the method was checked in simulation, not on real data.
  - A new subsection, §8.7, presents the simulation study, with Figures 12–14 and Table 7; the supplement gives the full design.
- **Bibliography.** Added Greenhalgh and Peacock (2005).

## Recovery from `recovered_main.pdf`

Commit `425ec47` (or `git show 425ec47:Collectively_Big_Locally_Small/main.tex`) is the 1:1 reconstruction. Checked against the recovered PDF:
- **Pages:** 52 on both sides.
- **Text:** a word diff of `pdftotext -raw` output gives 31,654 words each, similarity 0.99981. The only difference is the extraction order of the equation number "(2)".
- **Breaks:** identical line and page breaks.
- **Residual offsets:** vertical offsets under 2 pt on five pages (5, 10, 11, 26, 27).

Figure provenance:
- **TikZ redraws:** Figures 1, 3, 4 and 10 are redrawn from the PDF coordinates.
- **Pixel-identical regenerations:** Figures 2 and 5–8 come from `scripts/`.
- **Figure 9:** this is the original vector figure. `scripts/fig_gossip.py` reproduces its spectral gaps (λ = 0.977, 0.964, 0.937, 0.933) but not its random draws; its output is in `figures/regenerated/`.

`figures/original/` holds the vector figures extracted from the PDF, and `figures/extracted/` the raster masks from `pdfimages`.
