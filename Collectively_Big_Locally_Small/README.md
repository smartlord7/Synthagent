# Collectively Big, Locally Small — LaTeX sources

The work is split into two papers. This folder holds the survey; its companion methods paper on Graph Hierarchical Borrowing (GHB) is in `../Graph_Hierarchical_Borrowing/`, and reads its figures, numbers, code and bibliography from here (see that folder's README). The survey keeps a design-level summary of GHB (Section 8) and points to the companion paper for the method for realistic heads, the full privacy analysis and the simulation study, whose details moved from this supplement to the companion's.

This is the LaTeX project for the survey *Collectively Big, Locally Small: Formalizing Small Data in Healthcare and Learning Privately Across Siloed Hospitals*. It also contains its online supplement and the code behind every figure and number.

The project was first rebuilt 1:1 from `recovered_main.pdf`, which is kept unmodified. That exact reconstruction is commit `425ec47` (see [Recovery](#recovery-from-recovered_mainpdf)). Later commits revise the paper for submission (see [Changes since the recovery](#changes-since-the-recovery)).

## Build

```sh
latexmk -pdf main.tex                         # the paper (52 pages, ACM review mode)
cd supplement && latexmk -pdf supplement.tex  # the online supplement (21 pages); build main.tex first
```

`main.tex` is set up for submission to ACM Computing Surveys (`review` mode: line numbers and ACM metadata). For a preprint, swap the two `\documentclass` lines at the top.

You need TeX Live with `acmart` (v2.03 or later), `algorithm2e`, `pifont`, `makecell`, `xr` and TikZ. All figures and generated numbers are committed, so no Python is needed to build. To regenerate them you need Python 3 with numpy, scipy, matplotlib 3.11 and requests:

```sh
python3 scripts/make_figures.py      # Figures 2 and 6-10 of the paper
python3 simulations/run_all.py       # Simulations 1-15, forest plot, macros (about 20 minutes)
python3 simulations/make_numbers.py  # supplement/results/sim_numbers.tex, s7_table.tex, s15_table.tex, s15_hyper.tex
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
| `supplement/` | Online supplement of the survey (`supplement.tex`: systematic search and corpus table), plus the simulation figures and generated results shared with the companion paper |

## Systematic search (`prisma/`)

Section 5.4 of the paper runs the protocol query on PubMed (E-utilities) and on medRxiv and the life-science part of arXiv via Europe PMC, for 2010 to 8 October 2026.

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

`ghb.py` implements plug-in GHB: a DerSimonian–Laird or half-Cauchy-posterior estimate of τ², an inverse-variance leave-one-out mean by gossip, and shrinkage. It also implements the Gaussian baselines. `ghb_logit.py` is GHB for logistic heads: privatized score releases at the anchor (zCDP), public moment-matching inversion, and each hospital's posterior mean with τ integrated out; `ghb.ghb_full_bayes` is the same rule in the Gaussian model. `logit_world.py` is the logistic-regression world (private releases centred on a fixed public reference, case-mix shift, wrong anchors, and the federated, meta-analytic and Ditto-style baselines).

| Script | Simulation |
|---|---|
| `exp1_validate.py` | 1. Closed forms of Propositions 8.1 and 8.2 |
| `exp2_heterogeneity.py` | 2. Heterogeneity |
| `exp3_equity.py` | 3. Equity and the number of hospitals |
| `exp4_gossip.py` | 4. Gossip rounds |
| `exp5_joint_release.py` | 5. Joint release |
| `exp6_gmrf.py` | 6. GMRF prior |
| `exp7_logistic.py` | 7. Logistic-regression heads |
| `exp8_baselines.py` | 8 and 10. Federated baselines; case-mix shift |
| `exp9_atypical.py` | 9. Atypical hospitals (robust prior, opt-out) |
| `exp11_anchor.py` | 11. A wrong anchor (conflict check) |
| `exp12_byzantine.py` | 12. Dishonest hospitals (release screening) |
| `exp13_dropout.py` | 13. Link failures during gossip |
| `exp14_accounting.py` | 14. Repeated releases (basic composition vs zCDP) |
| `exp15_literature.py`, `literature.py` | 15. Iterative federated learning at the same zCDP budget: DP-FedAvg (with fine-tuning), decentralized DP-SGD, clustered DP-FedAvg, and a local control from the anchor |
| `fig_forest.py` | Summary forest plot of Simulations 2-15 |
| `make_numbers.py` | LaTeX macros and tables read by the paper and supplement |

`dev/` holds the prototypes and development runs (separate seeds) that led to the revised GHB and to three extensions that were not adopted; see `dev/README.md`. Each script writes a figure to `supplement/figures/` and its numbers to `supplement/results/`. All seeds are fixed. The closed forms are in `scripts/blueprint.py`.

## Changes since the recovery

- **Corpus count.** The recovered text said the bibliography had 335 entries, 45 of them excluded background. The compiled bibliography has 329, and 39 of them are not in the corpus. The text now reads "329 identified, 39 excluded (X1 18, X2 5, X3 16)".
- **Systematic search.** The §5.4 protocol has been executed and reported with a PRISMA 2020 flow diagram (new Figure 6). The author note is removed, and the threats to validity, abstract, gap discussion (§7), observation 6 and agenda are updated with what the search confirms and what it does not.
- **Appendix A.** The corpus table moved to the online supplement. Three works cited only there are kept in the bibliography with `\nocite`.
- **§8.**
  - The claim that GHB is never worse than local estimation is now limited to known τ² and M.
  - The caption of Table 6 now says the method was checked in simulation, not on real data.
  - A new subsection, §8.7, presents the simulation study, with Figures 12–14 and Table 7; the supplement gives the full design.
- **Bibliography.**
  - Added Greenhalgh and Peacock (2005) and Chaudhuri et al. (2011).
  - Filled pages, volume, number, publisher and DOI for six entries from Crossref (title-matched).
  - 54 entries still lack page numbers. These are ICLR papers (no page numbers exist), arXiv preprints, and NeurIPS and PMLR papers, whose proceedings sites are blocked by this environment's network policy.
- **Theory hardened.**
  - **§2.** Sample complexity is now defined uniformly over a family of distributions; with a single known distribution it is trivially zero. The VC rate has its scope (0-1 loss), Riley's criteria are marked as surrogates, the effective sample size names its estimator, and the shift factor states its support condition.
  - **§4.**
    - The external resource is renamed Z to avoid a clash with M in §8.
    - The no-free-lunch proposition gains a risk statement valid for every θ and every loss, without relying on Fano's inequality.
    - The six channels are shown to be exhaustive by construction.
    - The matching text no longer contradicts the matrix.
  - **§8.**
    - Proposition 8.1 is shown to need only means and variances, and to bound the risk averaged over sites, not each site's.
    - "The same B" is now "almost the same B".
    - Proposition 8.2 states its clipping assumptions, and a new paragraph sets the scope of the privacy guarantee: patients at several hospitals, public site sizes, ℓ2 release of vector heads, and the sensitivity of non-mean estimators.
    - Proposition 8.4 gains part (iii), the leave-one-out estimate under gossip and its error.
    - Algorithm 1 no longer gossips the unprivatized variance V_k: only privatized heads leave a hospital.
    - The admissibility claim is corrected to dominance, the GMRF sum is over unordered pairs, and "orders of magnitude" is made precise.
- **Simulations 8–14 and §8.7.**
  - Federated baselines, case-mix shift, atypical and dishonest hospitals, a wrong anchor, link failures and repeated releases.
  - A summary forest plot and the privacy-boundary diagram (TikZ) are new.
  - Algorithm 1 now centres releases on a fixed public reference, and lists the optional safeguards: anchor conflict check, opt-out, release screening.
  - The privacy scope gains repeated-release accounting.
- **Simulation 15.** Iterative federated baselines at the same privacy budget, with tuned hyperparameters. DP-FedAvg with fine-tuning ties with GHB at moderate heterogeneity and wins when hospitals are nearly identical or the budget is strict, at the cost of making large hospitals worse off than alone. GHB wins with stronger or regional heterogeneity and with a wrong anchor. GHB was not changed after these results; its conservative estimate of τ² is identified as the bottleneck.
- **GHB revised after Simulation 15.** Logistic heads now release a privatized score at the anchor, inverted by public moment matching; each hospital integrates τ out over its posterior, weighted by its own marginal likelihood; the anchor conflict check is on by default. Developed on separate seeds and checked on four held-out settings. GHB is now the most accurate private method in six of ten settings of Simulation 15 and within 10% of the best wherever any method gains; three extensions (joint release, two rounds, a two-level regional prior) were tried and not adopted (supplement, Section S3).
- **Search.** The direct arXiv search was dropped; arXiv is covered only through Europe PMC's life-science subset, and the text says so.

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
