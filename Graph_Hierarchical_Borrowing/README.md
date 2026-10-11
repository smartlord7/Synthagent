# Graph Hierarchical Borrowing — LaTeX sources

The methods paper *Graph Hierarchical Borrowing: Private, Serverless Collaboration for Networks of Small
Hospitals*, companion to the survey *Collectively Big, Locally Small* in `../Collectively_Big_Locally_Small/`.
The survey defines small data and reviews the field; it keeps a design-level summary of GHB and points here.
This paper develops GHB as a method, with its theory, privacy analysis and full simulation study.

## Build

```sh
latexmk -pdf ghb.tex                                   # the paper (28 pages, ACM review mode)
cd supplement && latexmk -pdf ghb_supplement.tex       # its online supplement (13 pages); build ghb.tex first
```

`ghb.tex` is set up for ACM Transactions on Computing for Healthcare (`review` mode). For a preprint, swap the
two `\documentclass` lines at the top.

## Shared assets

Nothing is duplicated. The paper reads from the survey's folder:

| Asset | Path |
|---|---|
| Figures of the theory (risk, gain, gossip, privacy boundary) | `../Collectively_Big_Locally_Small/figures/` |
| Simulation figures and tables | `../Collectively_Big_Locally_Small/supplement/figures/`, `.../supplement/results/` |
| Generated numbers (LaTeX macros) | `../Collectively_Big_Locally_Small/supplement/results/sim_numbers.tex` |
| Simulation code (`run_all.py`, `ghb.py`, `ghb_logit.py`, `dev/`) | `../Collectively_Big_Locally_Small/simulations/` |
| Bibliography and style | `../Collectively_Big_Locally_Small/references.bib`, `ACM-Reference-Format-recovered.bst` |

Regenerating the simulations (`python3 ../Collectively_Big_Locally_Small/simulations/run_all.py`) updates both papers.

## Structure

| Section | Content |
|---|---|
| 1 Introduction | The problem, what existing approaches give up, the idea, contributions, main findings |
| 2 Background and related work | Small data and siloed smallness; distributed statistics and empirical Bayes; federated and personalized learning; decentralized learning; privacy; what the literature leaves open |
| 3 Problem setting | Requirements (R1)–(R5), why existing data-exchange agreements are not enough, threat model, comparison of approaches |
| 4 Theory (Gaussian model) | Borrowing with an imperfect population estimate; the cost of privacy and its scope; gossip; unknown heterogeneity (τ integrated out); structured heterogeneity |
| 5 GHB for model heads | Architecture L0–L5; score releases and moment-matching recovery; protocol (Algorithm 1) and privacy boundary; why a principled default |
| 6 Experiments (`sections/experiments.tex`) | Fifteen simulations: evaluation protocol, estimation, equity, one-shot and iterative baselines, privacy, adversarial tests |
| 7 Discussion | When to use GHB; limits; what the simulations do not show; open problems |
| Supplement | Full design of every simulation; the revision of GHB, development runs and extensions tried |
