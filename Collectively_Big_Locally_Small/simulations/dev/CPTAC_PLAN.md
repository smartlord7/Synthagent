# A second held-out set: CPTAC (plan, committed before any model is fitted on these data)

Written 2026-11-02. GHB is frozen (real_v3.GHB_V5); the literature's realistic configurations are frozen
(dev/v4_realistic_configs.json, chosen on the 12 TCGA design networks). Nothing below is tuned on CPTAC.

## Why

Five held-out TCGA networks cannot establish GHB's lead statistically (with n = 5 the smallest one-sided
Wilcoxon p is 1/32, which does not survive Holm over four tests). Every further TCGA project was already used, and no
other GDC program records the institution, so the new networks come from the Clinical Proteomic Tumor Analysis
Consortium (CPTAC-2 and CPTAC-3, open-access clinical data from the NCI GDC API).

## Networks (fixed from counts alone)

- Unit: CPTAC does not release the tissue source site. Its cases were collected by country-specific centres, so the
  unit is the patient's country of birth (cases with a missing, "Unknown" or "Not Reported" country excluded). This is
  a proxy for the collecting centre, and the paper must say so.
- Network: the cancer cohort (primary site and disease type: PDA pancreas, CM melanoma, LUAD lung adenocarcinoma,
  LSCC lung squamous, HNSC head and neck squamous, CCRCC kidney, UCEC uterus, STAD stomach, BRCA, COAD, OV, GBM).
- Outcome, features, exclusions and the minimum unit size (5 usable patients) exactly as in real_v3.py; age from
  diagnoses.age_at_diagnosis (CPTAC has no age_at_index).
- Inclusion rule of the TCGA held-out set: at least 4 units and at least 15 deaths. From the counts this gives
  three networks: CPTAC-PDA (5 units, 124 patients, 76 deaths), CPTAC-CM (5, 100, 35), CPTAC-LUAD (5, 108, 28).

## Run

`cptac_heldout.py`: 10 repetitions of five-fold cross-validation within units (the protocol of real_v3.py), the
three networks fitted together by GHB as one consortium (as the TCGA held-out networks were), every method of the
TCGA held-out table: Local (default and evidence prior), GHB, GHB-lite, and the six literature families with every
configuration of their grids, at epsilon 1 and 10. Realistic arm: the frozen configurations. Oracle arm: the best
configuration on each network's own test folds (reported, not tested).

## Tests (added to stats_tests.py as family 6, primary)

Unit = network, n = 8 (the 5 TCGA held-out networks and the 3 CPTAC networks), on relative log-loss (GHB / Local).
For epsilon in {1, 10}: H1 GHB better than Local; H2 GHB better than the strongest realistic competitor (the realistic
literature method with the lowest mean over the 8 networks). One-sided exact Wilcoxon signed-rank tests, Holm over
the 4, alpha 0.05. Estimates: mean paired difference with a 95% t interval. Reported, not tested: the oracle arm.
The CPTAC networks alone are reported descriptively. Every outcome is reported, including those against GHB.
