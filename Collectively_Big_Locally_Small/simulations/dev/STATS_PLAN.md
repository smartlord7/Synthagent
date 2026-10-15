# Statistical test plan for the GHB v5 claims

Written on 2026-10-15, after the results of every data set below were known (they are in the paper). Only the
predictions were pre-registered (PREREG_v5.md, PREREG_image.md); this test plan was not. To keep the choice of tests
from being steered by the results, it uses the conventional tests, the strongest comparators and a standard
multiplicity correction, and it is committed before any test is computed. Every outcome is reported, including the
ones that go against GHB. The code is `simulations/stats_tests.py`; its output is `supplement/results/stats_tests.json`
and `supplement/results/tables/stats_*.tex`.

## Units and data

- The unit of evidence for a claim about networks is the **network** (a set of hospitals). Repetitions within a network
  only average out the randomness of splits and privacy noise, so they are not independent evidence about new networks.
- Lower is better for relative error (error divided by Local's error, 1 = working alone); higher is better for balanced
  accuracy and C-index.
- Confirmatory data (not used to develop v5): the 5 TCGA confirmation networks (v5_confirm.json) and Camelyon17
  (camelyon_image.json). Development data (used while developing v5): the 7 small networks, FLamby Heart, BRCA and
  ISIC 2019. Tests on development data are descriptive and cannot confirm anything.

## Family 1 (primary, confirmatory): TCGA confirmation networks, unit = network, n = 5

For epsilon in {1, 10}, with d_k = (GHB v5 relative error) - (comparator relative error) on network k:

- H1(eps): GHB v5 is better than Local (comparator relative error = 1).
- H2(eps): GHB v5 is better than the strongest realistic competitor, defined as the literature method in the realistic
  arm with the lowest mean relative error over the 5 networks at that epsilon (chosen as the strongest, which can only
  favour the comparator).

Test: one-sided exact Wilcoxon signed-rank test on the 5 paired differences (with n = 5 the smallest attainable p is
1/32 = 0.031). Multiplicity: Holm over the 4 hypotheses, familywise alpha = 0.05.
Estimates: mean paired difference with a 95% t interval over networks, and the number of networks where GHB is better.
Not tested, reported only: the difference to the oracle-tuned best literature method (tuned on the test data, so it is
not an attainable competitor), as a mean with a two-sided 95% t interval.

## Family 2 (confirmatory, one network): Camelyon17, unit = repetition, n = 10

The four predictions of PREREG_image.md, on the pre-registered primary metric (balanced accuracy, mean over the five
hospitals), tested over the 10 paired repetitions with one-sided exact Wilcoxon signed-rank tests, Holm over the
family of tests:

- eps = 10: GHB above Local; GHB above every realistic literature method (one test per method).
- eps = 1: non-inferiority, GHB not more than 0.005 below Local (test of d + 0.005 > 0).
- eps = inf: GHB above Local; GHB above Pooled.

Repetitions share one network, so these tests speak about this network only.

## Family 3 (descriptive): all real data sets, unit = data set, n = 16

The 5 TCGA confirmation networks, the 7 small networks, FLamby Heart, BRCA and ISIC 2019, and Camelyon17, each on its
paper metric. For epsilon in {1, 10}: is GHB v5 better than Local, and better than the strongest realistic competitor of
that data set (the realistic literature method with the best mean on that data set)? One-sided exact sign test over
the 16 data sets (ties count against GHB), Holm over the 4 tests. Descriptive, because 10 of the 16 data sets were used
to develop v5.

## Family 4 (descriptive): Simulation 15, unit = simulated scenario

For each scenario and epsilon saved in sim15_literature.json: the difference between GHB and each method, with a z
statistic from the saved Monte Carlo standard errors combined as if independent (conservative when the replicates are
paired). Reported as counts of scenarios where GHB is significantly better, not distinguishable, or significantly
worse (two-sided, Holm within each comparison).

## What will be claimed

A superiority claim in the paper requires the Holm-adjusted p below 0.05 in Family 1 or 2. Otherwise the paper reports
the estimate and interval and says the difference is not statistically established. Families 3 and 4 are described as
supporting evidence, never as confirmation.
