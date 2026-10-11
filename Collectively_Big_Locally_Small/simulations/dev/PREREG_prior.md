# Pre-registration: evidence-weighted default prior (written before the confirmation run)

## Why
The real-data benchmark (`real_networks.py`, committed in 3131cd5) showed that GHB's fixed default prior
N(0, 2.5^2) is too loose for hospitals of a few dozen patients: GHB stayed at the local level (about 1.00) while
local and federated models whose regularization was tuned on the held-out folds reached 0.93-0.96. The confirmation
networks below were therefore seen before this rule was fixed, with tuned priors; the rule itself was not evaluated on
them.

## Rule (frozen)
Each hospital integrates the scale of the default prior out over {0.5, 1, 2.5} (`ghb_logit.PRIOR_SDS`), with equal
prior weights, weighting every (tau component, scale) pair by the Laplace evidence of its own data, exactly as tau is
integrated out. No held-out data, no tuning. Implemented in `ghb_logit.combine(..., prior_sd=PRIOR_SDS)`.
The same rule applied without any network component (prior mean 0) is "Local, evidence prior".

## Development (done, `dev/prior_eb.py`, `dev/prior_eb_heart.py`)
Simulations, development seeds 900-909, 40 paired replicates, ratio to GHB as reported:
tau2=0.01 0.94, 0.05 0.97, 0.2 1.01, 0.5 1.03; eps=0.3 0.94; no DP 1.02; wrong anchor 0.96; shift 0.95;
no anchor 0.96; no anchor with median hospital size 25: 0.61.
UCI Heart Disease (the development network), relative to local: never worse than GHB as reported by more than 0.1%,
better by 0-0.6%.
Development criteria, written down after these development runs (so they document, not test): at most 5% worse
than GHB as reported in any simulation setting, no loss above 1% on UCI Heart Disease. Both hold. The confirmation
criteria below are fixed before the confirmation run.

## Confirmation (to be run once, reported whatever the outcome)
Networks: Arizona Medicare and the six TCGA networks of `real_networks.py`, same protocol and seeds.
New methods: GHB with the evidence prior at epsilon 1, 3, 10 and without privacy; Local, evidence prior.
Primary comparisons, held-out log-loss relative to local estimation, all hospitals:
1. GHB, evidence prior vs GHB as reported, at epsilon = 1 and 10.
2. GHB, evidence prior (untuned) vs the best model tuned on the held-out folds (local or federated), same budget.
Secondary: hospitals below the median size; the worst hospital with at least 20 patients.

## Decision
The evidence prior is reported as an extension of GHB in the paper ("GHB-EB") whatever the outcome. It is recommended
as the default for networks of small hospitals if, at epsilon = 1, it improves on GHB as reported in at least five of
the seven networks and does not raise the worst-hospital ratio by more than 0.1 in any. The simulations keep GHB as
reported; Simulation 16 gains one row for the evidence prior on the development seeds' settings.
