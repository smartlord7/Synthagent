# Pre-registration: GHB v4 on a third, unseen set of TCGA networks

Written 2026-10-12, after development on the 12 TCGA networks of `real_v3.py` (dev and fresh sets; the fresh set
became development data once its confirmation of GHB v3 had been run and reported in `PREREG_v3.md`), and before any
model is fitted on the confirmation set.

## Diagnosis that motivated v4

GHB-EB and v3 invert each hospital's noisy score through a synthetic public population. On real case mixes this
biases the network mean, and with weak privacy (eps = 10) borrowing hurts. With strong privacy the per-hospital
releases are so noisy that GHB falls back to a local model. GHB v4 (`ghb_v4.py`) gives GHB two kinds of context
instead:
1. The network mean and pooled information come from the hospitals' real data through secure sums: one release of the
   pooled information matrix and one of the pooled score at 0, each with half of the budget. This is the trust model
   of GHB-SA and DP-FedAvg-SA. The privacy-noise covariance is propagated.
2. Pan-network empirical Bayes: each network's mean is shrunk toward the other networks' mean, with the
   between-network variance from its half-Cauchy posterior.
Each hospital's prior is centred at the leave-one-out network mean. Its uncertainty is the sampling covariance plus
the privacy covariance plus tau^2, with tau taken at the half-Cauchy(0.5) quantiles. This is combined with the default
scales {0.5, 1, 2.5}, plus the no-borrowing components, and everything is weighted by the hospital's own evidence. No
hyperparameter is tuned on outcome data. The variant was chosen among about 15 on the development networks.

## Development results (12 networks, 10 repeats; mean relative log-loss vs local, default prior)

| eps | GHB v4 (locked) | GHB-EB | local, evidence prior | best literature, tuned on these 12 networks (in-sample) |
|---|---|---|---|---|
| 1  | 0.926 | 0.927 | 0.927 | 0.934 (DP-FedAvg + FT) |
| 10 | 0.904 | 0.941 | 0.927 | 0.913 (pFedMe), 0.914 (DP-FedAvg-SA + FT), 0.916 (Ditto) |

## Locked choices

- GHB v4: `GHB_V4 = dict(f=1.0, safety=True, R1=1, pan=True)` in `real_v3.py`. Its code is frozen at this commit.
- Confirmation set: TCGA-BRCA, MESO, UCEC, OV, UCS (`data/tcga_v4`). These are every further TCGA project with at
  least 4 institutions of at least 5 eligible patients and at least 15 deaths. The set was fixed from these counts
  alone, before any model was fitted. FIGO stage is used where AJCC is absent.
- Literature, realistic arm: each family's configuration with the lowest mean over the 12 development networks
  (`dev/v4_realistic_configs.json`), applied unchanged. Oracle arm: the best configuration per network.
- Protocol: as `real_v3.py` (10 repeats of five-fold CV within institutions, eps in {1, 10}, delta = 1e-5); the
  networks of the set are fitted jointly; seed `[30, rep, 2]`.

## Predictions

1. At both eps, GHB v4's mean is lower than the realistic arm of every literature family.
2. At eps = 10, GHB v4's mean is lower than GHB-EB's.
3. At eps = 10, GHB v4 beats the best realistic literature method in at least 3 of the 5 networks.
4. No prediction against the oracle arm. In development it was ahead of GHB at eps = 10 (about 0.88-0.89).

The outcome is reported as is and appended below, whatever it is.

## Outcome

Run once, 10 repeats, 2026-10-12 (`supplement/results/real_v4_confirm.json`, `dev/v4_confirm_agg.py`). Mean
relative log-loss over BRCA, MESO, OV, UCEC, UCS:

| eps | GHB v4 | GHB-EB | local, evidence prior | best literature, realistic | best literature, oracle |
|---|---|---|---|---|---|
| 1  | **0.928** | 0.932 | 0.933 | 0.976 (DP-FedAvg + FT) | 0.939 (DP-FedAvg + FT) |
| 10 | **0.915** | 0.965 | 0.933 | 0.919 (Ditto), 0.927 (pFedMe) | 0.898 (pFedMe), 0.911 (Ditto), 0.912 (FTSA) |

1. Held: at both eps GHB v4 has a lower mean than every literature family's realistic arm (eps = 10: 0.915 vs Ditto
   0.919, narrow; every other family 0.927-0.990).
2. Held: at eps = 10 GHB v4 improves on GHB-EB (0.915 vs 0.965).
3. Held: at eps = 10 GHB v4 beats realistic Ditto in 3 of 5 networks (it beats every other realistic family in 4 or
   5 of 5).
4. (No prediction.) Oracle arm: at eps = 1 GHB v4 beats every family even with per-network hindsight tuning; at
   eps = 10 it ties oracle DP-FedAvg + FT (0.915), is close behind oracle Ditto and FTSA (0.911, 0.912), and is behind
   oracle pFedMe (0.898).
GHB v4 also beats the local model with an evidence prior at both budgets (0.928 vs 0.933; 0.915 vs 0.933).
