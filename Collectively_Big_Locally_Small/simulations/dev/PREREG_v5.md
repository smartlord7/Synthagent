# GHB v5: pre-registered confirmation on the five TCGA confirmation networks

Written 2026-10-14, committed before GHB v5 is run on any of these networks.

## What v5 is (frozen: `real_v3.GHB_V5`)

GHB v4 (`real_v3.GHB_V4`) plus three changes, chosen during development (below):
1. **The intercept is not borrowed.** In every borrowing component the intercept gets an extra prior variance
   2.5^2 around the network value, and in the default prior its scale stays 2.5 whatever the scale s of the slopes
   (`int_sd=2.5, free_int=2.5`). Base rates differ across hospitals (case mix, referral); slopes are what is shared.
2. **Tighter sensitivity of the information release.** Per-patient information matrices are PSD with Frobenius norm
   at most m (G^2 / 4 for the logistic); for PSD A, B, ||A - B||_F^2 <= ||A||^2 + ||B||^2, so a replacement moves the
   sum by at most sqrt(2) m instead of 2 m (`tight=True`). Same privacy guarantee, about 30% less noise.
3. **Full or diagonal information release by a public rule** decided before the release: the diagonal alone when the
   expected spectral noise sh sqrt(P) exceeds N m / P (`hdiag="auto"`).
Evidence weighting of the components as in v4; leave-one-out weighting and a noise-aware network mean were tried
and rejected in development.

## Development (declared; none of it on the confirmation networks)

The 12 TCGA development networks (`dev/v5_dev.py`, `dev/v5_agg.py`), the seven small networks and UCI Heart Disease
(`dev/v5_networks.py`), FLamby Heart-Disease and TCGA-BRCA (`ghb_glm.py` options), and ISIC 2019. On the 12 development
networks v5 reached 0.915 (eps = 1) and 0.897 (eps = 10) of the local log-loss, against 0.934 and 0.913 for v4 and
0.946 and 0.923 for the best literature method tuned on them.

## Test set and protocol

The five confirmation networks of `PREREG_v4.md` (TCGA BRCA, MESO, OV, UCEC, UCS), the same ten repetitions of
stratified five-fold cross-validation (the folds of `real_v3.run_set("confirm", rep)`), eps in {1, 10}, delta 1e-5.
`dev/v5_confirm.py` runs GHB v5 and, with the same privacy noise, GHB v4 again (paired comparison). Literature rows are
those already stored (`results/real_v3_ckpt/confirm_*.json`, pFedMe with its corrected noise), realistic arm =
`dev/v4_realistic_configs.json`, oracle arm = best configuration per network. Outcome: held-out log-loss relative to
local estimation, mean over the five networks.

## Predictions

1. At eps = 1 and at eps = 10, GHB v5 has a lower mean relative log-loss than every literature method in the
   realistic arm.
2. At eps = 1 and at eps = 10, GHB v5 has a lower mean than GHB v4 (paired rerun).
3. At eps = 10, GHB v5 beats Ditto (realistic) in at least 4 of the 5 networks.
4. At eps = 1, GHB v5 has a lower mean than every literature method in the oracle arm.
5. At eps = 10, GHB v5 has a lower mean than every literature method in the oracle arm (the hardest; v4 failed it).

Each prediction is reported as held or not held, whatever the result.
