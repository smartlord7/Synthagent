# Pre-registration: GHB on the FLamby benchmark

Written 2026-10-13, before any model is fitted to the FLamby data. GHB, GHB-lite and every literature method keep the
configurations frozen in `PREREG_v4.md` and `v4_realistic_configs.json`; nothing below is tuned on FLamby outcomes
except the oracle arm, which is reported as an upper bound.

## Datasets and protocol (FLamby, Ogier du Terrail et al. 2022)

All three use FLamby's own centers and its static train/test split; every hospital is evaluated on its own test set,
and results are averaged over hospitals, as FLamby reports them. Each method is run 20 times (privacy noise and
method randomness), at epsilon in {1, 10, infinity}, delta = 1e-5, zero-concentrated accounting.

1. **Fed-Heart-Disease** (4 hospitals). FLamby's preprocessing (13 features, dummy coding, stratified split with
   seed 43, normalization with each hospital's own training statistics). Features are divided by 3 and clipped to
   [-1, 1] so that ||x|| <= sqrt(14) with the intercept (public constants). Model: logistic regression.
   FLamby metric: accuracy at 0.5. Secondary: held-out log-loss and AUC.
2. **Fed-TCGA-BRCA** (6 regions). FLamby's 39 features and split. Model: exponential (piecewise-exponential with one
   interval) survival regression, a Poisson GLM whose likelihood is a sum over patients, so every per-patient score
   at the origin, x (d - t), is bounded by ||x|| when follow-up time t is scaled by a public constant (10 years)
   and clipped to [0, 1]. Features: continuous ones standardized with each hospital's training statistics, divided
   by 3 and clipped; binary ones kept. Literature methods use per-patient gradient clipping at the same bound
   (DP-SGD style). FLamby metric: concordance index of the linear predictor. Secondary: held-out Poisson deviance.
   Note: these patients overlap with the BRCA network of the TCGA confirmation (there grouped by institution, here
   by region), so this dataset is not unseen in the sense of PREREG_v4.
3. **Fed-ISIC2019** (6 hospitals, 8 classes). Images embedded once by an ImageNet-pretrained EfficientNet-B0, the
   encoder of FLamby's baseline, kept frozen; embeddings projected to 16 dimensions by a fixed random Gaussian
   projection with a public seed (no data used), standardized with each hospital's training statistics, divided by
   3 and clipped. Model: eight one-versus-rest logistic heads (P = 17 each); the budget is split equally across the
   eight heads for every method. FLamby metric: balanced accuracy of the arg-max over heads. Secondary: log-loss.
   Requires pretrained weights; run only if they can be downloaded.

## Methods

Local (default prior); local with evidence-weighted prior; pooled without privacy (reference); GHB (frozen,
`f=1, R1=1, safety, no pan-network`: each dataset is one network); GHB-lite with the evidence-weighted prior;
DP-FedAvg + FT, DP-FedAvg-SA + FT, APFL, Ditto, pFedMe, clustered DP-FedAvg, each in two arms: realistic
(configuration frozen in `v4_realistic_configs.json`; at epsilon = infinity the epsilon = 10 configuration) and
oracle (best configuration of its grid on the test sets, an upper bound no hospital can reach).

## Predictions (written before running)

1. Heart, epsilon = 1: no private method improves local log-loss by more than 3% on average, and GHB stays within 3%
   of local (with 4 hospitals the joint releases carry little; Section 7.2.1 of the paper).
2. Heart, epsilon = 10 and infinity: GHB has a lower mean log-loss than the realistic arm of every literature method.
3. TCGA-BRCA, epsilon = 10 and infinity: GHB's mean concordance index is at least that of local estimation and of
   the realistic arm of every literature method.
4. ISIC2019, epsilon = 10 and infinity: GHB's mean balanced accuracy is at least that of the realistic arm of every
   literature method.
5. No prediction against the oracle arm, and none for TCGA-BRCA and ISIC2019 at epsilon = 1.

Outcomes are appended below as they come, whatever they are.

## Outcome

Run 2026-10-13, 20 repetitions each (`flamby_bench.py`, `supplement/results/flamby_results.json`). ISIC2019 was
run later the same day (prediction 4 below). Mean over hospitals; Heart: accuracy / log-loss; BRCA: C-index / Poisson deviance;
literature methods in the realistic arm.

| | GHB | best realistic literature | Local (default) | Local (evidence prior) |
|---|---|---|---|---|
| Heart, eps = 1 | .793 / .444 | Ditto .790 / .430 | .788 / .471 | .788 / .447 |
| Heart, eps = 10 | .793 / .440 | DP-FedAvg-SA + FT .828 / .419 | | |
| Heart, eps = inf | .807 / .432 | DP-FedAvg-SA + FT .823 / .418 | | |
| BRCA, eps = 10 | .799 / .456 | pFedMe .831 / .420 | .738 / .648 | .733 / .479 |
| BRCA, eps = inf | .814 / .439 | pFedMe .829 / .420 | | |

1. Not held. At eps = 1 private methods did improve on local log-loss by more than 3% (Ditto 8.7%, GHB 5.5%);
   GHB was within 0.5% of the local model with the evidence-weighted prior.
2. Not held. At eps = 10 and infinity, DP-FedAvg-SA + FT and Ditto (realistic) had lower log-loss than GHB; GHB beat
   the other four families. GHB-lite was more accurate than GHB on this dataset (.416 and .403).
3. Not held. On BRCA, GHB beat local estimation but not pFedMe (realistic) at either budget, nor DP-FedAvg-SA + FT
   at eps = 10.
4. Held (ISIC2019, run 2026-10-13 after the first report, with FLamby's own encoder, efficientnet_pytorch
   EfficientNet-B0 with its ImageNet weights downloaded from the package's GitHub release, checksum-verified; 23,247
   images; 20 repetitions). Balanced accuracy, mean over hospitals: GHB .301 / .300 / .301 at eps = 1 / 10 / infinity,
   against Ditto (realistic) .296, pFedMe .286 and the four federated-averaging families .20-.21; local .300, local with
   the evidence-weighted prior .303, pooled without privacy .250; best oracle .305. GHB's run-to-run SD is about .001.
   The prediction held, but every method that uses the hospital's own data sits at the local level: the frozen
   encoder with a 16-dimensional random projection leaves little to borrow, so this result shows that GHB loses
   nothing on images, not that it gains.

Deviations from the protocol: GHB-lite was not run on BRCA (its public-moment inversion needs a public model of
follow-up and censoring); the BRCA feature bound is G = 4 (one-hot coding of 14 variables, a public schema) rather
than sqrt(40); the Poisson GHB starts each MAP fit from the better of the prior mean and the origin and gives zero
weight to components with non-finite Laplace evidence; literature methods on BRCA use per-patient gradient clipping
at G throughout; oracle chosen per metric.
