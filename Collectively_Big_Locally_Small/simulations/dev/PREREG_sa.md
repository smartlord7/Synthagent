# Pre-registration: GHB with secure aggregation, version 2 (written before the confirmation run)

## Why
At realistic budgets, per-hospital privacy noise leaves little to share in the real networks (real_networks.py).
With secure aggregation the network sees only sums, and the noise is added once to each sum, at a scale set by one
patient rather than by the size of the smallest hospital. Version 1 (`ghb_sa.estimate`, dev/sa_dev.json) over-borrowed
at high heterogeneity because it clipped each hospital's deviation; version 2 fixes that.

## Rule (frozen)
`ghb_sa.estimate_v2(..., C=8, iv=False)`: two secure sums (score at the public point, budget 1/4; clipped per-patient
influences and their hospital-level squares, budget 3/4), each patient's influence clipped to norm 8, a posterior over
tau from the noisy heterogeneity statistic, and each hospital's own evidence weighting the tau components, two
high-heterogeneity components (tau^2 = 1, 4), a no-borrowing component and the default-prior scales {0.5, 1, 2.5}.
Chosen on development data by the rule: lowest geometric-mean error across the nine development settings among
configurations whose worst hospital stays at or below 1.10 (dev/sa_dev2.json: C = 8, no round 3: 0.389, worst 1.10;
C = 4: 0.375 but worst 1.15; C = 16: 0.422, worst 1.05).

## Development (done)
Simulations, development seeds 950-958, 20 replicates, no anchor (dev/sa_dev2.json, after fixing the GHB-EB reference):
geometric-mean error relative to local, worst hospital: GHB-EB 0.496 / 1.08; GHB-SA v2 0.389 / 1.10; GHB-EB without
privacy 0.283 / 1.11; DP-FedAvg-SA + fine-tuning tuned on the true loss 0.356 / 3.01, with one fixed configuration
0.496 / 7.27.
UCI Heart Disease (development network, real_networks.json): GHB-SA at epsilon 1, 3, 10: 0.999, 0.996, 0.998 of local,
worst hospital at most 1.02; nothing gains there.

## Realistic tuning arm (frozen; settings chosen on UCI Heart Disease, applied unchanged to the other seven)
eps = 1: DP-FedAvg R=30, step 0.5 · + fine-tuning R=30, 0.2, E=30 · decentralized R=30, 0.2, E=30 · clustered C=2, R=1,
2.0, E=30 · Ditto lambda=0.1 · DP-FedAvg-SA + FT R=30, 0.5, E=30 · local GD E=100 · local prior s.d. 1 · GHB prior s.d. 1.
eps = 10: DP-FedAvg R=30, 2.0 · + FT R=30, 2.0, E=30 · decentralized R=30, 2.0, E=30 · clustered C=2, R=10, 2.0, E=30 ·
Ditto lambda=1 · DP-FedAvg-SA + FT R=30, 2.0, E=30 · GHB prior s.d. 1.
The oracle arm (tuned on each network's held-out folds) is reported too, as an upper bound for the baselines.

## Confirmation (run once, reported whatever the outcome)
Networks: Arizona Medicare and the six TCGA networks, same protocol and seeds as before (GHB-SA and DP-FedAvg-SA on
their own random stream, so every earlier number is unchanged).
Primary: held-out log-loss relative to local, all hospitals, epsilon = 1:
1. GHB-SA vs GHB-EB and GHB as reported.
2. GHB-SA vs DP-FedAvg-SA + fine-tuning, realistic arm and oracle arm.
Secondary: epsilon = 10; hospitals below the median size; worst hospital with at least 20 patients.

## Decision
GHB-SA is recommended as GHB's mode when the network can run secure aggregation if, at epsilon = 1, it improves on
GHB-EB in at least five of the seven networks and its worst-hospital ratio exceeds GHB-EB's by at most 0.1 in every
network. The comparison with DP-FedAvg-SA is reported as it comes, in both arms; no claim of superiority is made unless
GHB-SA is better in at least five of seven networks against the realistic arm.

Caveat: these seven networks were seen before (with GHB, GHB-EB and the baselines); the rule above was fixed without
evaluating GHB-SA on them.

## Outcome (added after the confirmation run, supplement/results/real_networks.json)
Held-out log-loss relative to local, epsilon = 1 (GHB as reported / GHB-EB / GHB-SA | DP-FedAvg-SA + FT realistic /
oracle | worst hospital GHB-SA / GHB-EB / DP-FedAvg-SA realistic):
Arizona Medicare 0.998 / 0.966 / 0.965 | 0.967 / 0.950 | 1.06 / 1.05 / 1.03
Lung squamous 1.000 / 0.959 / 1.007 | 0.993 / 0.961 | 1.20 / 1.12 / 1.04
Head and neck 1.000 / 0.941 / 0.970 | 1.012 / 0.931 | 1.02 / 1.00 / 1.01
Bladder 1.002 / 0.933 / 0.955 | 1.040 / 0.922 | 1.18 / 1.10 / 1.09
Stomach 1.006 / 0.966 / 0.984 | 1.037 / 0.963 | 1.03 / 1.02 / 1.07
Liver 1.000 / 0.922 / 0.962 | 1.093 / 0.921 | 2.87 / 1.86 / 1.25
Kidney 0.996 / 1.005 / 1.033 | 1.015 / 0.993 | 1.07 / 1.02 / 1.09
Decision criterion (GHB-SA improves on GHB-EB in at least five of seven, worst hospital within +0.1): not met (one of
seven; liver +1.01). GHB-SA is therefore not recommended as GHB's mode; it is reported as a negative result.
Against DP-FedAvg-SA + FT: GHB-SA better in five of seven against the realistic arm, none against the oracle arm. GHB-EB,
with no tuning, is better than the realistic DP-FedAvg-SA + FT in seven of seven networks and than the realistic
DP-FedAvg + FT in seven of seven; the oracle arms of both are better than GHB-EB in six or seven of seven.
At epsilon = 10 GHB-SA is within 0.01 of GHB-EB in every network.
