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
