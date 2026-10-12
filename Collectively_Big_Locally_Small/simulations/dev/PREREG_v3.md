# Pre-registration: GHB v3 on fresh TCGA networks

Written 2026-10-12, after the development run (`real_v3.py dev`, 10 repeats of five-fold CV on LUSC, HNSC, BLCA,
STAD, LIHC, KIRC) and before any run on the fresh networks (LUAD, COAD, READ, KIRP, PAAD, ESCA), which have
never been loaded by any script.

## Development results (mean relative log-loss vs local with default prior, 6 networks)

| eps | GHB-EB | +safety | +learned scale | +correlated | +cross-cancer | v3 (all) | best literature, dev-tuned | best literature, oracle |
|---|---|---|---|---|---|---|---|---|
| 1  | 0.912 | 0.912 | 0.912 | 0.912 | 0.913 | 0.912 | 0.915 (FTSA) | 0.900 (FT) |
| 10 | 0.910 | 0.910 | 0.927 | 0.912 | 0.907 | 0.932 | 0.892 (FTSA) | 0.885 (FTSA) |

The learned prior scale hurts at eps=10 and is dropped. No addition changes anything at eps=1. Cross-cancer
borrowing is the only addition with a gain (eps=10).

## Locked choices

- GHB configuration (no tuning on fresh data): **GHB-EB + cross-cancer** (`safety=False, learned_scale=False,
  fullcov=False, cross=True`). GHB-EB is reported alongside as the reference.
- Literature, realistic arm: for each family, the configuration with the lowest mean relative log-loss over the
  six dev networks at the same eps, applied unchanged. Oracle arm: best configuration per fresh network.
- Metric: per network, pooled held-out log-loss over 10 repeats, relative to local with the default prior;
  summarised by the mean over the 6 fresh networks and by per-network wins.

## Predictions (stated before the run)

1. eps=1: GHB (locked) has a lower mean than every literature family in the realistic arm.
2. eps=10: GHB (locked) does NOT beat dev-tuned DP-FedAvg-SA + FT or pFedMe (dev gap ~0.015).
3. Against the oracle arm GHB loses at both budgets.
4. Cross-cancer improves on GHB-EB at eps=10 (mean).

Whatever the outcome, it is reported as is and appended below.

## Outcome

(to be appended after `python3 real_v3.py fresh`)
