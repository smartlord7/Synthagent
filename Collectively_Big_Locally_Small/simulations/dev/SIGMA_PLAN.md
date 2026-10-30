# Does the smallness index predict who gains from a remedy? Test plan

Written on 2026-10-30 and committed before any correlation is computed. Code: `simulations/sigma_validation.py`;
output: `supplement/results/sigma_validation.json` and `supplement/tables/sigma_*.tex`. Every outcome is reported,
whichever way it goes.

## Question

The survey defines the smallness index sigma = n* / n_eff (Section 2). A supervisor asked whether it is better than the
existing notions. Here "better" is tested in one specific, checkable sense: does sigma predict how much a hospital
gains from borrowing (GHB) better than its raw size n, its number of events, or its events per variable (EPV)?

## Units and outcome

- Primary set: every hospital of the five held-out TCGA networks (`v5_confirm_ckpt`, GHB as released), the twelve TCGA
  design networks (`v5f_dev_ckpt`, the frozen released configuration), Arizona Medicare and UCI Heart Disease (rerun
  of `real_networks.py` folds with Local and GHB as released only, per hospital, `sigma_networks_ckpt`).
- Secondary set: the six small TCGA networks of `real_networks.py` (same cancer types and largely the same hospitals as
  six design networks, with a smaller model), reported separately so that no hospital is counted twice in the primary
  analysis.
- Outcome for hospital k at budget eps in {1, 10}: gain_k = 1 - (summed held-out log-loss of GHB) / (summed held-out
  log-loss of Local), over all repetitions and folds.

## Predictors

- n_k: number of patients of hospital k; events_k = n_k * min(phi_k, 1 - phi_k), phi_k its outcome prevalence;
  EPV_k = events_k / p, with p the number of predictor parameters of the model (intercept excluded).
- n_eff,k = n_k. These data have one record per patient and complete, adjudicated labels, and the target population is
  the hospital's own, so DEFF = rho_L = rho_S = 1 and eta = 0 in Equation (neff). This is a limitation of the test: the
  quality factors of n_eff do not vary here, so the test can show only whether the task-relativity of sigma (its
  dependence on p and on prevalence through n*) adds predictive value, not whether the quality factors do.
- n*_k: the minimum sample size of Riley et al. for a binary outcome, the maximum of their three criteria (shrinkage
  S = 0.9; optimism in Nagelkerke R^2 at most 0.05; intercept within 0.05 at 95%), with p as above, phi_k clipped to
  [0.5/n_k, 1 - 0.5/n_k], and anticipated R^2_CS = 0.15 times its maximum given phi_k (as in Example 2.4).
- sigma_k = n*_k / n_eff,k.

## Statistics

- Spearman correlation of gain_k with each predictor over the hospitals of the set. Expected signs: positive for sigma,
  negative for n, events and EPV. Predictors are compared by the absolute value of the correlation.
- Primary statistic, at each budget: D = |rho(sigma)| - max(|rho(n)|, |rho(events)|, |rho(EPV)|), with a 95% interval
  from a cluster bootstrap over networks (2,000 resamples, seed 2026).
- Claim rule: sigma is called a better predictor only if the lower end of the interval of D is above 0 at both budgets.
  If not, the survey says so and reframes sigma as a task-relative, decomposable, multi-site diagnostic rather than a
  better predictor.
- Sensitivity: the mean of the within-network Spearman correlations (networks with at least 5 hospitals), which removes
  differences between networks.

## SynthAgent (descriptive)

Six tables of the SynthAgent study (`Synthagent__An_Agentic_Framework_for_Synthetic_Data_Generation/generated/`):
sigma, n, minority events and EPV from `datasets.csv` (p = number of features, phi = minority share), against the F1
change of the reference configuration in `full_results.csv`. With n = 6 the Spearman correlations are reported without
tests or claims.
