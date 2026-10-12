# Development of the revised GHB (prototype code)

These scripts are the prototypes used to revise GHB after Simulation 15 (first run). They use development
seeds (900 to 904), which no reported simulation uses, and they are kept as written, so that every design
choice can be traced. The reported simulations use `../ghb_logit.py` and `../ghb.py`.

| Script | Question |
|---|---|
| `ghb2.py` | Releases of sufficient statistics X'y/n: are their stated variances calibrated? |
| `dev.py` | Plug-in tau^2 (posterior mean, precision, median) against tau integrated out; Student-t tail |
| `dev2.py` | Which release: score at the anchor or ridge fits (lambda 0.2 or 1, centred on the anchor or 0); anchor-free linearization and the conflict check; case-mix shift and a wrong anchor |
| `gauss_fb.py` | Tau integrated out in the Gaussian model |
| `ext.py` | Extension 1: joint release by groups of hospitals (secure aggregation) |
| `ext2.py` | Extension 2: two release rounds under zCDP |
| `ext3.py` | Extension 3: a two-level prior for public region labels |

Run from this folder, e.g. `python3 dev2.py`. Outputs are printed, not saved; the numbers quoted in the
supplement's development note come from these runs.

## Real data: development and pre-registrations of GHB

| Script / file | Purpose |
|---|---|
| `prior_eb.py`, `prior_eb_heart.py`, `PREREG_prior.md` | Evidence-weighted default prior for GHB-lite (pre-registered; criterion not met) |
| `sa_dev.py`, `sa_dev2.py`, `PREREG_sa.md` | GHB-SA: secure aggregation of per-hospital quantities (pre-registered; criterion not met) |
| `PREREG_v3.md` | Four extensions of GHB-lite on TCGA (`../real_v3.py dev`, then `fresh` once) |
| `v4_dev.py`, `v4_agg.py` | Development of GHB (joint releases, `../ghb_v4.py`) on the 12 TCGA development networks, with the folds of `real_v3.py`; `python3 v4_dev.py 10 <checkpoint folder>` |
| `v4_realistic_configs.json` | Each literature method's configuration chosen on the development networks, frozen before the confirmation |
| `PREREG_v4.md`, `v4_confirm_agg.py` | Pre-registration of GHB and its confirmation (`../real_v3.py confirm`, run once) |

The ablation of GHB in the supplement is `python3 v4_dev.py 10 v4_ablation_ckpt` (variants in `v4_dev.py`).
