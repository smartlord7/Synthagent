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
