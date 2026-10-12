"""Run every simulation (1-18, with 8 and 10 in one script), the summary figure and the macros; figures go to
supplement/figures, numbers to supplement/results.

The real-data runs are long and checkpointed, and are run separately (each resumes where it stopped):
    python3 real_networks.py          # eight networks: Medicare, six TCGA (5 coefficients), UCI Heart
    python3 real_v3.py dev            # TCGA, 8 coefficients: development networks (and `fresh`)
    python3 real_v3.py confirm        # the pre-registered confirmation of GHB (dev/PREREG_v4.md), run once
    python3 dev/v4_confirm_agg.py     # scores it; final_compare.py draws the eight-network comparison
make_numbers.py and make_tables.py read all of them."""
import runpy
import time
from pathlib import Path

HERE = Path(__file__).parent
for name in ["exp1_validate", "exp2_heterogeneity", "exp3_equity", "exp4_gossip", "exp5_joint_release",
             "exp6_gmrf", "exp7_logistic", "exp8_baselines", "exp9_atypical", "exp11_anchor", "exp12_byzantine",
             "exp13_dropout", "exp14_accounting", "exp15_literature", "exp16_ablation", "exp17_sensitivity", "exp18_fairness", "real_heart", "fig_forest", "make_numbers", "make_tables"]:
    t = time.time()
    print(f"== {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
    print(f"   {time.time() - t:.0f}s")
