"""Run every simulation (1-17, with 8 and 10 in one script), the summary figure and the macros; figures go to supplement/figures, numbers to supplement/results."""
import runpy
import time
from pathlib import Path

HERE = Path(__file__).parent
for name in ["exp1_validate", "exp2_heterogeneity", "exp3_equity", "exp4_gossip", "exp5_joint_release",
             "exp6_gmrf", "exp7_logistic", "exp8_baselines", "exp9_atypical", "exp11_anchor", "exp12_byzantine",
             "exp13_dropout", "exp14_accounting", "exp15_literature", "exp16_ablation", "exp17_sensitivity", "fig_forest", "make_numbers"]:
    t = time.time()
    print(f"== {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
    print(f"   {time.time() - t:.0f}s")
