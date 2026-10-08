"""Run every supplementary experiment (S1-S7); figures go to supplement/figures, numbers to supplement/results."""
import runpy
import time
from pathlib import Path

HERE = Path(__file__).parent
for name in ["exp1_validate", "exp2_heterogeneity", "exp3_equity", "exp4_gossip", "exp5_joint_release",
             "exp6_gmrf", "exp7_logistic"]:
    t = time.time()
    print(f"== {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
    print(f"   {time.time() - t:.0f}s")
