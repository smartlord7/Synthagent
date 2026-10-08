"""Regenerate every data figure of the paper (Figures 2 and 5-9) into figures/."""
import runpy
from pathlib import Path

HERE = Path(__file__).parent
for name in ["fig_deflation", "fig_matrix", "fig_corpus", "fig_gapmap", "fig_risk", "fig_ess_gain",
             "fig_gossip"]:
    print(f"== {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
