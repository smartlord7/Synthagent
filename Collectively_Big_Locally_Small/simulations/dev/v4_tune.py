"""GHB tuned like the literature methods (exploratory, not pre-registered): a grid of 36 configurations around the
frozen one, run with the folds of real_v3.py on the 12 development and the 5 confirmation networks. Arms as for the
baselines: realistic (the configuration with the lowest mean over the development networks, applied unchanged to the
confirmation networks) and oracle (the best configuration on each confirmation network's own held-out folds).
Usage: python3 dev/v4_tune.py [run|score]"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_dev as D

GRID = {f"s={sc}|w={w:.2f}|h={h}": dict(D.LOCKED, scale=sc, safety_w=w, safety=w > 0, hshare=h)
        for sc in (0.1, 0.25, 0.5, 1.0) for w in (0.0, 1 / 3, 1.0) for h in (0.25, 0.5, 0.75)}
D.VARIANTS = {f"T {k}": v for k, v in GRID.items()}
D.OUT = D.ghb.RESDIR / "v4_tune_ckpt"

if __name__ == "__main__":
    D.OUT.mkdir(exist_ok=True)
    todo = [(w, r) for r in range(10) for w in ("dev", "fresh", "confirm") if not (D.OUT / f"{w}_{r}.json").exists()]
    with Pool(4) as pool:
        for it in pool.imap_unordered(D.run, todo):
            print("done", it, flush=True)
