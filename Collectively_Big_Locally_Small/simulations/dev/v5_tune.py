"""GHB as released (real_v3.GHB_V5) tuned like the literature methods (exploratory): the grid of dev/v4_tune.py (36
configurations: half-Cauchy scale of tau, weight of the non-borrowing components, budget share of the information
release) around the released configuration, with the folds and privacy noise of dev/v4_dev.py, on the 12 design and
the 5 held-out networks. Arms as for the baselines: realistic (the configuration with the lowest mean over the design
networks, applied unchanged to the held-out networks) and oracle (the best configuration on each held-out network's own
held-out folds).
    python3 dev/v5_tune.py run      then   python3 dev/v5_tune.py score   (results/real_v5_tuned.json)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_dev as D  # noqa: E402
import real_v3 as RV  # noqa: E402

GRID = {f"T s={sc}|w={w:.2f}|h={h}": dict(RV.GHB_V5, scale=sc, safety_w=w, safety=w > 0, hshare=h)
        for sc in (0.1, 0.25, 0.5, 1.0) for w in (0.0, 1 / 3, 1.0) for h in (0.25, 0.5, 0.75)}
D.VARIANTS = GRID
D.OUT = D.ghb.RESDIR / "v5_tune_ckpt"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        import v4_tune_score as S  # noqa: E402  (scores with the v5 checkpoints; see the swap below)
        sys.exit()
    D.OUT.mkdir(exist_ok=True)
    todo = [(w, r) for r in range(10) for w in ("dev", "fresh", "confirm") if not (D.OUT / f"{w}_{r}.json").exists()]
    with Pool(4) as pool:
        for it in pool.imap_unordered(D.run, todo):
            print("done", it, flush=True)
