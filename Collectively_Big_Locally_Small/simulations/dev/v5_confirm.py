"""Pre-registered confirmation of GHB v5 (dev/PREREG_v5.md) on the five TCGA confirmation networks.
    python3 dev/v5_confirm.py [reps]      then   python3 dev/v5_confirm.py score
"""
import glob
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v4_dev as D  # noqa: E402
import real_v3 as RV  # noqa: E402
from real_v3 import GRIDS, ckey  # noqa: E402

D.VARIANTS = {"GHB v5": RV.GHB_V5, "GHB v4 (paired rerun)": RV.GHB_V4}
D.OUT = D.ghb.RESDIR / "v5_confirm_ckpt"


def score():
    acc = defaultdict(lambda: [0.0, 0.0])
    files = sorted(glob.glob(str(D.OUT / "confirm_*.json")))
    for f in files:
        for k, (ll, c) in json.load(open(f)).items():
            acc[k][0] += sum(ll); acc[k][1] += sum(c)
        for k, (ll, c) in json.load(open(D.ghb.RESDIR / "real_v3_ckpt" / Path(f).name))["out"].items():
            if not k.endswith("||Local"):
                acc[k][0] += sum(ll); acc[k][1] += sum(c)
    nets = RV.SETS["confirm"]
    L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
    sel = json.load(open(Path(__file__).with_name("v4_realistic_configs.json")))
    out = {"reps": len(files), "nets": nets, "rel": {}}
    for e in ("1", "10"):
        r = {m: [L(p, f"{m}, eps={e}") for p in nets] for m in D.VARIANTS}
        for fam, grid in GRIDS.items():
            r[fam + " (realistic)"] = [L(p, f"{sel[f'{fam}, eps={e}']}, eps={e}") for p in nets]
            R = np.array([[L(p, f"{ckey(c)}, eps={e}") for p in nets] for c in grid])
            r[fam + " (oracle)"] = list(R.min(0))
        out["rel"][e] = r
        print(f"eps={e}  ({len(files)} repetitions)")
        for m, v in r.items():
            print(f"  {m:32s} " + " ".join(f"{x:.3f}" for x in v) + f"  mean {np.mean(v):.4f}")
    v5 = {e: np.mean(out["rel"][e]["GHB v5"]) for e in ("1", "10")}
    lit = lambda e, arm: {m: np.mean(v) for m, v in out["rel"][e].items() if m.endswith(f"({arm})")}
    p = {1: all(v5[e] < min(lit(e, "realistic").values()) for e in ("1", "10")),
         2: all(v5[e] < np.mean(out["rel"][e]["GHB v4 (paired rerun)"]) for e in ("1", "10")),
         3: sum(a < b for a, b in zip(out["rel"]["10"]["GHB v5"], out["rel"]["10"]["Ditto (realistic)"])) >= 4,
         4: v5["1"] < min(lit("1", "oracle").values()),
         5: v5["10"] < min(lit("10", "oracle").values())}
    out["predictions"] = {str(k): bool(v) for k, v in p.items()}
    print("predictions:", {k: ("held" if v else "NOT held") for k, v in p.items()})
    json.dump(out, open(D.ghb.RESDIR / "v5_confirm.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        D.OUT.mkdir(exist_ok=True)
        reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        todo = [("confirm", r) for r in range(reps) if not (D.OUT / f"confirm_{r}.json").exists()]
        with Pool(3) as pool:
            for it in pool.imap_unordered(D.run, todo):
                print("done", it, flush=True)
