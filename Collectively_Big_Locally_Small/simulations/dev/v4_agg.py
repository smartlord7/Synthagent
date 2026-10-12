"""Aggregate v4 dev results against the stored real_v3 results (same folds). Realistic literature arm: best mean over
the networks of the set (in-sample, favourable to the baselines)."""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb
from real_v3 import GRIDS, ckey

reps = sorted({Path(f).stem for f in glob.glob(str(ghb.RESDIR / (sys.argv[1] if len(sys.argv) > 1 else "v4_dev_ckpt") / "*.json"))})
acc = defaultdict(lambda: [0.0, 0.0])
for r in reps:
    for k, (ll, c) in json.load(open(ghb.RESDIR / (sys.argv[1] if len(sys.argv) > 1 else "v4_dev_ckpt") / f"{r}.json")).items():
        acc[k][0] += sum(ll); acc[k][1] += sum(c)
    for k, (ll, c) in json.load(open(ghb.RESDIR / "real_v3_ckpt" / f"{r}.json"))["out"].items():
        if not k.endswith("||Local"):
            acc[k][0] += sum(ll); acc[k][1] += sum(c)
nets = sorted({k.split("||")[0] for k in acc})
L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
v4 = sorted({k.split("||")[1].split(", eps")[0] for k in acc if "||v4" in k})
print(len(reps), "files;", " ".join(p[5:9] for p in nets))
for e in ("1", "10"):
    rows = {m: [L(p, f"{m}, eps={e}") for p in nets] for m in v4 + ["GHB-EB", "+ cross-cancer"]}
    rows["Local, evidence prior"] = [L(p, "Local, evidence prior") for p in nets]
    for fam, grid in GRIDS.items():
        R = np.array([[L(p, f"{ckey(c)}, eps={e}") for p in nets] for c in grid])
        rows[fam + " (set-tuned)"] = list(R[R.mean(1).argmin()])
        rows[fam + " (oracle)"] = list(R.min(0))
    print(f"\neps={e}")
    for k, r in rows.items():
        print(f"{k:34s}" + " ".join(f"{x:.3f}" for x in r) + f"  mean {np.mean(r):.3f}")
