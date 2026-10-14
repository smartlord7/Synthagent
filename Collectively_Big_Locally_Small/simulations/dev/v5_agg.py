"""Mean relative held-out log-loss of every variant in a v5 development checkpoint dir (default v5_dev_ckpt), against
local estimation, over the 12 TCGA development networks; with the best set-tuned literature method for reference."""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb  # noqa: E402
from real_v3 import GRIDS, ckey  # noqa: E402

d = ghb.RESDIR / (sys.argv[1] if len(sys.argv) > 1 else "v5_dev_ckpt")
acc = defaultdict(lambda: [0.0, 0.0])
for f in sorted(glob.glob(str(d / "*.json"))):
    for k, (ll, c) in json.load(open(f)).items():
        acc[k][0] += sum(ll); acc[k][1] += sum(c)
    for k, (ll, c) in json.load(open(ghb.RESDIR / "real_v3_ckpt" / Path(f).name))["out"].items():
        if not k.endswith("||Local"):
            acc[k][0] += sum(ll); acc[k][1] += sum(c)
nets = sorted({k.split("||")[0] for k in acc})
L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
var = sorted({k.split("||")[1].split(", eps")[0] for k in acc if "||v" in k and ", eps" in k})
for e in ("1", "10"):
    print(f"eps={e}")
    for m in var:
        r = [L(p, f"{m}, eps={e}") for p in nets]
        print(f"  {m:28s} mean {np.mean(r):.4f}  worst net {max(r):.3f}")
    best = min((np.mean([L(p, f"{ckey(c)}, eps={e}") for p in nets]), fam) for fam, g in GRIDS.items() for c in g)
    print(f"  best set-tuned literature    mean {best[0]:.4f} ({best[1]})")
