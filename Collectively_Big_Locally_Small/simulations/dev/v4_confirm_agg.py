"""Score the pre-registered confirmation of GHB v4 (dev/PREREG_v4.md) on the third TCGA set."""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb
from real_v3 import GRIDS, ckey

sel = json.load(open(Path(__file__).with_name("v4_realistic_configs.json")))
files = sorted(glob.glob(str(ghb.RESDIR / "real_v3_ckpt" / "confirm_*.json")))
acc = defaultdict(lambda: [0.0, 0.0])
for f in files:
    for k, (ll, c) in json.load(open(f))["out"].items():
        acc[k][0] += sum(ll); acc[k][1] += sum(c)
nets = sorted({k.split("||")[0] for k in acc})
L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
print(len(files), "repeats;", " ".join(p[5:] for p in nets))
res = {}
for e in ("1", "10"):
    rows = {m: [L(p, f"{m}, eps={e}") for p in nets] for m in ("GHB v4", "GHB-EB", "+ cross-cancer")}
    rows["Local, evidence prior"] = [L(p, "Local, evidence prior") for p in nets]
    for fam, grid in GRIDS.items():
        rows[fam + " (realistic)"] = [L(p, f"{sel[f'{fam}, eps={e}']}, eps={e}") for p in nets]
        rows[fam + " (oracle)"] = list(np.min([[L(p, f"{ckey(c)}, eps={e}") for p in nets] for c in grid], 0))
    g = rows["GHB v4"]
    print(f"\neps={e}{'':29s}" + " ".join(f"{p[5:]:>5s}" for p in nets) + "   mean  v4 wins")
    for k, r in rows.items():
        print(f"{k:34s}" + " ".join(f"{x:.3f}" for x in r) + f"  {np.mean(r):.3f}  {sum(a < b for a, b in zip(g, r))}/{len(nets)}")
    res[e] = rows
json.dump(dict(nets=nets, repeats=len(files), rel=res), open(ghb.RESDIR / "real_v4_confirm.json", "w"), indent=1)
