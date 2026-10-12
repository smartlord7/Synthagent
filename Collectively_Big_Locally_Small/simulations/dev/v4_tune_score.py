"""Score the tuning of GHB (dev/v4_tune.py) against the literature methods tuned the same ways (real_v3.py)."""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb
from real_v3 import GRIDS, ckey


def load(pattern, key=None):
    acc = defaultdict(lambda: [0.0, 0.0])
    for fn in glob.glob(str(pattern)):
        d = json.load(open(fn))
        for k, (ll, c) in (d["out"] if "out" in d else d).items():
            acc[k][0] += sum(ll); acc[k][1] += sum(c)
    return acc


def rel(acc, p, m):
    return (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])


T = ghb.RESDIR / "v4_tune_ckpt"
tdev = load(T / "dev_*.json"); tdev.update(load(T / "fresh_*.json"))
tcon = load(T / "confirm_*.json")
bdev = load(ghb.RESDIR / "real_v3_ckpt" / "dev_*.json"); bdev.update(load(ghb.RESDIR / "real_v3_ckpt" / "fresh_*.json"))
bcon = load(ghb.RESDIR / "real_v3_ckpt" / "confirm_*.json")
dn = sorted({k.split("||")[0] for k in tdev}); cn = sorted({k.split("||")[0] for k in tcon})
cfgs = sorted({k.split("||")[1].split(", eps")[0] for k in tdev if "||T " in k})
FROZEN = "T s=0.5|w=0.33|h=0.5"
out = {"nets": cn, "configs": len(cfgs)}
for e in ("1", "10"):
    M = {c: np.array([rel(tdev, p, f"{c}, eps={e}") for p in dn]) for c in cfgs}
    C = {c: np.array([rel(tcon, p, f"{c}, eps={e}") for p in cn]) for c in cfgs}
    best = min(cfgs, key=lambda c: M[c].mean())
    r = {"GHB, frozen": C[FROZEN].tolist(), "GHB, tuned on development": C[best].tolist(),
         "GHB, oracle": np.min([C[c] for c in cfgs], 0).tolist(), "chosen": best,
         "dev_frozen": float(M[FROZEN].mean()), "dev_tuned": float(M[best].mean())}
    sel = json.load(open(Path(__file__).with_name("v4_realistic_configs.json")))
    for fam, grid in GRIDS.items():
        r[fam + " (realistic)"] = [rel(bcon, p, f"{sel[f'{fam}, eps={e}']}, eps={e}") for p in cn]
        r[fam + " (oracle)"] = np.min([[rel(bcon, p, f"{ckey(c)}, eps={e}") for p in cn] for c in grid], 0).tolist()
    out[e] = r
    print(f"eps={e}: chosen on development {best} (dev mean {M[best].mean():.3f} vs frozen {M[FROZEN].mean():.3f})")
    for k, v in r.items():
        if isinstance(v, list):
            print(f"  {k:36s}" + " ".join(f"{x:.3f}" for x in v) + f"  mean {np.mean(v):.3f}")
json.dump(out, open(ghb.RESDIR / "real_v4_tuned.json", "w"), indent=1)
