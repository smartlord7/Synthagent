"""Development of GHB v4 on the 12 TCGA networks already used (real_v3 dev and fresh sets), with the same folds as
real_v3.py (same random stream up to the folds), so the stored baselines of real_v3_ckpt are compared fold for fold."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghb
import ghb_v4 as V4
import ghb_v3 as V3
import real_v3 as RV
from exp7_logistic import fit
from real_heart import folds

LOCKED = dict(f=1.0, safety=True, R1=1, pan=True)  # GHB v4 as pre-registered (dev/PREREG_v4.md)
VARIANTS = {  # ablation: one change at a time from the locked configuration
    "v4 locked": LOCKED,
    "v4 - pan-network": dict(LOCKED, pan=False),
    "v4 - no-borrowing components": dict(LOCKED, safety=False),
    "v4 - privacy covariance": dict(LOCKED, vdp=False),
    "v4 - leave-one-out": dict(LOCKED, loo=False),
    "v4 two Newton rounds": dict(LOCKED, R1=2),
    "v4 tau from releases (f=.5)": dict(LOCKED, f=0.5),
    "v4 + institution effects (f=.5)": dict(LOCKED, f=0.5, cross=True),
}
OUT = ghb.RESDIR / (sys.argv[2] if len(sys.argv) > 2 else "v4_dev_ckpt")


def run(item):
    which, rep = item
    sid = {"dev": 0, "fresh": 1, "confirm": 2}[which]
    rng = np.random.default_rng([30, rep, sid])
    nets = {p: RV.load(p) for p in RV.SETS[which]}
    data = {p: v[0] for p, v in nets.items()}
    inst = {p: v[1] for p, v in nets.items()}
    for p in data:
        RV.public(rng)
    fs = {p: [folds(rng, y, RV.FOLDS) for _, y in data[p]] for p in data}
    rng = np.random.default_rng([31, rep, sid])
    acc = {}
    for f in range(RV.FOLDS):
        train = {p: [(X[fs[p][k] != f], y[fs[p][k] != f]) for k, (X, y) in enumerate(data[p])] for p in data}
        test = {p: [(X[fs[p][k] == f], y[fs[p][k] == f]) for k, (X, y) in enumerate(data[p])] for p in data}
        est = {p: {"Local": np.array([fit(X, y, np.zeros(RV.P), RV.PRIOR) for X, y in train[p]])} for p in data}
        for e in RV.EPS:
            for name, cfg in VARIANTS.items():
                seed = [31, rep, sid, f, int(e)]  # the same privacy noise for every variant
                res = V4.estimate(train, np.random.default_rng(seed), e, RV.DELTA, RV.G, inst=inst, cfg=cfg)
                for p in data:
                    est[p][f"{name}, eps={e:g}"] = res[p]
        for p in data:
            for m, th in est[p].items():
                ll = np.array([np.sum(np.logaddexp(0, Xt @ th[k]) - yt * (Xt @ th[k])) for k, (Xt, yt) in enumerate(test[p])])
                a = acc.setdefault(f"{p}||{m}", [np.zeros(len(ll)), np.zeros(len(ll))])
                a[0] += ll; a[1] += np.array([len(yt) for _, yt in test[p]], float)
    json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in acc.items()}, open(OUT / f"{which}_{rep}.json", "w"))
    return item


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    todo = [(w, r) for r in range(reps) for w in ("dev", "fresh") if not (OUT / f"{w}_{r}.json").exists()]
    with Pool(4) as pool:
        for it in pool.imap_unordered(run, todo):
            print("done", it, flush=True)
