"""Stress tests of GHB on the TCGA networks: when does its lead over the literature hold, and when does it not?

FLamby, where GHB is not the most accurate method, differs from the TCGA networks in three ways: fewer hospitals per
network, a larger head, and one network fitted alone (no borrowing across cancer types). Each condition below changes
one of them on the 17 TCGA networks (12 design, 5 held-out), with the folds of dev/v4_dev.py, and reruns every method:
local estimation (default and evidence prior), GHB as released, GHB-lite with the evidence-weighted prior, and the six
literature families at every configuration of their grids. The realistic arm is re-tuned under each condition: the
configuration with the lowest mean error over the 12 design networks under that condition, applied to the 5 held-out
networks. The oracle arm is the best configuration on each held-out network's own folds.

Conditions
  standard   as in the paper (pipeline check: must reproduce the paper's numbers)
  alone      every network fitted on its own (no borrowing across cancer types); the literature is unchanged
  K6, K4     each network reduced to 6 or 4 hospitals, drawn at random (seeded) per repetition; smaller ones kept whole
  p14, p40   6 or 32 pure-noise features appended (N(0,1)/3 clipped to [-1,1], as standardized features), so the head
             has 14 or 40 coefficients and the feature bound is sqrt(p)
  flamby     K4 + p14 + alone (the shape of FLamby's Fed-Heart-Disease)

    python3 stress_tcga.py run [conditions...]    python3 stress_tcga.py score    (results/stress_tcga.json)
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

import ghb
import ghb_sa as SA
import ghb_v3 as V3
import ghb_v4 as V4
import literature as LIT
import personalized as PZ
import real_v3 as RV
from exp7_logistic import fit
from real_heart import folds

CONDITIONS = {"standard": {}, "alone": dict(alone=True), "K6": dict(K=6), "K4": dict(K=4), "p14": dict(p=14),
              "p40": dict(p=40), "flamby": dict(K=4, p=14, alone=True)}
CKPT = ghb.RESDIR / "stress_tcga_ckpt"
OUT = ghb.RESDIR / "stress_tcga.json"
SID = {"dev": 0, "fresh": 1, "confirm": 2}
REPS = 10
BASE_P = 8  # the eight clinical coefficients of real_v3 (RV.P is changed per condition for real_v3.baselines)


def run(item):
    cond, which, rep = item
    c = CONDITIONS[cond]
    sid = SID[which]
    rng = np.random.default_rng([30, rep, sid])  # the folds of dev/v4_dev.py and real_v3.py
    nets = {p: RV.load(p) for p in RV.SETS[which]}
    data = {p: v[0] for p, v in nets.items()}
    inst = {p: v[1] for p, v in nets.items()}
    for p in data:
        RV.public(rng)
    fs = {p: [folds(rng, y, RV.FOLDS) for _, y in data[p]] for p in data}
    srng = np.random.default_rng([40, rep, sid])  # condition-specific draws: hospital subsets and noise features
    P = c.get("p", BASE_P)
    for i, p in enumerate(sorted(data)):
        keep = np.arange(len(data[p]))
        if "K" in c and len(keep) > c["K"]:
            keep = np.sort(srng.choice(len(keep), size=c["K"], replace=False))
        data[p] = [data[p][k] for k in keep]
        fs[p] = [fs[p][k] for k in keep]
        inst[p] = [inst[p][k] for k in keep]
        if P > BASE_P:
            data[p] = [(np.hstack([X, np.clip(srng.standard_normal((len(y), P - BASE_P)) / 3, -1, 1)]), y) for X, y in data[p]]
    G = np.sqrt(float(P))
    LIT.configure(P, G); PZ.configure(P, G); SA.G = G; SA.P = P
    RV.P = P  # real_v3.baselines sizes its parameter vectors with the module constant
    prior = np.full(P, 1 / 2.5 ** 2)
    pub0 = np.random.default_rng([42, rep, sid])
    pubs = {}
    for p in data:
        Xp = RV.public(pub0)
        if P > BASE_P:
            Xp = np.hstack([Xp, np.clip(pub0.standard_normal((len(Xp), P - BASE_P)) / 3, -1, 1)])
        pubs[p] = V3.Public(Xp)
    groups = [[p] for p in data] if c.get("alone") else [list(data)]
    acc = {}
    for f in range(RV.FOLDS):
        train = {p: [(X[fs[p][k] != f], y[fs[p][k] != f]) for k, (X, y) in enumerate(data[p])] for p in data}
        test = {p: [(X[fs[p][k] == f], y[fs[p][k] == f]) for k, (X, y) in enumerate(data[p])] for p in data}
        est = {p: {"Local": np.array([fit(X, y, np.zeros(P), prior) for X, y in train[p]]),
                   "Local, evidence prior": np.array([V3.combine(X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in V3.SDS]) for X, y in train[p]])}
               for p in data}
        for e in RV.EPS:
            nrng = np.random.default_rng([31, rep, sid, f, int(e)])  # the privacy noise stream of dev/v4_dev.py
            for grp in groups:
                res = V4.estimate({p: train[p] for p in grp}, nrng, e, RV.DELTA, G, inst={p: inst[p] for p in grp}, cfg=RV.GHB_V5)
                lite = V3.estimate({p: train[p] for p in grp}, {p: pubs[p] for p in grp}, nrng, e, RV.DELTA, G,
                                   inst={p: inst[p] for p in grp}, cfg=RV.VARIANTS["GHB-EB"])
                for p in grp:
                    est[p][f"GHB, eps={e:g}"] = res[p]
                    est[p][f"GHB-lite, eps={e:g}"] = lite[p]
            brng = np.random.default_rng([32, rep, sid, f, int(e)])
            for p in data:
                for k, v in RV.baselines(brng, train[p], e).items():
                    est[p][f"{k}, eps={e:g}"] = v
        for p in data:
            for m, th in est[p].items():
                ll = np.array([np.sum(np.logaddexp(0, Xt @ th[k]) - yt * (Xt @ th[k])) for k, (Xt, yt) in enumerate(test[p])])
                cnt = np.array([len(yt) for _, yt in test[p]], float)
                a = acc.setdefault(f"{p}||{m}", [np.zeros(len(ll)), np.zeros(len(ll))])
                a[0] += ll; a[1] += cnt
    json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in acc.items()}, open(CKPT / f"{cond}_{which}_{rep}.json", "w"))
    return item


def score():
    from real_v3 import GRIDS, ckey
    out = {}
    for cond in CONDITIONS:
        files = sorted(CKPT.glob(f"{cond}_*.json"))
        if len(files) < 3 * REPS:
            continue
        acc = {}
        for fn in files:
            for k, (ll, cnt) in json.load(open(fn)).items():
                a = acc.setdefault(k, [0.0, 0.0])
                a[0] += sum(ll); a[1] += sum(cnt)
        rel = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
        design, held = RV.SETS["dev"] + RV.SETS["fresh"], RV.SETS["confirm"]
        r = {}
        for e in ("1", "10"):
            row = {m: float(np.mean([rel(p, f"{m}, eps={e}") for p in held])) for m in ("GHB", "GHB-lite")}
            row["Local, evidence prior"] = float(np.mean([rel(p, "Local, evidence prior") for p in held]))
            fam_real, fam_orc, cfgs = {}, {}, {}
            for fam, grid in GRIDS.items():
                keys = [f"{ckey(g)}, eps={e}" for g in grid]
                best = min(keys, key=lambda k: np.mean([rel(p, k) for p in design]))
                cfgs[fam] = best
                fam_real[fam] = float(np.mean([rel(p, best) for p in held]))
                fam_orc[fam] = float(np.mean([min(rel(p, k) for k in keys) for p in held]))
            orc_net = float(np.mean([min(min(rel(p, f"{ckey(g)}, eps={e}") for g in grid) for grid in GRIDS.values()) for p in held]))
            g = np.array([rel(p, f"GHB, eps={e}") for p in held])
            br = min(fam_real, key=fam_real.get)
            bo = min(fam_orc, key=fam_orc.get)
            row.update({"realistic": fam_real, "oracle": fam_orc, "configs": cfgs, "best_realistic": br,
                        "best_oracle": bo, "oracle_per_network": orc_net,
                        "ghb_wins_vs_best_realistic": int((g < np.array([rel(p, cfgs[br]) for p in held])).sum()),
                        "design_GHB": float(np.mean([rel(p, f"GHB, eps={e}") for p in design])),
                        "design_best_realistic": float(min(np.mean([rel(p, cfgs[f]) for p in design]) for f in GRIDS))})
            r[e] = row
        out[cond] = r
        print(f"\n{cond}  ({len(files) // 3} repetitions; held-out mean relative log-loss)")
        for e in ("1", "10"):
            x = r[e]
            print(f"  eps={e}: GHB {x['GHB']:.3f}  GHB-lite {x['GHB-lite']:.3f}  local EB {x['Local, evidence prior']:.3f}  "
                  f"best realistic {x['realistic'][x['best_realistic']]:.3f} ({x['best_realistic']})  "
                  f"best oracle {x['oracle'][x['best_oracle']]:.3f} ({x['best_oracle']})  GHB wins {x['ghb_wins_vs_best_realistic']}/5")
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
        sys.exit()
    conds = sys.argv[2:] or list(CONDITIONS)
    CKPT.mkdir(exist_ok=True)
    todo = [(c, w, r) for c in conds for r in range(REPS) for w in ("confirm", "dev", "fresh")
            if not (CKPT / f"{c}_{w}_{r}.json").exists()]
    with Pool(4) as pool:
        for it in pool.imap_unordered(run, todo):
            print("done", it, flush=True)
