"""Recompute only the pFedMe rows of existing checkpoints after the privacy fix of commit 15e769a.

The proximal fit of pFedMe minimizes the SUM of the losses, so the noise standard deviation was n times too small
(personalized.pfedme, ghb_glm.pfedme). Every other method consumes exactly the same random draws as before, so only the
pFedMe rows of the checkpoints are wrong. Rerunning everything takes many hours; this script recomputes ONLY the
pFedMe rows, with the corrected noise, on their own random stream (the run's seed list + [777]), on exactly the data
splits of the original runs, and overwrites those rows in place (each directory is first copied to
<dir>_pre_pfedme_fix/).

    python3 dev/pfedme_fix.py validate [families]  checks that the splits and the code reproduce the checkpoints
    python3 dev/pfedme_fix.py patch [families]  families among v3, rn, flamby (all by default)

Families:
  v3      supplement/results/real_v3_ckpt/{dev,fresh,confirm}_{rep}.json (real_v3.run_set). Folds replayed from
          rng [30, rep, set]: public() per network, then folds() per unit; pFedMe on rng [30, rep, set, 777], consumed
          fold -> eps -> network -> configuration.
  rn      supplement/results/real_networks_ckpt/ and real_networks_v4_ckpt/ (real_networks.run). The main stream
          [20, i] is consumed by every estimator between repetitions, so it is replayed with real_networks.estimates
          (main stream only; the SA, personalized and v4 streams are separate); pFedMe on rng [22, i, 777], consumed
          repetition -> fold -> eps -> configuration. Both directories share the main and personalized streams
          (checked), so one computation patches both.
  flamby  supplement/results/flamby_ckpt/{heart,brca,isic}_{rep}.json (flamby_bench.run_rep); fixed splits; pFedMe on
          rng [2026, rep, dataset, 777], consumed (head ->) eps -> configuration.

Checks: the Local rows (no noise, depend only on the splits) must reproduce exactly in every v3 and rn checkpoint, and
the pFedMe rows at eps = inf (no noise) in every flamby checkpoint, before any file is written; `validate` also
reproduces the old pFedMe rows exactly with the old (pre-fix) noise on the original streams where that is possible.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np

import ghb
import ghb_glm as GLM
import ghb_logit as GL
import literature as LIT
import personalized as PZ
from real_heart import folds

RES = ghb.RESDIR
NPROC = 2
SUFFIX = "_pre_pfedme_fix"


# ------------------------------------------------------------------------------------------------ the old (bug) code
def pz_pfedme_old(rng, data, n, rho, R, eta, lam, S=5):
    """personalized.pfedme before 15e769a (noise n times too small), for validation only."""
    beta = min(eta * lam, 1.0)
    sd = PZ._sd(n, rho, R, factor=S * beta / lam)
    w = np.zeros(PZ.P)
    for _ in range(R):
        deltas = []
        for X, y in data:
            wk = w.copy()
            for _ in range(S):
                th = PZ.fit(X, y, wk, np.full(PZ.P, lam))
                wk = wk - beta * (wk - th)
            deltas.append(wk - w)
        w = w + PZ._avg(rng, deltas, n, sd)
    return np.array([PZ.fit(X, y, w, np.full(PZ.P, lam)) for X, y in data])


def relmax(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0


def backup(d):
    b = d.parent / (d.name + SUFFIX)
    if not b.exists():
        shutil.copytree(d, b)
        print("backup", b, flush=True)
    return b


# ------------------------------------------------------------------------------------------------------- real_v3
def v3_task(item):
    import real_v3 as RV3
    which, rep = item
    LIT.configure(RV3.P, RV3.G)
    PZ.configure(RV3.P, RV3.G)
    idx = {"dev": 0, "fresh": 1, "confirm": 2}[which]
    rng = np.random.default_rng([30, rep, idx])
    data = {p: RV3.load(p)[0] for p in RV3.SETS[which]}
    for p in data:  # pubs = {p: V3.Public(public(rng)) for p in data}
        RV3.public(rng)
    fs = {p: [folds(rng, y, RV3.FOLDS) for _, y in data[p]] for p in data}
    rng_p = np.random.default_rng([30, rep, idx, 777])
    acc = {}
    for f in range(RV3.FOLDS):
        train = {p: [(X[fs[p][k] != f], y[fs[p][k] != f]) for k, (X, y) in enumerate(data[p])] for p in data}
        test = {p: [(X[fs[p][k] == f], y[fs[p][k] == f]) for k, (X, y) in enumerate(data[p])] for p in data}
        est = {p: {"Local": np.array([RV3.fit(X, y, np.zeros(RV3.P), RV3.PRIOR) for X, y in train[p]])} for p in data}
        for e in RV3.EPS:
            rho = LIT.rho_total(e, RV3.DELTA)
            for p in data:
                n = np.array([len(y) for _, y in train[p]], float)
                for R, eta, lam in PZ.GRID["pFedMe"]:
                    est[p][f"{RV3.ckey(('PME', R, eta, lam))}, eps={e:g}"] = PZ.pfedme(rng_p, train[p], n, rho, R, eta, lam)
        for p in data:
            for m, th in est[p].items():
                ll = np.array([np.sum(np.logaddexp(0, Xt @ th[k]) - yt * (Xt @ th[k])) for k, (Xt, yt) in enumerate(test[p])])
                cnt = np.array([len(yt) for _, yt in test[p]], float)
                a = acc.setdefault((p, m), [np.zeros(len(ll)), np.zeros(len(ll))])
                a[0] += ll; a[1] += cnt
    return which, rep, {f"{p}||{m}": [v[0].tolist(), v[1].tolist()] for (p, m), v in acc.items()}


def v3_apply(which, rep, out, d):
    path = d / f"{which}_{rep}.json"
    ck = json.load(open(path))
    loc = {k: v for k, v in out.items() if k.endswith("||Local")}
    dl = max(max(relmax(v[0], ck["out"][k][0]), relmax(v[1], ck["out"][k][1])) for k, v in loc.items())
    pme = {k: v for k, v in out.items() if "||PME|" in k}
    assert set(pme) == {k for k in ck["out"] if "||PME|" in k}, "pFedMe keys differ"
    if dl > 1e-8:
        raise RuntimeError(f"{path.name}: Local rows not reproduced (max rel diff {dl:.3g}); not patched")
    for k, v in pme.items():
        assert v[1] == ck["out"][k][1]
        ck["out"][k] = v
    json.dump(ck, open(path, "w"))
    return dl


def v3_validate_old(item):
    """real_v3.run_set with the old pFedMe: every row of the checkpoint must be reproduced."""
    import real_v3 as RV3
    which, rep = item
    PZ.pfedme = pz_pfedme_old
    out, _ = RV3.run_set(which, rep)
    ck = json.load(open(RES / "real_v3_ckpt" / f"{which}_{rep}.json"))["out"]
    assert set(out) == set(ck)
    pme = max(relmax(out[k][0], ck[k][0]) for k in out if "||PME|" in k)
    allr = max(relmax(out[k][0], ck[k][0]) for k in out)
    return pme, allr


# ------------------------------------------------------------------------------------------------- real_networks
def rn_task(item):
    import real_networks as RN
    from exp7_logistic import auc
    i, old = item
    name, label = RN.NETWORKS[i]
    rng = np.random.default_rng([20, i])
    data, ids = RN.load(name)
    K = len(data)
    n_all = np.array([len(y) for _, y in data])
    rng_p = np.random.default_rng([22, i] if old else [22, i, 777])
    pub, W = GL.Public(RN.public_features(rng, name)), RN.mixing(K)
    avg = ghb.Averager(W, 60)
    if old:
        PZ.pfedme = pz_pfedme_old
    S = {}
    for rep in range(RN.REPEATS):
        fs = [folds(rng, y, RN.FOLDS) for _, y in data]
        for f in range(RN.FOLDS):
            train, test = [], []
            for k, (X, y) in enumerate(data):
                tr, te = np.flatnonzero(fs[k] != f), np.flatnonzero(fs[k] == f)
                if len(tr) == 0:
                    tr = te
                train.append((X[tr], y[tr])); test.append((X[te], y[te]))
            main = RN.estimates(rng, train, pub, avg, W)  # replays the main stream (next repetition's folds)
            est = {"Local": main["Local"]}
            n = np.array([len(y) for _, y in train], float)
            if old:  # the original personalized stream, every method in its order
                for e in RN.BASE_EPS:
                    est.update({k: v for k, v in PZ.all_estimates(rng_p, train, n, e, RN.DELTA).items() if k.startswith("PME")})
            else:
                for e in RN.BASE_EPS:
                    rho, t = LIT.rho_total(e, RN.DELTA), f"{e:g}"
                    for R, eta, lam in PZ.GRID["pFedMe"]:
                        est[f"PME{t}|R={R}|eta={eta}|lam={lam}"] = PZ.pfedme(rng_p, train, n, rho, R, eta, lam)
            for m, e_ in est.items():
                rec = S.setdefault(m, dict(ll=np.zeros((RN.REPEATS * RN.FOLDS, K)), cnt=np.zeros((RN.REPEATS * RN.FOLDS, K)), s=[], y=[]))
                j = rep * RN.FOLDS + f
                for k, (Xt, yt) in enumerate(test):
                    s = Xt @ e_[k]
                    rec["ll"][j, k] = np.sum(np.logaddexp(0, s) - yt * s)
                    rec["cnt"][j, k] = len(yt)
                    rec["s"].append(s); rec["y"].append(yt)
    small = n_all < np.median(n_all)
    big = n_all >= 20
    loc = S["Local"]

    def metrics(r, full=True):  # real_networks.run's metrics
        hosp, hosp_loc = r["ll"].sum(0) / r["cnt"].sum(0), loc["ll"].sum(0) / loc["cnt"].sum(0)
        out = dict(rel=float(r["ll"].sum() / loc["ll"].sum()),
                   rel_small=float(r["ll"][:, small].sum() / loc["ll"][:, small].sum()),
                   worst=float((hosp[big] / hosp_loc[big]).max()) if big.any() else float("nan"))
        if full:
            ratio = (r["ll"].sum(1) / r["cnt"].sum(1)) / (loc["ll"].sum(1) / loc["cnt"].sum(1))
            sm = (r["ll"][:, small].sum(1) / r["cnt"][:, small].sum(1)) / (loc["ll"][:, small].sum(1) / loc["cnt"][:, small].sum(1))
            out.update(ll=float(r["ll"].sum() / r["cnt"].sum()), rel_se=float(ratio.std(ddof=1) / np.sqrt(len(ratio))),
                       rel_small_se=float(sm.std(ddof=1) / np.sqrt(len(sm))), share_gain=float(np.mean(hosp < hosp_loc)),
                       auc=float(auc(np.concatenate(r["s"]), np.concatenate(r["y"]))))
        return out

    fams = {k: v for k, v in PZ.families(RN.BASE_EPS).items() if k.startswith("pFedMe")}
    cand = {c: metrics(S[c], full=False) for cs in fams.values() for c in cs}
    chosen, famres = {}, {}
    for fam, cs in fams.items():
        best = min(cs, key=lambda c: S[c]["ll"].sum())
        chosen[fam], famres[fam] = best, metrics(S[best])
    return name, dict(old=old, local=metrics(loc), cand=cand, chosen=chosen, fam=famres)


def rn_compare_local(name, r, d):
    ck = json.load(open(d / f"{name}.json"))
    return max(abs(r["local"][k] - ck["Local"][k]) / max(abs(ck["Local"][k]), 1e-300) for k in r["local"])


def rn_apply(name, r, d):
    path = d / f"{name}.json"
    ck = json.load(open(path))
    dl = rn_compare_local(name, r, d)
    if dl > 1e-8:
        raise RuntimeError(f"{d.name}/{name}: Local not reproduced (max rel diff {dl:.3g}); not patched")
    assert set(r["cand"]) == {k for k in ck["cand"] if k.startswith("PME")}
    for c, v in r["cand"].items():
        ck["cand"][c] = v
    for fam in r["fam"]:
        assert fam in ck and fam in ck["chosen"]
        ck["chosen"][fam] = r["chosen"][fam]
        ck[fam] = r["fam"][fam]
    json.dump(ck, open(path, "w"))
    return dl


# --------------------------------------------------------------------------------------------------------- FLamby
def flamby_task(item):
    import flamby_bench as FB
    ds, rep = item
    rng_p = np.random.default_rng([FB.PUBLIC_SEED, rep, FB.DSID[ds], 777])
    grid = PZ.GRID["pFedMe"]
    key = lambda R, eta, lam, e: f"{GLM.ckey(('PME', R, eta, lam))}, eps={FB.etag(e)}"
    est = {}
    if ds in ("heart", "brca"):
        if ds == "heart":
            train, test, _ = FB.load_heart()
            P, G, fam, kind = 14, np.sqrt(14.0), GLM.LOGISTIC, "logistic"
        else:
            train, test, _ = FB.load_brca()
            P, G, fam, kind = 40, FB.BRCA_G, GLM.CLIPPED_POISSON, "poisson"
        n = np.array([len(y) for _, y in train], float)
        for e in FB.EPS:
            rho = GL.rho_total(e, FB.DELTA)
            for R, eta, lam in grid:
                est[key(R, eta, lam, e)] = GLM.pfedme(fam, rng_p, train, n, rho, R, eta, lam, G, P)
    else:
        train, test, _ = FB.load_isic()
        P, G, kind = 17, np.sqrt(17.0), "isic"
        heads = {}
        for c in range(8):
            tc = [(X, (y == c).astype(float)) for X, y in train]
            n = np.array([len(y) for _, y in tc], float)
            for e in FB.EPS:
                rho = GL.rho_total(e, FB.DELTA) * (1 / 8)
                for R, eta, lam in grid:
                    heads.setdefault(key(R, eta, lam, e), [None] * 8)[c] = GLM.pfedme(GLM.LOGISTIC, rng_p, tc, n, rho, R, eta, lam, G, P)
        est = {k: np.array(v).transpose(1, 0, 2) for k, v in heads.items()}
    res = {m: [FB.metrics(kind, th[k], Xt, yt) for k, (Xt, yt) in enumerate(test)] for m, th in est.items()}
    return ds, rep, res


def flamby_diff(a, b):
    """max relative difference between two lists (per hospital) of metric dicts."""
    m = 0.0
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            if x[k] is None or y[k] is None:
                assert x[k] is None and y[k] is None, (k, x[k], y[k])
                continue
            m = max(m, abs(x[k] - y[k]) / max(abs(y[k]), 1e-300))
    return m


def flamby_apply(ds, rep, res, d):
    path = d / f"{ds}_{rep}.json"
    ck = json.load(open(path))
    assert set(res) == {k for k in ck["res"] if k.startswith("PME|")}
    dinf = max(flamby_diff(v, ck["res"][k]) for k, v in res.items() if k.endswith("eps=inf"))
    if dinf > 1e-8:
        raise RuntimeError(f"{path.name}: pFedMe at eps=inf not reproduced (max rel diff {dinf:.3g}); not patched")
    for k, v in res.items():
        if not k.endswith("eps=inf"):  # eps = inf has no noise: unchanged by the fix (checked above), left as stored
            ck["res"][k] = v
    json.dump(ck, open(path, "w"))
    return dinf


# ---------------------------------------------------------------------------------------------------------- driver
def validate(parts):
    t0 = time.time()
    if "flamby" in parts:
        validate_flamby()
    if "rn" in parts:
        validate_rn()
    if "v3" in parts:
        validate_v3()
    print(f"validate {parts}: {time.time() - t0:.0f} s", flush=True)


def validate_flamby():
    # 1. flamby: pFedMe at eps = inf (no noise) against the checkpoints, rep 0 of each dataset
    with Pool(NPROC) as pool:
        for ds, rep, res in pool.imap_unordered(flamby_task, [("heart", 0), ("brca", 0), ("isic", 0)]):
            ck = json.load(open(RES / "flamby_ckpt" / f"{ds}_{rep}.json"))["res"]
            dinf = max(flamby_diff(v, ck[k]) for k, v in res.items() if k.endswith("eps=inf"))
            d1 = max(flamby_diff(v, ck[k]) for k, v in res.items() if not k.endswith("eps=inf"))
            print(f"flamby {ds}_{rep}: pFedMe eps=inf max rel diff {dinf:.3g}; eps=1,10 max rel diff (new vs old) {d1:.3g}", flush=True)


def validate_rn():
    # 2. real_networks, heart: the old pFedMe on the original streams reproduces the stored rows exactly
    #    (and the two checkpoint directories store identical pFedMe rows)
    with Pool(NPROC) as pool:
        for name, r in pool.imap_unordered(rn_task, [(7, True), (7, False)]):
            for d in ("real_networks_ckpt", "real_networks_v4_ckpt"):
                ck = json.load(open(RES / d / f"{name}.json"))
                dl = rn_compare_local(name, r, RES / d)
                dc = max(max(abs(v[k] - ck["cand"][c][k]) / abs(ck["cand"][c][k]) for k in v) for c, v in r["cand"].items())
                df = max(max(abs(v[k] - ck[f][k]) / abs(ck[f][k]) for k in v) for f, v in r["fam"].items())
                print(f"rn {d}/{name} ({'old pFedMe, original stream' if r['old'] else 'new pFedMe'}): "
                      f"Local max rel diff {dl:.3g}; pFedMe candidates {dc:.3g}; tuned families {df:.3g}", flush=True)
    for name, _ in __import__("real_networks").NETWORKS:
        a = json.load(open(RES / "real_networks_ckpt" / f"{name}.json"))
        b = json.load(open(RES / "real_networks_v4_ckpt" / f"{name}.json"))
        same = all(a["cand"][c] == b["cand"][c] for c in a["cand"] if c.startswith("PME")) and \
            all(a[f] == b[f] for f in a if f.startswith("pFedMe")) and a["Local"] == b["Local"]
        print(f"rn {name}: the two directories store identical Local and pFedMe rows: {same}", flush=True)


def validate_v3():
    # 3. real_v3: the replayed folds reproduce the Local rows (new pFedMe differs from the stored one), and run_set
    #    with the old pFedMe reproduces every row of the checkpoint
    with Pool(NPROC) as pool:
        a = pool.apply_async(v3_validate_old, (("dev", 0),))
        which, rep, out = pool.apply(v3_task, (("confirm", 0),))
        ck = json.load(open(RES / "real_v3_ckpt" / f"{which}_{rep}.json"))["out"]
        dl = max(relmax(v[0], ck[k][0]) for k, v in out.items() if k.endswith("||Local"))
        dp = max(relmax(v[0], ck[k][0]) for k, v in out.items() if "||PME|" in k)
        print(f"v3 {which}_{rep}, replayed folds: Local max rel diff {dl:.3g}; pFedMe new vs old max rel diff {dp:.3g}", flush=True)
        pme, allr = a.get()
    print(f"v3 dev_0, run_set with the old pFedMe: pFedMe rows max rel diff {pme:.3g}; all rows {allr:.3g}", flush=True)


def patch(fams):
    t0 = time.time()
    import real_networks as RN
    import real_v3 as RV3
    if "flamby" in fams:
        d = RES / "flamby_ckpt"
        backup(d)
        todo = [(ds, r) for ds in ("isic", "brca", "heart") for r in range(20) if (d / f"{ds}_{r}.json").exists()]
        worst = 0.0
        with Pool(NPROC) as pool:
            for ds, rep, res in pool.imap_unordered(flamby_task, todo):
                dinf = flamby_apply(ds, rep, res, d)
                worst = max(worst, dinf)
                print(f"patched flamby {ds}_{rep} (eps=inf check {dinf:.3g}) {time.time() - t0:.0f} s", flush=True)
        print(f"flamby: {len(todo)} files, max rel diff at eps=inf {worst:.3g}", flush=True)
    if "rn" in fams:
        dirs = [RES / "real_networks_ckpt", RES / "real_networks_v4_ckpt"]
        for d in dirs:
            backup(d)
        order = ["medpar", "TCGA-BRCA", "TCGA-LUSC", "TCGA-HNSC", "TCGA-BLCA", "TCGA-KIRC", "TCGA-STAD", "TCGA-LIHC", "heart"]
        todo = sorted([(i, False) for i, (nm, _) in enumerate(RN.NETWORKS)],
                      key=lambda it: order.index(RN.NETWORKS[it[0]][0]) if RN.NETWORKS[it[0]][0] in order else 0)
        worst = 0.0
        with Pool(NPROC) as pool:
            for name, r in pool.imap_unordered(rn_task, todo):
                for d in dirs:
                    dl = rn_apply(name, r, d)
                    worst = max(worst, dl)
                    print(f"patched {d.name}/{name} (Local check {dl:.3g}) {time.time() - t0:.0f} s", flush=True)
        print(f"rn: max rel diff of Local {worst:.3g}", flush=True)
    if "v3" in fams:
        d = RES / "real_v3_ckpt"
        backup(d)
        todo = [(w, r) for w in ("dev", "fresh", "confirm") for r in range(RV3.REPEATS) if (d / f"{w}_{r}.json").exists()]
        worst = 0.0
        with Pool(NPROC) as pool:
            for which, rep, out in pool.imap_unordered(v3_task, todo):
                dl = v3_apply(which, rep, out, d)
                worst = max(worst, dl)
                print(f"patched real_v3 {which}_{rep} (Local check {dl:.3g}) {time.time() - t0:.0f} s", flush=True)
        print(f"v3: max rel diff of Local {worst:.3g}", flush=True)
    print(f"patch: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        validate(sys.argv[2:] or ["flamby", "rn", "v3"])
    elif cmd == "patch":
        patch(sys.argv[2:] or ["flamby", "rn", "v3"])
