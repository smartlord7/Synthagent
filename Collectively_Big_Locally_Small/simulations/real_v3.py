"""Real data 3: GHB v3 on TCGA with richer models and cross-cancer borrowing.

Two sets of TCGA networks, each network a cancer type and each unit an institution (tissue source site name) with at
least MIN_UNIT usable patients:
  dev    LUSC, HNSC, BLCA, STAD, LIHC, KIRC (already used in real_networks.py) - development and tuning
  fresh  LUAD, COAD, READ, KIRP, PAAD, ESCA (never used before) - confirmation, run once after pre-registration
Outcome: death within two years of diagnosis (patients alive with less than two years of follow-up excluded).
Features (P = 8, ||x|| <= sqrt(8)): intercept, age ((age - 62) / 25 clipped to [-1, 1]), male, pathologic stage III,
stage IV, T3 or T4, any nodal involvement (N1-N3), prior malignancy; patients without a pathologic stage are excluded.
Evaluation: repeated stratified five-fold cross-validation within each unit (REPEATS repeats), all networks of a set
fitted jointly (cross-cancer borrowing needs them together; every other method is fitted per network).
Methods: local (default prior; evidence prior), GHB-EB (= GHB v3 with every addition off), each addition of GHB v3
alone and all together, at epsilon in {1, 10}; baselines at the same budgets: DP-FedAvg + fine-tuning (with and
without secure aggregation), APFL, Ditto, pFedMe, clustered DP-FedAvg, each with every configuration of its grid
stored, so that both arms can be computed afterwards: oracle (tuned on the network's own held-out folds) and realistic
(the configuration with the lowest mean relative error over the dev networks, applied unchanged to the fresh ones).
Results are checkpointed per set.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ghb
import ghb_logit as GL
import ghb_sa as SA
import ghb_v3 as V3
import ghb_v4 as V4
import literature as LIT
import personalized as PZ
from exp7_logistic import fit
from real_heart import folds

DATA = Path(__file__).resolve().parents[1] / "data" / "tcga_v3"
SETS = {"dev": ["TCGA-LUSC", "TCGA-HNSC", "TCGA-BLCA", "TCGA-STAD", "TCGA-LIHC", "TCGA-KIRC"],
        "fresh": ["TCGA-LUAD", "TCGA-COAD", "TCGA-READ", "TCGA-KIRP", "TCGA-PAAD", "TCGA-ESCA"],
        # third set (data/tcga_v4): every further TCGA project with >= 4 institutions and >= 15 deaths under the rules
        # above (AJCC or, for gynecologic cancers, FIGO stage), fixed from counts alone before any model was fitted
        "confirm": ["TCGA-BRCA", "TCGA-MESO", "TCGA-UCEC", "TCGA-OV", "TCGA-UCS"]}
GHB_V4 = dict(f=1.0, safety=True, R1=1, pan=True)  # locked in dev/PREREG_v4.md
P, G = 8, np.sqrt(8.0)
REPEATS, FOLDS, DELTA, MIN_UNIT = 10, 5, 1e-5, 5
EPS = (1.0, 10.0)
PRIOR = np.full(P, 1 / 2.5 ** 2)
VARIANTS = {"GHB-EB": dict(safety=False, learned_scale=False, fullcov=False, cross=False),
            "+ safety": dict(safety=True, learned_scale=False, fullcov=False, cross=False),
            "+ learned scale": dict(safety=False, learned_scale=True, fullcov=False, cross=False),
            "+ correlated": dict(safety=False, learned_scale=False, fullcov=True, cross=False),
            "+ cross-cancer": dict(safety=False, learned_scale=False, fullcov=False, cross=True),
            "GHB v3": dict(safety=True, learned_scale=True, fullcov=True, cross=True)}
GRIDS = {
    "DP-FedAvg + FT": [("FT", R, eta, E) for R in (1, 3, 10, 30) for eta in (0.2, 0.5, 2.0) for E in (1, 5, 30)],
    "DP-FedAvg-SA + FT": [("FTSA", R, eta, E) for R in (3, 10, 30) for eta in (0.5, 2.0) for E in (0, 5, 30)],
    "APFL": [("APFL", R, eta, a) for R, eta, a in PZ.GRID["APFL"]],
    "Ditto": [("PDIT", R, eta, lam) for R, eta, lam in PZ.GRID["Ditto"]],
    "pFedMe": [("PME", R, eta, lam) for R, eta, lam in PZ.GRID["pFedMe"]],
    "Clustered DP-FedAvg": [("CLU", C, R, eta, E) for C in (2, 3) for R in (1, 3, 10) for eta in (0.2, 0.5, 2.0) for E in (0, 5, 30)],
}


def ckey(c):
    return "|".join(str(x) for x in c)


def encode(d, dg):
    age, sex = d.get("age_at_index"), d.get("sex_at_birth") or d.get("gender")
    stage = dg.get("ajcc_pathologic_stage") or dg.get("figo_stage") or ""  # FIGO only where AJCC is absent (gynecologic)
    if age is None or sex not in ("male", "female") or not stage.startswith("Stage"):
        return None
    s = stage.replace("Stage ", "").rstrip("ABCabc12")
    t, nn = (dg.get("ajcc_pathologic_t") or ""), (dg.get("ajcc_pathologic_n") or "")
    return [1.0, float(np.clip((age - 62) / 25, -1, 1)), float(sex == "male"), float(s == "III"), float(s == "IV"),
            float(t.startswith("T3") or t.startswith("T4")), float(nn[:2] in ("N1", "N2", "N3")),
            float((dg.get("prior_malignancy") or "") == "yes")]


def load(project):
    units = {}
    path = DATA / f"{project}.json"
    if not path.exists():
        path = DATA.parent / "tcga_v4" / f"{project}.json"
    for c in json.load(open(path)):
        d = c.get("demographic") or {}
        dgs = [x for x in (c.get("diagnoses") or []) if x.get("diagnosis_is_primary_disease") is not False] or (c.get("diagnoses") or [{}])
        fu = max([x.get("days_to_last_follow_up") or -1 for x in dgs] + [-1])
        dead, dtd = d.get("vital_status") == "Dead", d.get("days_to_death")
        if dead and dtd is not None and dtd <= 730:
            y = 1.0
        elif (dead and dtd is not None and dtd > 730) or (not dead and fu > 730):
            y = 0.0
        else:
            continue
        x = encode(d, dgs[0])
        name = (c.get("tissue_source_site") or {}).get("name")
        if x is None or not name:
            continue
        units.setdefault(name, []).append((x, y))
    out, ids = [], []
    for name, rows in sorted(units.items(), key=lambda kv: -len(kv[1])):
        if len(rows) >= MIN_UNIT:
            out.append((np.array([r[0] for r in rows]), np.array([r[1] for r in rows])))
            ids.append(name)
    return out, ids


def public(rng, m=20000):
    a = np.clip((rng.normal(62, 12, m) - 62) / 25, -1, 1)
    s = rng.choice(3, m, p=[0.65, 0.25, 0.10])
    b = lambda p: (rng.random(m) < p).astype(float)
    return np.column_stack([np.ones(m), a, b(0.5), s == 1, s == 2, b(0.35), b(0.4), b(0.1)]).astype(float)


def baselines(rng, train, eps):
    n = np.array([len(y) for _, y in train], float)
    K = len(train)
    rho = LIT.rho_total(eps, DELTA)
    out = {}
    z = np.zeros(P)
    for R in (1, 3, 10, 30):
        for eta in (0.2, 0.5, 2.0):
            g = LIT.fedavg(rng, train, n, z, rho, R, eta)
            for E, th in LIT.fine_tune(train, np.repeat(g[None], K, 0), steps=(1, 5, 30)).items():
                out[ckey(("FT", R, eta, E))] = th
    for R in (3, 10, 30):
        for eta in (0.5, 2.0):
            g = SA.fedavg_sa(rng, train, n, z, rho, R, eta)
            for E, th in LIT.fine_tune(train, np.repeat(g[None], K, 0), steps=(0, 5, 30)).items():
                out[ckey(("FTSA", R, eta, E))] = th
    for R, eta, a in PZ.GRID["APFL"]:
        out[ckey(("APFL", R, eta, a))] = PZ.apfl(rng, train, n, rho, R, eta, a)
    for R, eta, lam in PZ.GRID["Ditto"]:
        out[ckey(("PDIT", R, eta, lam))] = PZ.ditto(rng, train, n, rho, R, eta, lam)
    for R, eta, lam in PZ.GRID["pFedMe"]:
        out[ckey(("PME", R, eta, lam))] = PZ.pfedme(rng, train, n, rho, R, eta, lam)
    for C in (2, 3):
        lab = LIT.private_clusters(rng, train, n, rho / 2, C)
        for R in (1, 3, 10):
            for eta in (0.2, 0.5, 2.0):
                th0 = np.empty((K, P))
                for c in range(C):
                    mem = np.flatnonzero(lab == c)
                    if len(mem):
                        th0[mem] = LIT.fedavg(rng, train, n, z, rho / 2, R, eta, mem)
                for E, th in LIT.fine_tune(train, th0, steps=(0, 5, 30)).items():
                    out[ckey(("CLU", C, R, eta, E))] = th
    return out


def run_set(which, rep):
    LIT.configure(P, G)
    PZ.configure(P, G)
    SA.G = G
    SA.P = P
    rng = np.random.default_rng([30, rep, {"dev": 0, "fresh": 1, "confirm": 2}[which]])
    nets = {p: load(p) for p in SETS[which]}
    data = {p: v[0] for p, v in nets.items()}
    inst = {p: v[1] for p, v in nets.items()}
    pubs = {p: V3.Public(public(rng)) for p in data}
    acc = {}  # (network, method) -> (ll sums per unit, counts per unit)
    fs = {p: [folds(rng, y, FOLDS) for _, y in data[p]] for p in data}
    for f in range(FOLDS):
        train = {p: [(X[fs[p][k] != f], y[fs[p][k] != f]) for k, (X, y) in enumerate(data[p])] for p in data}
        test = {p: [(X[fs[p][k] == f], y[fs[p][k] == f]) for k, (X, y) in enumerate(data[p])] for p in data}
        est = {p: {"Local": np.array([fit(X, y, np.zeros(P), PRIOR) for X, y in train[p]]),
                   "Local, evidence prior": np.array([V3.combine(X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in V3.SDS]) for X, y in train[p]])}
               for p in data}
        for e in EPS:
            for name, cfg in VARIANTS.items():
                if which == "confirm" and name not in ("GHB-EB", "+ cross-cancer"):
                    continue
                res = V3.estimate(train, pubs, rng, e, DELTA, G, inst=inst, cfg=cfg)
                for p in data:
                    est[p][f"{name}, eps={e:g}"] = res[p]
            if which == "confirm":
                res = V4.estimate(train, rng, e, DELTA, G, inst=inst, cfg=GHB_V4)
                for p in data:
                    est[p][f"GHB v4, eps={e:g}"] = res[p]
            for p in data:
                for k, v in baselines(rng, train[p], e).items():
                    est[p][f"{k}, eps={e:g}"] = v
        for p in data:
            for m, th in est[p].items():
                ll = np.array([np.sum(np.logaddexp(0, Xt @ th[k]) - yt * (Xt @ th[k])) for k, (Xt, yt) in enumerate(test[p])])
                cnt = np.array([len(yt) for _, yt in test[p]], float)
                a = acc.setdefault((p, m), [np.zeros(len(ll)), np.zeros(len(ll))])
                a[0] += ll; a[1] += cnt
    return {f"{p}||{m}": [v[0].tolist(), v[1].tolist()] for (p, m), v in acc.items()}, {p: dict(K=len(data[p]), n=[len(y) for _, y in data[p]], ids=inst[p]) for p in data}


def task(item):
    which, rep = item
    out, info = run_set(which, rep)
    print(which, rep, "done", flush=True)
    return which, rep, out, info


CKPT = ghb.RESDIR / "real_v3_ckpt"


def main(which):
    CKPT.mkdir(exist_ok=True)
    todo = [(which, r) for r in range(REPEATS) if not (CKPT / f"{which}_{r}.json").exists()]
    print("to run:", todo, flush=True)
    with Pool(4) as pool:
        for w, r, out, info in pool.imap_unordered(task, todo):
            json.dump(dict(out=out, info=info), open(CKPT / f"{w}_{r}.json", "w"))
            print("checkpointed", w, r, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
