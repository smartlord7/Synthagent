"""A second held-out set: three CPTAC networks (plan in dev/CPTAC_PLAN.md, committed before this was run).

Units are countries of birth (CPTAC does not release the tissue source site; its centres are country-specific), the
networks are cancer cohorts, and outcome, features and exclusions are those of real_v3.py. GHB as released
(real_v3.GHB_V5) fits the three networks together; every literature method is fitted per network at every
configuration of its grid. Realistic arm: the configurations frozen on the 12 TCGA design networks
(dev/v4_realistic_configs.json). Oracle arm: the best configuration on each network's own test folds.
    python3 fetch_tcga.py is not used: data/cptac/CPTAC.json comes from the GDC API (see fetch() below)
    python3 cptac_heldout.py run     python3 cptac_heldout.py score     (results/cptac_heldout.json)
"""
import json
import sys
import urllib.parse
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ghb
import ghb_sa as SA
import ghb_v3 as V3
import ghb_v4 as V4
import literature as LIT
import personalized as PZ
import real_v3 as RV
from exp7_logistic import fit
from fetch_tcga import FIELDS
from real_heart import folds

DATA = Path(__file__).resolve().parents[1] / "data" / "cptac" / "CPTAC.json"
NETS = ["CPTAC-PDA", "CPTAC-CM", "CPTAC-LUAD"]  # fixed from counts alone (dev/CPTAC_PLAN.md)
CKPT = ghb.RESDIR / "cptac_ckpt"
OUT = ghb.RESDIR / "cptac_heldout.json"
MISSING = (None, "", "Unknown", "Not Reported", "not reported")


def fetch():
    f = FIELDS + (",disease_type,primary_site,project.project_id,demographic.country_of_birth,"
                  "diagnoses.age_at_diagnosis,diagnoses.primary_diagnosis")
    filt = {"op": "in", "content": {"field": "project.project_id", "value": ["CPTAC-3", "CPTAC-2"]}}
    q = urllib.parse.urlencode({"filters": json.dumps(filt), "fields": f, "size": 5000, "format": "json"})
    with urllib.request.urlopen("https://api.gdc.cancer.gov/cases?" + q, timeout=300) as r:
        hits = json.load(r)["data"]["hits"]
    DATA.parent.mkdir(exist_ok=True)
    json.dump(hits, open(DATA, "w"))


def cohort(h):
    s, d = h.get("primary_site"), h.get("disease_type") or ""
    if s == "Bronchus and lung":
        return "LUAD" if d.startswith("Aden") else "LSCC" if d.startswith("Squamous") else None
    if d.startswith("Squamous"):
        return "HNSC"
    return {"Kidney": "CCRCC", "Uterus, NOS": "UCEC", "Skin": "CM", "Stomach": "STAD", "Pancreas": "PDA",
            "Breast": "BRCA", "Colon": "COAD", "Ovary": "OV", "Brain": "GBM"}.get(s)


def load_all():
    units = {}
    for h in json.load(open(DATA)):
        d = dict(h.get("demographic") or {})
        dgs = [x for x in (h.get("diagnoses") or []) if x.get("diagnosis_is_primary_disease") is not False] or (h.get("diagnoses") or [{}])
        if dgs[0].get("age_at_diagnosis") is not None:
            d["age_at_index"] = dgs[0]["age_at_diagnosis"] / 365.25
        fu = max([x.get("days_to_last_follow_up") or -1 for x in dgs] + [-1])
        dead, dtd = d.get("vital_status") == "Dead", d.get("days_to_death")
        if dead and dtd is not None and dtd <= 730:
            y = 1.0
        elif (dead and dtd is not None and dtd > 730) or (not dead and fu > 730):
            y = 0.0
        else:
            continue
        x, c, net = RV.encode(d, dgs[0]), d.get("country_of_birth"), cohort(h)
        if x is None or c in MISSING or net is None:
            continue
        units.setdefault("CPTAC-" + net, {}).setdefault(c, []).append((x, y))
    out = {}
    for net, u in units.items():
        rows = [(c, r) for c, r in sorted(u.items(), key=lambda kv: -len(kv[1])) if len(r) >= RV.MIN_UNIT]
        out[net] = ([(np.array([a for a, _ in r]), np.array([b for _, b in r])) for _, r in rows], [c for c, _ in rows])
    return out


def counts():
    """The inclusion rule of the TCGA held-out set (>= 4 units, >= 15 deaths), from counts alone."""
    res = {}
    for net, (data, ids) in sorted(load_all().items()):
        res[net] = dict(units=len(data), patients=int(sum(len(y) for _, y in data)), deaths=int(sum(y.sum() for _, y in data)),
                        sizes=[len(y) for _, y in data], countries=ids)
        res[net]["included"] = res[net]["units"] >= 4 and res[net]["deaths"] >= 15
    return res


def run(rep):
    P, G = RV.P, RV.G
    LIT.configure(P, G); PZ.configure(P, G); SA.G = G; SA.P = P
    allnets = load_all()
    data = {p: allnets[p][0] for p in NETS}
    inst = {p: allnets[p][1] for p in NETS}
    rng = np.random.default_rng([50, rep])
    fs = {p: [folds(rng, y, RV.FOLDS) for _, y in data[p]] for p in data}
    pubs = {p: V3.Public(RV.public(np.random.default_rng([52, rep, i]))) for i, p in enumerate(NETS)}
    acc = {}
    for f in range(RV.FOLDS):
        train = {p: [(X[fs[p][k] != f], y[fs[p][k] != f]) for k, (X, y) in enumerate(data[p])] for p in data}
        test = {p: [(X[fs[p][k] == f], y[fs[p][k] == f]) for k, (X, y) in enumerate(data[p])] for p in data}
        est = {p: {"Local": np.array([fit(X, y, np.zeros(P), RV.PRIOR) for X, y in train[p]]),
                   "Local, evidence prior": np.array([V3.combine(X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in V3.SDS]) for X, y in train[p]])}
               for p in data}
        for e in RV.EPS:
            res = V4.estimate(train, np.random.default_rng([51, rep, f, int(e)]), e, RV.DELTA, G, inst=inst, cfg=RV.GHB_V5)
            lite = V3.estimate(train, pubs, np.random.default_rng([53, rep, f, int(e)]), e, RV.DELTA, G, inst=inst, cfg=RV.VARIANTS["GHB-EB"])
            for p in data:
                est[p][f"GHB, eps={e:g}"] = res[p]
                est[p][f"GHB-lite, eps={e:g}"] = lite[p]
            brng = np.random.default_rng([54, rep, f, int(e)])
            for p in data:
                for k, v in RV.baselines(brng, train[p], e).items():
                    est[p][f"{k}, eps={e:g}"] = v
        for p in data:
            for m, th in est[p].items():
                ll = np.array([np.sum(np.logaddexp(0, Xt @ th[k]) - yt * (Xt @ th[k])) for k, (Xt, yt) in enumerate(test[p])])
                cnt = np.array([len(yt) for _, yt in test[p]], float)
                a = acc.setdefault(f"{p}||{m}", [np.zeros(len(ll)), np.zeros(len(ll))])
                a[0] += ll; a[1] += cnt
    json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in acc.items()}, open(CKPT / f"cptac_{rep}.json", "w"))
    return rep


def score():
    files = sorted(CKPT.glob("cptac_*.json"))
    acc = {}
    for fn in files:
        for k, (ll, c) in json.load(open(fn)).items():
            a = acc.setdefault(k, [0.0, 0.0]); a[0] += sum(ll); a[1] += sum(c)
    L = lambda p, m: (acc[f"{p}||{m}"][0] / acc[f"{p}||{m}"][1]) / (acc[f"{p}||Local"][0] / acc[f"{p}||Local"][1])
    sel = json.load(open(Path(__file__).resolve().parent / "dev" / "v4_realistic_configs.json"))
    out = {"reps": len(files), "nets": NETS, "counts": counts(), "rel": {}}
    for e in ("1", "10"):
        r = {"GHB v5": [L(p, f"GHB, eps={e}") for p in NETS], "GHB-lite": [L(p, f"GHB-lite, eps={e}") for p in NETS],
             "Local, evidence prior": [L(p, "Local, evidence prior") for p in NETS]}
        for fam, grid in RV.GRIDS.items():
            r[fam + " (realistic)"] = [L(p, f"{sel[f'{fam}, eps={e}']}, eps={e}") for p in NETS]
            R = np.array([[L(p, f"{RV.ckey(c)}, eps={e}") for p in NETS] for c in grid])
            r[fam + " (oracle)"] = list(R.min(0))
        r["Best per network (oracle)"] = list(np.min([r[m] for m in r if m.endswith("(oracle)")], axis=0))
        out["rel"][e] = r
        print(f"eps={e} ({len(files)} repetitions)")
        for m, v in r.items():
            print(f"  {m:34s} " + " ".join(f"{x:.3f}" for x in v) + f"  mean {np.mean(v):.4f}")
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        fetch()
    elif cmd == "counts":
        print(json.dumps(counts(), indent=1))
    elif cmd == "score":
        score()
    else:
        CKPT.mkdir(exist_ok=True)
        todo = [r for r in range(RV.REPEATS) if not (CKPT / f"cptac_{r}.json").exists()]
        with Pool(4) as pool:
            for r in pool.imap_unordered(run, todo):
                print("done", r, flush=True)
