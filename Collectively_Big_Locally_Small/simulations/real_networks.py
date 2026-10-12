"""Real data 2: networks of many small hospitals.

Seven real networks, each a set of hospitals that contributed patients, with the hospital as the unit of
collaboration:
  medpar   Arizona Medicare 1991, admissions for percutaneous cardiovascular procedures (DRG 112; R package COUNT,
           Hilbe 2011): 1,495 patients from 54 hospitals (providers), outcome in-hospital death; features: age 80 or
           over, HMO membership, urgent admission, emergency admission (elective as reference).
  TCGA     The Cancer Genome Atlas, clinical data from the NCI Genomic Data Commons, one network per cancer type
           (lung squamous LUSC, head and neck HNSC, bladder BLCA, stomach STAD, liver LIHC, kidney clear cell KIRC),
           the hospital being the tissue source site coded in the patient barcode; outcome death within two years of
           diagnosis (patients alive with less than two years of follow-up are excluded); features: age (scaled by
           public constants, (age - 62) / 25 clipped to [-1, 1]), male sex, pathologic stage III, stage IV (stages I
           and II as reference; patients without a pathologic stage are excluded).
Hospitals with fewer than MIN_SITE usable patients are dropped. Every x satisfies ||x|| <= 2 <= sqrt(5).

Evaluation and methods as in real_heart.py: repeated stratified five-fold cross-validation within each hospital
(REPEATS repeats); GHB without an anchor at epsilon in {1, 3, 10} and without privacy, gossip on the ring with
two-hop links (T = 60); the public feature distribution is nominal and independent (MEDPAR_PUBLIC, TCGA_PUBLIC), with
a non-private diagnostic that uses the pooled empirical features; local MAP and the pooled model with the default
prior N(0, 2.5^2); local gradient descent with tuned stopping, DP-FedAvg + fine-tuning and Ditto-style at epsilon = 1,
all three tuned on the held-out folds. Metrics: held-out log-loss over all test patients (patient-weighted), relative
to local estimation, for all hospitals and for the hospitals below the median size; the largest ratio to local over
hospitals with at least 20 patients; and the AUC of the network's pooled held-out predictions.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import ghb
import ghb_logit as GL
import literature as LIT
import ghb_sa as SA
import personalized as PZ
import style
from exp7_logistic import P, auc, fit
from real_heart import folds

DATA = Path(__file__).resolve().parents[1] / "data"
REPEATS, FOLDS, DELTA, MIN_SITE = 20, 5, 1e-5, 5
BASE_EPS = (1.0, 10.0)  # budgets at which every private baseline is run, as GHB
PRIOR = np.full(P, 1 / 2.5 ** 2)
EPS_GHB = (1.0, 3.0, 10.0)
TCGA = [("TCGA-LUSC", "Lung squamous"), ("TCGA-HNSC", "Head and neck"), ("TCGA-BLCA", "Bladder"),
        ("TCGA-STAD", "Stomach"), ("TCGA-LIHC", "Liver"), ("TCGA-KIRC", "Kidney")]
NETWORKS = [("medpar", "Arizona Medicare")] + TCGA + [("heart", "UCI Heart Disease")]
# nominal public feature distributions, set before looking at the data
MEDPAR_PUBLIC = dict(age80=0.15, hmo=0.15, urgent=0.25, emergency=0.25)
TCGA_PUBLIC = dict(age_mean=62, age_sd=12, male=0.5, stage3=0.25, stage4=0.10)
FT_GRID = [(R, eta, E) for R in (1, 3, 10, 30) for eta in (0.2, 0.5, 2.0) for E in (1, 5, 30)]
AVG_GRID = [(R, eta) for R in (1, 3, 10, 30) for eta in (0.2, 0.5, 2.0)]
DEC_GRID = [(R, eta, E) for R in (3, 10, 30) for eta in (0.2, 0.5, 2.0) for E in (0, 5, 30)]
CLU_GRID = [(C, R, eta, E) for C in (2, 3) for R in (1, 3, 10) for eta in (0.2, 0.5, 2.0) for E in (0, 5, 30)]
DITTO_GRID = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LGD_STEPS = (1, 2, 3, 5, 10, 30, 100)
FAMILIES = ["DP-FedAvg", "DP-FedAvg + fine-tuning", "Decentralized DP-SGD", "Clustered DP-FedAvg", "Ditto-style (DP)"]
TUNED = {"Local GD, tuned stopping": [f"LGD|E={E}" for E in LGD_STEPS],
         "Local, tuned prior": [f"LOCP|sd={s_}" for s_ in (0.5, 1.0, 2.5)],
         "GHB, tuned prior (no DP)": [f"GHBPinf|sd={s_}" for s_ in (0.5, 1.0, 2.5)]}
for _e in BASE_EPS:
    TUNED[f"DP-FedAvg-SA + fine-tuning, eps={_e:g}"] = [f"FTSA{_e:g}|R={R}|eta={eta}|E={E}" for R in (3, 10, 30) for eta in (0.5, 2.0) for E in (0, 5, 30)]
    TUNED[f"GHB, tuned prior (DP), eps={_e:g}"] = [f"GHBP{_e:g}|sd={s_}" for s_ in (0.5, 1.0, 2.5)]
    TUNED[f"DP-FedAvg, eps={_e:g}"] = [f"AVG{_e:g}|R={R}|eta={eta}" for R, eta in AVG_GRID]
    TUNED[f"DP-FedAvg + fine-tuning, eps={_e:g}"] = [f"FT{_e:g}|R={R}|eta={eta}|E={E}" for R, eta, E in FT_GRID]
    TUNED[f"Decentralized DP-SGD, eps={_e:g}"] = [f"DEC{_e:g}|R={R}|eta={eta}|E={E}" for R, eta, E in DEC_GRID]
    TUNED[f"Clustered DP-FedAvg, eps={_e:g}"] = [f"CLU{_e:g}|C={C}|R={R}|eta={eta}|E={E}" for C, R, eta, E in CLU_GRID]
    TUNED[f"Ditto-style (DP), eps={_e:g}"] = [f"DIT{_e:g}|lambda={lam}" for lam in DITTO_GRID]
TUNED.update(PZ.families(BASE_EPS))  # personalized federated baselines (personalized.py)
GHB_NAMES = [f"GHB (DP, eps={e:g})" for e in EPS_GHB]
METHODS = (["Local"] + GHB_NAMES + ["GHB (no DP)", "GHB, empirical features (no DP)", "Global (pooled, no DP)", "Local GD, tuned stopping",
                                    "Local, tuned prior", "GHB, tuned prior (no DP)", "Local, evidence prior"]
           + [f"GHB-EB {e:g}" for e in EPS_GHB] + ["GHB-EB inf"]
           + [f"GHB-SA {e:g}" for e in EPS_GHB] + ["GHB-SA inf"]
           + [f"Meta-analysis (DP), eps={e:g}" for e in BASE_EPS]
           + [k for k in TUNED if k not in ("Local GD, tuned stopping", "Local, tuned prior", "GHB, tuned prior (no DP)")])


def load_medpar():
    d = np.genfromtxt(DATA / "medpar" / "medpar.csv", delimiter=",", names=True, dtype=None, encoding=None)
    sites = {}
    for r in d:
        x = [1.0, r["age80"], r["hmo"], r["type"] == 2, r["type"] == 3]
        sites.setdefault(int(r["provnum"]), []).append((x, float(r["died"])))
    return sites


def load_tcga(project):
    hits = json.load(open(DATA / "tcga" / f"{project}.json"))
    sites = {}
    for c in hits:
        d = c.get("demographic") or {}
        dg = [x for x in (c.get("diagnoses") or []) if x.get("diagnosis_is_primary_disease") is not False] or (c.get("diagnoses") or [{}])
        fu = max([x.get("days_to_last_follow_up") or -1 for x in dg] + [-1])
        dead, dtd = d.get("vital_status") == "Dead", d.get("days_to_death")
        if dead and dtd is not None and dtd <= 730:
            y = 1.0
        elif (dead and dtd is not None and dtd > 730) or (not dead and fu > 730):
            y = 0.0
        else:
            continue
        stage, age, sex = (dg[0].get("ajcc_pathologic_stage") or ""), d.get("age_at_index"), d.get("sex_at_birth") or d.get("gender")
        if not stage.startswith("Stage") or age is None or sex not in ("male", "female"):
            continue
        s = stage.replace("Stage ", "").rstrip("ABCabc")
        x = [1.0, float(np.clip((age - 62) / 25, -1, 1)), sex == "male", s == "III", s == "IV"]
        sites.setdefault(c["submitter_id"].split("-")[1], []).append((x, y))
    return sites


def load(name):
    if name == "heart":  # real_heart.py's encoding: age, male, asymptomatic, anginal chest pain
        import real_heart
        return real_heart.load(), [s for _, s in real_heart.SITES]
    raw = load_medpar() if name == "medpar" else load_tcga(name)
    data, ids = [], []
    for sid, rows in sorted(raw.items(), key=lambda kv: -len(kv[1])):
        if len(rows) >= MIN_SITE:
            data.append((np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float)))
            ids.append(str(sid))
    return data, ids


def public_features(rng, name, m=20000):
    if name == "heart":
        import real_heart
        return real_heart.public_features(rng, m)
    if name == "medpar":
        p = MEDPAR_PUBLIC
        t = rng.choice(3, m, p=[1 - p["urgent"] - p["emergency"], p["urgent"], p["emergency"]])
        return np.column_stack([np.ones(m), rng.random(m) < p["age80"], rng.random(m) < p["hmo"], t == 1, t == 2]).astype(float)
    p = TCGA_PUBLIC
    a = np.clip((rng.normal(p["age_mean"], p["age_sd"], m) - 62) / 25, -1, 1)
    s = rng.choice(3, m, p=[1 - p["stage3"] - p["stage4"], p["stage3"], p["stage4"]])
    return np.column_stack([np.ones(m), a, rng.random(m) < p["male"], s == 1, s == 2]).astype(float)


def mixing(K):
    sys.path.insert(0, str(ghb.ROOT / "scripts"))
    import fig_gossip as fg
    return ghb.metropolis_hastings(fg.ring(K, hops=2 if K > 4 else 1))


def gossip(K):
    return ghb.Averager(mixing(K), 60)


PRIOR_SDS = (0.5, 1.0, 2.5)  # default-prior scales offered to GHB and local estimation when tuned like the baselines


def ghb_fit(rng, train, n, pub, avg, eps, tuned=None):
    """GHB at budget eps; with tuned = (est, key), also stores est[key + f"|sd={s}"] for every s in PRIOR_SDS."""
    mu0 = np.zeros(P)
    g, sd = GL.release(rng, train, n, mu0, eps, DELTA)
    th, r, drop = GL.invert(pub, g, sd, n, mu0, avg, anchor_sd=GL.PRIOR_SD, check=False, pool_prior_sd=GL.PRIOR_SD)
    out = GL.estimate(train, th, r, drop, mu0, avg, anchor_sd=None)
    if tuned is not None:
        est, key = tuned
        for s_ in PRIOR_SDS:
            est[f"{key}|sd={s_}"] = out if s_ == GL.PRIOR_SD else GL.estimate(train, th, r, drop, mu0, avg, anchor_sd=None, prior_sd=s_)
        # the evidence-weighted default prior (dev/PREREG_prior.md): the same releases, so no extra random draws
        est[key.replace("GHBP", "GHB-EB ")] = GL.estimate(train, th, r, drop, mu0, avg, anchor_sd=None, prior_sd=GL.PRIOR_SDS)
    return out, th, r


SA_C = 8.0  # GHB-SA version 2, frozen on development data (dev/sa_dev2.py, dev/PREREG_sa.md): C = 8, no round 3


def estimates(rng, train, pub, avg, W, rng_sa=None, rng_p=None):
    n = np.array([len(y) for _, y in train], float)
    K = len(n)
    est = {"Local": np.array([fit(X, y, np.zeros(P), PRIOR) for X, y in train])}
    rel = {}
    est["Local, evidence prior"] = np.array([GL.combine(X, y, [np.zeros(P)], [np.full(P, 1e12)], GL.PRIOR_SDS) for X, y in train])
    for s_ in PRIOR_SDS:
        est[f"LOCP|sd={s_}"] = np.array([fit(X, y, np.zeros(P), np.full(P, 1 / s_ ** 2)) for X, y in train])
    for e, name in zip(EPS_GHB, GHB_NAMES):
        est[name], th, r = ghb_fit(rng, train, n, pub, avg, e, (est, f"GHBP{e:g}"))
        rel[e] = (th, r)
    est["GHB (no DP)"], _, _ = ghb_fit(rng, train, n, pub, avg, np.inf, (est, "GHBPinf"))
    est["GHB, empirical features (no DP)"], _, _ = ghb_fit(rng, train, n, GL.Public(np.vstack([X for X, _ in train])), avg, np.inf)
    Xa = np.vstack([X for X, _ in train]); ya = np.concatenate([y for _, y in train])
    est["Global (pooled, no DP)"] = np.repeat(fit(Xa, ya, np.zeros(P), PRIOR)[None], K, 0)
    for E, th in LIT.fine_tune(train, np.zeros((K, P)), steps=LGD_STEPS).items():
        est[f"LGD|E={E}"] = th
    if rng_p is not None:  # personalized baselines, on their own random stream
        for e in BASE_EPS:
            est.update(PZ.all_estimates(rng_p, train, np.array([len(y) for _, y in train], float), e, DELTA))
    if rng_sa is not None:  # secure aggregation: GHB-SA and DP-FedAvg-SA, on their own random stream
        for e in EPS_GHB + (np.inf,):
            est[f"GHB-SA {e:g}"] = SA.estimate_v2(rng_sa, pub, train, n, np.zeros(P), e, DELTA, C=SA_C, iv=False)
        for e in BASE_EPS:
            rho_sa = LIT.rho_total(e, DELTA)
            for R in (3, 10, 30):
                for eta in (0.5, 2.0):
                    for E, thf in SA.fedavg_sa_ft(rng_sa, train, n, np.zeros(P), rho_sa, R, eta, steps=(0, 5, 30)).items():
                        est[f"FTSA{e:g}|R={R}|eta={eta}|E={E}"] = thf
    for e in BASE_EPS:
        rho, t = LIT.rho_total(e, DELTA), f"{e:g}"
        th, r = rel[e] if e in rel else ghb_fit(rng, train, n, pub, avg, e)[1:]
        # one-shot baselines on GHB's own releases at the same budget
        t2 = np.array([ghb.dersimonian_laird(th[None, :, d], r[None, :, d], ghb.Averager())[0, 0] for d in range(P)])
        w = 1.0 / (t2[None, :] + r)
        est[f"Meta-analysis (DP), eps={t}"] = np.repeat(((th * w).sum(0) / w.sum(0))[None], K, 0)
        centre = (th * (n / n.sum())[:, None]).sum(0)
        for lam in DITTO_GRID:
            est[f"DIT{t}|lambda={lam}"] = np.array([fit(X, y, centre, np.full(P, lam)) for X, y in train])
        for R in (1, 3, 10, 30):
            for eta in (0.2, 0.5, 2.0):
                g = LIT.fedavg(rng, train, n, np.zeros(P), rho, R, eta)
                est[f"AVG{t}|R={R}|eta={eta}"] = np.repeat(g[None], K, 0)
                for E, thf in LIT.fine_tune(train, np.repeat(g[None], K, 0), steps=(1, 5, 30)).items():
                    est[f"FT{t}|R={R}|eta={eta}|E={E}"] = thf
        for R in (3, 10, 30):
            for eta in (0.2, 0.5, 2.0):
                th0 = LIT.decentralized(rng, train, n, np.zeros(P), rho, R, eta, W)
                for E, thf in LIT.fine_tune(train, th0, steps=(0, 5, 30)).items():
                    est[f"DEC{t}|R={R}|eta={eta}|E={E}"] = thf
        for C in (2, 3):
            lab = LIT.private_clusters(rng, train, n, rho / 2, C)
            for R in (1, 3, 10):
                for eta in (0.2, 0.5, 2.0):
                    th0 = np.empty((K, P))
                    for c in range(C):
                        mem = np.flatnonzero(lab == c)
                        if len(mem):
                            th0[mem] = LIT.fedavg(rng, train, n, np.zeros(P), rho / 2, R, eta, mem)
                    for E, thf in LIT.fine_tune(train, th0, steps=(0, 5, 30)).items():
                        est[f"CLU{t}|C={C}|R={R}|eta={eta}|E={E}"] = thf
    return est


def run(item):
    i, (name, label) = item
    rng = np.random.default_rng([20, i])
    data, ids = load(name)
    K = len(data)
    n_all = np.array([len(y) for _, y in data])
    rng_sa = np.random.default_rng([21, i])
    rng_p = np.random.default_rng([22, i])
    pub, W = GL.Public(public_features(rng, name)), mixing(K)
    avg = ghb.Averager(W, 60)
    # per method: summed log-loss per hospital and split, and pooled test scores per split
    S = {}
    for rep in range(REPEATS):
        fs = [folds(rng, y, FOLDS) for _, y in data]
        for f in range(FOLDS):
            train, test = [], []
            for k, (X, y) in enumerate(data):
                tr, te = np.flatnonzero(fs[k] != f), np.flatnonzero(fs[k] == f)
                if len(tr) == 0:  # a hospital with fewer patients than folds sits this fold out of training
                    tr = te
                train.append((X[tr], y[tr])); test.append((X[te], y[te]))
            est = estimates(rng, train, pub, avg, W, rng_sa, rng_p)
            for m, e in est.items():
                rec = S.setdefault(m, dict(ll=np.zeros((REPEATS * FOLDS, K)), cnt=np.zeros((REPEATS * FOLDS, K)), s=[], y=[]))
                j = rep * FOLDS + f
                for k, (Xt, yt) in enumerate(test):
                    s = Xt @ e[k]
                    rec["ll"][j, k] = np.sum(np.logaddexp(0, s) - yt * s)
                    rec["cnt"][j, k] = len(yt)
                    rec["s"].append(s); rec["y"].append(yt)
    chosen = {}
    for fam, cands in TUNED.items():  # tuned on the held-out folds themselves, which favours the baselines
        best = min(cands, key=lambda c: S[c]["ll"].sum())
        S[fam], chosen[fam] = S[best], best
    small = n_all < np.median(n_all)
    big = n_all >= 20
    loc = S["Local"]

    def metrics(r, full=True):
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

    # every tuning candidate, for the realistic arm (settings chosen on the development network, then frozen)
    cand = {c: metrics(S[c], full=False) for cs in TUNED.values() for c in cs}
    res = {"chosen": chosen, "cand": cand, "info": dict(label=label, K=K, n=n_all.tolist(), ids=ids,
                                                       prevalence=float(np.mean(np.concatenate([y for _, y in data]))))}
    for m in METHODS:
        res[m] = metrics(S[m])
    print(label, K, chosen, {m: (round(o["rel"], 3), round(o["rel_small"], 3), round(o["worst"], 2))
                             for m, o in res.items() if m not in ("chosen", "cand", "info")}, flush=True)
    return name, res


FAM = [("GHB (DP, eps={e})", "GHB", style.TEAL700), ("GHB, tuned prior (DP), eps={e}", "GHB, tuned prior", "#2dd4bf"), ("DP-FedAvg, eps={e}", "DP-FedAvg", style.AMBER600),
       ("DP-FedAvg + fine-tuning, eps={e}", "DP-FedAvg + FT", "#92400e"), ("Decentralized DP-SGD, eps={e}", "Decentralized DP-SGD", "#be185d"),
       ("Clustered DP-FedAvg, eps={e}", "Clustered DP-FedAvg", style.SLATE800), ("Ditto-style (DP), eps={e}", "Ditto-style", style.BLUE600),
       ("Meta-analysis (DP), eps={e}", "Meta-analysis", "#a3a3a3")]
SHORTNET = {"medpar": "Medicare", "TCGA-LUSC": "LUSC", "TCGA-HNSC": "HNSC", "TCGA-BLCA": "BLCA", "TCGA-STAD": "STAD",
            "TCGA-LIHC": "LIHC", "TCGA-KIRC": "KIRC", "heart": "Heart"}


def figure(res):
    style.setup()
    names = [n for n, _ in NETWORKS if n in res]
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 4.6), gridspec_kw=dict(hspace=0.55))
    fig.subplots_adjust(bottom=0.14)
    x = np.arange(len(names))
    w = 0.8 / len(FAM)
    for ax, e in zip(axs, ("1", "10")):
        for i, (m, lab, c) in enumerate(FAM):
            v = np.array([res[nm][m.format(e=e)]["rel"] for nm in names])
            ax.bar(x + (i - len(FAM) / 2 + 0.5) * w, np.minimum(v, 1.4), w, color=c, label=lab, edgecolor="white", linewidth=0.3)
        lgd = [min(res[nm]["Local GD, tuned stopping"]["rel"], res[nm]["Local, tuned prior"]["rel"]) for nm in names]
        ax.scatter(x, lgd, marker="_", s=300, color=style.RED700, zorder=4, linewidths=1.4,
                   label="Best tuned local model (no other hospital)")
        ax.axhline(1, color=style.SLATE800, lw=0.6)
        ax.set_ylim(0.8, 1.4)
        ax.set_xticks(x, [f"{SHORTNET[nm]} (K={res[nm]['info']['K']})" for nm in names], fontsize=6.3)
        ax.set_ylabel("Held-out log-loss\nrelative to local", fontsize=6.5)
        ax.set_title(f"({'a' if e == '1' else 'b'}) Every private method at $\\epsilon$={e} (bars clipped at 1.4)", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.2)
    ghb.tidy(*axs)
    style.save(fig, ghb.FIGDIR / "real_networks.pdf")


CKPT = ghb.RESDIR / "real_networks_ckpt"


def main():
    """Runs the requested networks (all by default) in parallel. Each network's result is checkpointed to
    real_networks_ckpt/<name>.json the moment it finishes, and finished networks are skipped on a rerun, so an
    interrupted run resumes where it stopped (delete the checkpoint folder to start over). The final file
    real_networks.json merges the checkpoints with any earlier stored networks."""
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    CKPT.mkdir(exist_ok=True)
    todo = [(i, nw) for i, nw in enumerate(NETWORKS) if (not only or nw[0] in only) and not (CKPT / f"{nw[0]}.json").exists()]
    print("to run:", [nw[0] for _, nw in todo], flush=True)
    with Pool(4) as pool:
        for name, res in pool.imap_unordered(run, todo):
            json.dump(res, open(CKPT / f"{name}.json", "w"))
            print("checkpointed", name, flush=True)
    out = ghb.RESDIR / "real_networks.json"
    merged = json.load(open(out))["res"] if out.exists() else {}
    for nw, _ in NETWORKS:
        if (CKPT / f"{nw}.json").exists():
            merged[nw] = json.load(open(CKPT / f"{nw}.json"))
    res = {nw: merged[nw] for nw, _ in NETWORKS if nw in merged}
    json.dump(dict(repeats=REPEATS, folds=FOLDS, min_site=MIN_SITE, public=dict(medpar=MEDPAR_PUBLIC, tcga=TCGA_PUBLIC), res=res),
              open(out, "w"), indent=1)
    figure(res)


if __name__ == "__main__":
    if "--redraw" in sys.argv:
        figure(json.load(open(ghb.RESDIR / "real_networks.json"))["res"])
    else:
        main()
