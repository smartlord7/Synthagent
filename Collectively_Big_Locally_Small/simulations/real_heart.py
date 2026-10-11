"""Real data 1: UCI Heart Disease, four hospitals (Detrano et al. 1989; CC BY 4.0).

Cleveland Clinic (303 patients), Hungarian Institute of Cardiology, Budapest (294), University Hospitals of Zurich
and Basel (123) and the VA Medical Center, Long Beach (200). Outcome: angiographic coronary disease (num > 0).
Prevalence differs from 36% to 93% across sites, so the heterogeneity is real, not simulated.

Model: the pretest-probability model of coronary disease, logistic in age, sex and chest-pain type, the predictors
that every site records completely: x = (1, age, male, asymptomatic, typical or atypical angina), with non-anginal
pain as reference and age scaled by public constants, (age - 55) / 25, clipped to [-1, 1], so ||x|| <= 2 <= sqrt(5)
and the release sensitivity of the simulations holds without fitting anything to the data.

Evaluation: within each hospital, repeated stratified 5-fold cross-validation (20 repeats, 100 splits). Each hospital
trains on its training folds, optionally subsampled to at most n_max patients (stratified), to emulate siloed
smallness, and is tested on its own held-out fold. Metrics per hospital: log-loss, Brier score and AUC on held-out
patients, averaged over splits.

Methods (all start from 0 and share the weakly informative default prior N(0, 2.5^2) where they use a prior):
  Local                         MAP on the hospital's own training data;
  GHB (DP), epsilon = 1         ghb_logit.py without an anchor (no published model is assumed): the score release
                                at the public point 0, gossip on the ring of four hospitals (T = 60); the second
                                linearization point is regularized by the default prior N(0, 2.5^2) in place of an
                                anchor, which the final fit does not use;
  GHB (no DP)                   the same without noise;
  Global (pooled, no DP)        one MAP model on the pooled training data;
  GHB (DP), epsilon = 5         the same at a looser budget;
  DP-FedAvg + fine-tuning       epsilon = 1; rounds R in {1, 3, 10, 30}, step in {0.2, 0.5, 2}, fine-tuning steps
                                E in {0, 1, 5, 30};
  Ditto-style (DP)              proximal fit toward one-shot federated averaging of GHB's releases (epsilon = 1),
                                lambda in {0.01, 0.1, 0.3, 1, 3, 10, 30, 100}.
  Local GD, tuned stopping      full-batch gradient descent on the hospital's own data from 0, step 1, stopped
                                after E in {1, 2, 3, 5, 10, 30, 100} steps: a local-only control for the
                                regularization that early stopping gives the fine-tuned baseline.
  As in Simulation 15, the hyperparameters of the baselines are chosen on the held-out folds themselves, separately
  for each training size, which favours them; GHB has none.
The public feature distribution needed by GHB's moment inversion is not taken from the data: it is a nominal
chest-pain clinic population, age N(55, 9^2), 75% male, chest pain asymptomatic 50%, anginal 30%, non-anginal 20%,
independent. Its mismatch with each hospital is a real case-mix shift. The variant "GHB, empirical features (no DP)"
replaces it by the pooled training features, a non-private diagnostic of what that mismatch costs.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import ghb
import ghb_logit as GL
import literature as LIT
import style
from exp7_logistic import P, auc, fit

DATA = Path(__file__).resolve().parents[1] / "data" / "heart_disease"
SITES = [("cleveland", "Cleveland"), ("hungarian", "Budapest"), ("switzerland", "Zurich/Basel"), ("va", "Long Beach VA")]
COLS = "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal num".split()
NMAX = [20, 40, 80, None]
REPEATS, FOLDS, EPS, DELTA = 20, 5, 1.0, 1e-5
PRIOR = np.full(P, 1 / 2.5 ** 2)
METHODS = ["Local", "GHB (DP)", "GHB (DP, eps=5)", "GHB (no DP)", "GHB, empirical features (no DP)", "Global (pooled, no DP)",
           "Local GD, tuned stopping", "DP-FedAvg + fine-tuning", "Ditto-style (DP)"]
FT_GRID = [(R, eta, E) for R in (1, 3, 10, 30) for eta in (0.2, 0.5, 2.0) for E in (0, 1, 5, 30)]
DITTO_GRID = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TUNED = {"Local GD, tuned stopping": [f"LGD|E={E}" for E in (1, 2, 3, 5, 10, 30, 100)],
         "DP-FedAvg + fine-tuning": [f"FT|R={R}|eta={eta}|E={E}" for R, eta, E in FT_GRID],
         "Ditto-style (DP)": [f"Ditto|lambda={lam}" for lam in DITTO_GRID]}


def encode(age, sex, cp):
    a = np.clip((np.asarray(age, float) - 55) / 25, -1, 1)
    return np.column_stack([np.ones(len(a)), a, sex, cp == 4, (cp == 1) | (cp == 2)]).astype(float)


def load():
    out = []
    for f, _ in SITES:
        d = np.genfromtxt(DATA / f"processed.{f}.data", delimiter=",", missing_values="?", filling_values=np.nan)
        out.append((encode(d[:, 0], d[:, 1], d[:, 2]), (d[:, -1] > 0).astype(float)))
    return out


def public_features(rng, m=20000):
    age = rng.normal(55, 9, m)
    sex = (rng.random(m) < 0.75).astype(float)
    cp = rng.choice([4, 1, 3], m, p=[0.5, 0.3, 0.2])
    return encode(age, sex, cp)


def folds(rng, y, k):
    idx = np.empty(len(y), int)
    for c in (0, 1):
        sel = rng.permutation(np.flatnonzero(y == c))
        idx[sel] = np.arange(len(sel)) % k
    return idx


def subsample(rng, y, tr, nmax):
    if nmax is None or len(tr) <= nmax:
        return tr
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    npos = int(round(nmax * len(pos) / len(tr)))
    npos = min(max(npos, 1), len(pos), nmax - 1)
    return np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nmax - npos, replace=False)])


def ghb_fit(rng, train, n, pub, avg, eps):
    mu0 = np.zeros(P)
    g, sd = GL.release(rng, train, n, mu0, eps, DELTA)
    th, r, drop = GL.invert(pub, g, sd, n, mu0, avg, anchor_sd=GL.PRIOR_SD, check=False)  # linearize under the default prior
    return GL.estimate(train, th, r, drop, mu0, avg, anchor_sd=None), th


def methods(rng, train, pub_nominal, avg):
    n = np.array([len(y) for _, y in train], float)
    est = {"Local": np.array([fit(X, y, np.zeros(P), PRIOR) for X, y in train])}
    est["GHB (DP)"], th_dp = ghb_fit(rng, train, n, pub_nominal, avg, EPS)
    est["GHB (no DP)"], _ = ghb_fit(rng, train, n, pub_nominal, avg, np.inf)
    est["GHB, empirical features (no DP)"], _ = ghb_fit(rng, train, n, GL.Public(np.vstack([X for X, _ in train])), avg, np.inf)
    Xa = np.vstack([X for X, _ in train]); ya = np.concatenate([y for _, y in train])
    est["Global (pooled, no DP)"] = np.repeat(fit(Xa, ya, np.zeros(P), PRIOR)[None], len(n), 0)
    est["GHB (DP, eps=5)"], _ = ghb_fit(rng, train, n, pub_nominal, avg, 5.0)
    rho = LIT.rho_total(EPS, DELTA)
    for R in (1, 3, 10, 30):
        for eta in (0.2, 0.5, 2.0):
            g = LIT.fedavg(rng, train, n, np.zeros(P), rho, R, eta)
            for E, th in LIT.fine_tune(train, np.repeat(g[None], len(n), 0), steps=(0, 1, 5, 30)).items():
                est[f"FT|R={R}|eta={eta}|E={E}"] = th
    for E, th in LIT.fine_tune(train, np.zeros((len(n), P)), steps=(1, 2, 3, 5, 10, 30, 100)).items():
        est[f"LGD|E={E}"] = th
    fedavg = (th_dp * (n / n.sum())[:, None]).sum(0)
    for lam in DITTO_GRID:
        est[f"Ditto|lambda={lam}"] = np.array([fit(X, y, fedavg, np.full(P, lam)) for X, y in train])
    return est


def run(nmax):
    rng = np.random.default_rng([19, 0 if nmax is None else nmax])
    data = load()
    K = len(data)
    pub = GL.Public(public_features(rng))
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies(K)["Ring (degree 2)"]), 60)
    acc = {}
    for rep in range(REPEATS):
        fs = [folds(rng, y, FOLDS) for _, y in data]
        for f in range(FOLDS):
            train, test = [], []
            for k, (X, y) in enumerate(data):
                tr = subsample(rng, y, np.flatnonzero(fs[k] != f), nmax)
                te = np.flatnonzero(fs[k] == f)
                train.append((X[tr], y[tr])); test.append((X[te], y[te]))
            est = methods(rng, train, pub, avg)
            for m, e in est.items():
                acc.setdefault(m, {k: dict(ll=[], brier=[], auc=[]) for k in range(K)})
                for k, (Xt, yt) in enumerate(test):
                    s = Xt @ e[k]
                    p = expit(s)
                    acc[m][k]["ll"].append(float(np.mean(np.logaddexp(0, s) - yt * s)))
                    acc[m][k]["brier"].append(float(np.mean((p - yt) ** 2)))
                    if 0 < yt.sum() < len(yt):
                        acc[m][k]["auc"].append(auc(s, yt))
    chosen = {}
    for fam, cands in TUNED.items():  # tuned on the held-out folds themselves, which favours the baselines
        best = min(cands, key=lambda c: np.mean([np.mean(acc[c][k]["ll"]) for k in range(K)]))
        acc[fam], chosen[fam] = acc[best], best
    res = {"chosen": chosen}
    loc = np.array([np.mean(acc["Local"][k]["ll"]) for k in range(K)])
    for m in METHODS:
        ll = np.array([acc[m][k]["ll"] for k in range(K)])          # (K, splits)
        res[m] = dict(
            ll=[float(v) for v in ll.mean(1)], ll_se=[float(v) for v in ll.std(1, ddof=1) / np.sqrt(ll.shape[1])],
            brier=[float(np.mean(acc[m][k]["brier"])) for k in range(K)],
            auc=[float(np.mean(acc[m][k]["auc"])) for k in range(K)],
            ll_mean=float(ll.mean()), ll_mean_se=float(ll.mean(0).std(ddof=1) / np.sqrt(ll.shape[1])),
            rel=float(ll.mean() / loc.mean()), worst=float((ll.mean(1) / loc).max()),
            rel_paired_se=float((ll.mean(0) / np.array([acc["Local"][k]["ll"] for k in range(K)]).mean(0)).std(ddof=1)
                                / np.sqrt(ll.shape[1])))
    tag = "all" if nmax is None else str(nmax)
    print(tag, chosen, {m: (round(r["ll_mean"], 4), round(r["rel"], 3), round(r["worst"], 2)) for m, r in res.items() if m != "chosen"}, flush=True)
    return tag, res


SHORT = {"Local": "Local", "GHB (DP)": r"GHB, $\epsilon$=1", "GHB (DP, eps=5)": r"GHB, $\epsilon$=5", "GHB (no DP)": "GHB, no DP",
         "GHB, empirical features (no DP)": "GHB, empirical features", "Global (pooled, no DP)": "Global (no DP)", "Local GD, tuned stopping": "Local GD, tuned",
         "DP-FedAvg + fine-tuning": "DP-FedAvg + FT", "Ditto-style (DP)": "Ditto-style"}
COL = {"Local": style.GRAY400, "GHB (DP)": style.TEAL700, "GHB (DP, eps=5)": "#0d9488", "GHB (no DP)": "#14b8a6", "GHB, empirical features (no DP)": "#99c2bd",
       "Global (pooled, no DP)": style.RED700, "Local GD, tuned stopping": "#9ca3af", "DP-FedAvg + fine-tuning": "#92400e", "Ditto-style (DP)": style.BLUE600}
MK = {"Local": "o", "GHB (DP)": "o", "GHB (DP, eps=5)": "^", "GHB (no DP)": "s", "GHB, empirical features (no DP)": "D", "Global (pooled, no DP)": "x", "Local GD, tuned stopping": "v",
      "DP-FedAvg + fine-tuning": ">", "Ditto-style (DP)": "D"}


def figure(res):
    style.setup()
    tags = ["20", "40", "80", "all"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.3]))
    fig.subplots_adjust(bottom=0.3)
    a = axs[0]
    for m in METHODS:
        a.errorbar(range(4), [res[t][m]["rel"] for t in tags], yerr=[2 * res[t][m]["rel_paired_se"] for t in tags],
                   color=COL[m], marker=MK[m], ms=3, lw=1.1, elinewidth=0.6, ls="--" if "no DP" in m else "-", label=SHORT[m])
    a.axhline(1, color=style.SLATE800, lw=0.6)
    a.set_ylim(0.9, 1.15)
    off = [f"{res[t]['Ditto-style (DP)']['rel']:.2f}" for t in tags if res[t]["Ditto-style (DP)"]["rel"] > 1.15]
    if off:
        a.text(0.02, 0.97, "Ditto-style off scale at 20 and 40: " + ", ".join(off), transform=a.transAxes, fontsize=5.5,
               va="top", color=style.BLUE600)
    a.set_xticks(range(4), ["20", "40", "80", "all"])
    a.set_xlabel("Training patients per hospital (at most)")
    a.set_ylabel("Held-out log-loss\nrelative to local")
    a.set_title("(a) Mean over the four hospitals", fontsize=7.5)
    b = axs[1]
    x = np.arange(len(SITES))
    w = 0.8 / len(METHODS)
    for i, m in enumerate(METHODS):
        loc = np.array(res["40"]["Local"]["ll"])
        b.bar(x + (i - len(METHODS) / 2 + 0.5) * w, np.array(res["40"][m]["ll"]) / loc, w, color=COL[m], edgecolor="white", linewidth=0.3)
    b.axhline(1, color=style.SLATE800, lw=0.6)
    b.set_ylim(0.8, 1.3)
    b.set_xticks(x, ["Cleveland", "Budapest", "Zurich/Basel", "Long Beach"], fontsize=6.5)
    b.set_ylabel("Held-out log-loss\nrelative to local")
    b.set_title("(b) Each hospital, at most 40 training patients", fontsize=7.5)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.2)
    ghb.tidy(*axs)
    style.save(fig, ghb.FIGDIR / "real_heart.pdf")


def main():
    with Pool(4) as pool:
        res = dict(pool.map(run, NMAX))
    data = load()
    info = dict(sites=[s for _, s in SITES], n=[len(y) for _, y in data], prevalence=[float(y.mean()) for _, y in data])
    json.dump(dict(repeats=REPEATS, folds=FOLDS, eps=EPS, info=info, res=res), open(ghb.RESDIR / "real_heart.json", "w"), indent=1)
    figure(res)


if __name__ == "__main__":
    import sys
    if "--redraw" in sys.argv:
        figure(json.load(open(ghb.RESDIR / "real_heart.json"))["res"])
    else:
        main()
