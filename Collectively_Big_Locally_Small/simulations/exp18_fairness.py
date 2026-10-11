"""Simulation 18: group fairness. What does borrowing do to how a hospital's model treats a protected group?

The logistic world of Simulation 7 with one feature replaced by a binary protected attribute S (S = 1: the
minority group). Features x = (1, S, z_1, z_2, z_3) with z ~ N(0, I) clipped to ||z|| <= sqrt(3), so that
||x|| <= sqrt(5) as before. Hospital k's share of the minority group is pi_k ~ Beta, so the group is rare at some
hospitals (the sensitive-attribute heterogeneity of FAIR-FATE's experiments). theta_k = mu + u_k; the coefficient
of S is the hospital's true disparity in outcome risk (log-odds), with its own between-hospital variance tau_S^2.

Scenarios (epsilon = 1, delta = 1e-5, 24 hospitals, ring with two-hop links, T = 60):
  shared      tau^2 = tau_S^2 = 0.05, minority share 0.3 on average;
  varies      tau_S^2 = 0.5: hospitals differ much more in disparity than in anything else;
  rare        minority share 0.1 on average (tau^2 = tau_S^2 = 0.05);
  opposite    mu_S = 0 and tau_S^2 = 0.5: no disparity on average, but large disparities of either sign.

Methods:
  Local                       MAP with prior N(mu_0, I) (no other hospital);
  GHB (DP)                    ghb_logit.py, as reported;
  GHB, S local (DP)           GHB, but the coefficient of S keeps the local prior N(mu_0,S, 1): the hospital does not
                              borrow its disparity (same release, same privacy);
  Global (pooled, no DP)      one model on the pooled data of all hospitals;
  FedAvg (no DP)              FAIR-FATE's training protocol with lambda = 0;
  FAIR-FATE (no DP), F=SP/EQO Algorithm 1 of Salazar et al. (ICCS 2023): each round a third of the clients train
                              locally (E = 10 epochs, mini-batches of 10, learning rate 0.01); the server keeps the
                              updates whose fairness F on its validation set is at least that of the global model,
                              weights them by F, and applies decaying momentum (beta_0 = 0.9) and an exponentially
                              growing weight lambda_t = min(lambda_0 (1 + rho)^t, MAX) (lambda_0 = 0.5, rho = 0.05,
                              MAX = 0.9), for T = 100 rounds. As in the paper, the server holds a labelled validation
                              set of 20% of all patients, drawn from the hospitals' own distributions, and nothing is
                              privatized. All federated models start from the published anchor mu_0.

Metrics per hospital, on 4,000 new patients from its own distribution: excess log-loss over the true model for
each group; the error of the estimated disparity |theta^_S - theta_S|; the calibration bias in the minority group,
mean(p^ - p | S = 1). On the pooled test patients of all hospitals, each scored by the model its hospital uses:
statistical parity (SP), equal opportunity (EO) and equalized odds (EQO) ratios at threshold 0.5, as in FAIR-FATE
(1 = parity), with the true models as a reference.
"""
import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import ghb
import ghb_logit as GL
import style
from exp7_logistic import P, fit

REPS, N_TEST, EPS, DELTA = 100, 4000, 1.0, 1e-5
MU = np.array([-0.5, 0.5, 0.8, -0.6, 0.4])  # intercept, S, z_1..z_3
S = 1                                      # index of the protected attribute
CONC = 6.0                                 # concentration of the Beta distribution of minority shares
SCENARIOS = {
    "shared": dict(tau2=0.05, tauS2=0.05, share=0.3, muS=0.5),
    "varies": dict(tau2=0.05, tauS2=0.5, share=0.3, muS=0.5),
    "rare": dict(tau2=0.05, tauS2=0.05, share=0.1, muS=0.5),
    "opposite": dict(tau2=0.05, tauS2=0.5, share=0.3, muS=0.0),
}
FF = dict(lam0=0.5, rho=0.05, MAX=0.9, beta0=0.9, T=100, E=10, B=10, lr=0.01, frac=1 / 3, val=0.2)
METHODS = ["Local", "GHB (DP)", "GHB, S local (DP)", "Global (pooled, no DP)", "FedAvg (no DP)",
           "FAIR-FATE (no DP), F=SP", "FAIR-FATE (no DP), F=EQO"]


def feats(rng, m, pi):
    s = rng.random(m) < pi
    z = rng.standard_normal((m, 3))
    z = z * np.minimum(1.0, np.sqrt(3.0) / np.linalg.norm(z, axis=1, keepdims=True))
    return np.column_stack([np.ones(m), s.astype(float), z])


def world(rng, n, sc):
    K = len(n)
    mu = MU.copy()
    mu[S] = sc["muS"]
    sd = np.full(P, np.sqrt(sc["tau2"]))
    sd[S] = np.sqrt(sc["tauS2"])
    theta = mu + sd * rng.standard_normal((K, P))
    a = sc["share"] * CONC
    pi = np.clip(rng.beta(a, CONC - a, K), 0.02, 0.98)
    mu0 = mu + 0.3 * rng.standard_normal(P)
    data = []
    for k in range(K):
        X = feats(rng, int(n[k]), pi[k])
        data.append((X, rng.binomial(1, expit(X @ theta[k])).astype(float)))
    return theta, pi, mu0, data


def ratio(a, b):
    if a == b:
        return 1.0
    return min(a, b) / max(a, b)


def fairness(s, yhat, y):
    """SP, EO, EQO ratios (1 = parity) of decisions yhat for groups s, true labels y."""
    g0, g1 = s == 0, s == 1
    rate = lambda sel: yhat[sel].mean() if sel.any() else 0.0
    sp = ratio(rate(g0), rate(g1))
    eo = ratio(rate(g0 & (y == 1)), rate(g1 & (y == 1)))
    fp = ratio(rate(g0 & (y == 0)), rate(g1 & (y == 0)))
    return dict(SP=sp, EO=eo, EQO=(eo + fp) / 2)


def local_sgd(rng, X, y, th, E, B, lr):
    th = th.copy()
    n = len(y)
    for _ in range(E):
        idx = rng.permutation(n)
        for i in range(0, n, B):
            b = idx[i:i + B]
            th -= lr * X[b].T @ (expit(X[b] @ th) - y[b]) / len(b)
    return th


def fair_fate(rng, data, n, init, val, F=None, p=FF):
    """FAIR-FATE (Algorithm 1 of Salazar et al. 2023); F=None gives FedAvg with the same protocol (lambda = 0)."""
    Xv, yv = val
    K = len(data)
    th, v = np.array(init, float), np.zeros(P)
    m = max(1, int(round(p["frac"] * K)))
    for t in range(p["T"]):
        sel = rng.choice(K, m, replace=False)
        locs = {k: local_sgd(rng, *data[k], th, p["E"], p["B"], p["lr"]) for k in sel}
        w = n[sel] / n[sel].sum()
        aN = sum(w[i] * (locs[k] - th) for i, k in enumerate(sel))
        if F is None:
            th = th + aN
            continue
        f_glob = fairness(Xv[:, S], (Xv @ th > 0).astype(float), yv)[F]
        fk = {k: fairness(Xv[:, S], (Xv @ locs[k] > 0).astype(float), yv)[F] for k in sel}
        fair = [k for k in sel if fk[k] >= f_glob]
        tot = sum(fk[k] for k in fair)
        aF = sum(fk[k] / tot * (locs[k] - th) for k in fair) if fair and tot > 0 else np.zeros(P)
        frac = 1 - t / p["T"]
        beta = p["beta0"] * frac / ((1 - p["beta0"]) + p["beta0"] * frac)
        v = beta * v + (1 - beta) * aF
        lam = min(p["lam0"] * (1 + p["rho"]) ** t, p["MAX"])
        th = th + lam * v + (1 - lam) * aN
    return th


def ghb_estimates(rng, data, n, mu0, pub, avg):
    g, sd = GL.release(rng, data, n, mu0, EPS, DELTA)
    th, r, drop = GL.invert(pub, g, sd, n, mu0, avg)
    full = GL.estimate(data, th, r, drop, mu0, avg)
    t2 = GL.tau2_quantiles(th, r)
    comps = [GL.anchor(*GL.network(th, r, avg, t2[q]), mu0, 0.3, drop) + (t2[q],) for q in range(len(t2))]
    own = []
    for k, (X, y) in enumerate(data):
        means, variances = [], []
        for c in comps:
            m_, v_ = c[0][k].copy(), c[1][k] + c[2]
            m_[S], v_[S] = mu0[S], 1.0  # the hospital's own prior for its disparity, as in local estimation
            means.append(m_); variances.append(v_)
        own.append(GL.combine(X, y, means, variances))
    return full, np.array(own)


def one_rep(rng, n, sc, avg):
    theta, pi, mu0, data = world(rng, n, sc)
    pub = GL.Public(feats(rng, 20000, sc["share"]))
    est = {"Local": np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])}
    est["GHB (DP)"], est["GHB, S local (DP)"] = ghb_estimates(rng, data, n, mu0, pub, avg)
    Xall = np.vstack([X for X, _ in data]); yall = np.concatenate([y for _, y in data])
    est["Global (pooled, no DP)"] = np.repeat(fit(Xall, yall, np.zeros(P), np.full(P, 1e-3))[None], len(n), 0)
    # the server's validation set: 20% of all patients, i.e. a quarter as many as the training data, per hospital
    vs = [feats(rng, max(1, int(round(n[k] * FF["val"] / (1 - FF["val"])))), pi[k]) for k in range(len(n))]
    Xv = np.vstack(vs)
    yv = np.concatenate([rng.binomial(1, expit(X @ theta[k])).astype(float) for k, X in enumerate(vs)])
    est["FedAvg (no DP)"] = np.repeat(fair_fate(rng, data, n, mu0, (Xv, yv))[None], len(n), 0)
    for F in ("SP", "EQO"):
        est[f"FAIR-FATE (no DP), F={F}"] = np.repeat(fair_fate(rng, data, n, mu0, (Xv, yv), F)[None], len(n), 0)
    # evaluation
    per = {m: dict(kl=[], kl1=[], kl0=[], dis=[], cal=[]) for m in est}
    pooled = {m: ([], [], []) for m in list(est) + ["True model"]}
    for k in range(len(n)):
        Xt = feats(rng, N_TEST, pi[k])
        pt = expit(Xt @ theta[k])
        yt = rng.binomial(1, pt).astype(float)
        g1 = Xt[:, S] == 1
        ent = -(pt * np.log(pt) + (1 - pt) * np.log1p(-pt))
        for m, e in est.items():
            sc_ = Xt @ e[k]
            ex = np.logaddexp(0, sc_) - pt * sc_ - ent
            per[m]["kl"].append(ex.mean())
            per[m]["kl1"].append(ex[g1].mean() if g1.any() else np.nan)
            per[m]["kl0"].append(ex[~g1].mean())
            per[m]["dis"].append(abs(e[k][S] - theta[k][S]))
            per[m]["cal"].append((expit(sc_[g1]) - pt[g1]).mean() if g1.any() else np.nan)
            pooled[m][0].append(Xt[:, S]); pooled[m][1].append((sc_ > 0).astype(float)); pooled[m][2].append(yt)
        pooled["True model"][0].append(Xt[:, S]); pooled["True model"][1].append((pt > 0.5).astype(float))
        pooled["True model"][2].append(yt)
    out = {m: {key: np.array(v) for key, v in d.items()} for m, d in per.items()}
    fair = {m: fairness(np.concatenate(a), np.concatenate(b), np.concatenate(c)) for m, (a, b, c) in pooled.items()}
    return out, fair, pi


def run(item):
    i, (name, sc) = item
    rng = np.random.default_rng([18, i])
    n = ghb.site_sizes()
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies(24)["Ring + 2-hop (degree 4)"]), 60)
    acc, fair = {}, {}
    for _ in range(REPS):
        out, f, _ = one_rep(rng, n, sc, avg)
        for m, d in out.items():
            for key, v in d.items():
                acc.setdefault(m, {}).setdefault(key, []).append(v)
        for m, d in f.items():
            for key, v in d.items():
                fair.setdefault(m, {}).setdefault(key, []).append(v)
    res = {}
    loc = np.array(acc["Local"]["kl"])  # (REPS, K)
    for m, d in acc.items():
        a = {key: np.array(v) for key, v in d.items()}
        rep_mean = lambda x: np.nanmean(x, axis=1)
        res[m] = dict(
            kl=float(a["kl"].mean()), kl_se=float(a["kl"].mean(1).std(ddof=1) / np.sqrt(REPS)),
            kl1=float(np.nanmean(a["kl1"])), kl1_se=float(rep_mean(a["kl1"]).std(ddof=1) / np.sqrt(REPS)),
            kl0=float(a["kl0"].mean()),
            dis=float(a["dis"].mean()), dis_se=float(a["dis"].mean(1).std(ddof=1) / np.sqrt(REPS)),
            cal=float(np.nanmean(np.abs(a["cal"]))),
            worst=float((a["kl"].mean(0) / loc.mean(0)).max()),
            worst1=float(np.nanmax(np.nanmean(a["kl1"], 0) / np.nanmean(np.array(acc["Local"]["kl1"]), 0))))
    for m, d in fair.items():
        res.setdefault(m, {}).update({key: float(np.mean(v)) for key, v in d.items()})
    print(name, {m: (round(1000 * r.get("kl", np.nan), 1), round(1000 * r.get("kl1", np.nan), 1),
                     round(r.get("dis", np.nan), 3), round(r["EQO"], 3)) for m, r in res.items()}, flush=True)
    return name, res


SHORT = {"Local": "Local", "GHB (DP)": "GHB", "GHB, S local (DP)": "GHB, $S$ local", "Global (pooled, no DP)": "Global*",
         "FedAvg (no DP)": "FedAvg*", "FAIR-FATE (no DP), F=SP": "FAIR-FATE SP*", "FAIR-FATE (no DP), F=EQO": "FAIR-FATE EQO*"}
COL = {"Local": style.GRAY400, "GHB (DP)": style.TEAL700, "GHB, S local (DP)": "#14b8a6",
       "Global (pooled, no DP)": style.RED700, "FedAvg (no DP)": style.AMBER600, "FAIR-FATE (no DP), F=SP": "#7c3aed",
       "FAIR-FATE (no DP), F=EQO": "#c4b5fd"}
TITLES = {"shared": "Shared disparity", "varies": "Disparity varies", "rare": "Rare minority", "opposite": "Opposite disparities"}


def figure(res):
    style.setup()
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw=dict(wspace=0.45))
    fig.subplots_adjust(bottom=0.24)
    names = list(SCENARIOS)
    x = np.arange(len(names))
    w = 0.8 / len(METHODS)
    panels = [("kl1", 1000, "Minority excess log-loss\n($10^{-3}$ nats)", "(a) Accuracy for the minority"),
              ("dis", 1, r"$|\hat\theta_S-\theta_S|$", "(b) Error in the disparity"),
              ("EQO", 1, "Equalized odds (1 = parity)", "(c) Network-wide fairness")]
    for ax, (key, f, yl, tt) in zip(axs, panels):
        for i, m in enumerate(METHODS):
            ax.bar(x + (i - len(METHODS) / 2 + 0.5) * w, [f * res[s][m][key] for s in names], w, color=COL[m],
                   label=SHORT[m], edgecolor="white", linewidth=0.3)
        if key == "EQO":
            for j, s in enumerate(names):
                ax.plot([j - 0.42, j + 0.42], [res[s]["True model"]["EQO"]] * 2, color=style.SLATE800, lw=0.9, ls="--")
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x, names, fontsize=6)
        ax.set_ylabel(yl, fontsize=6.5)
        ax.set_title(tt, fontsize=7)
        ax.tick_params(axis="y", labelsize=5.8)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=style.SLATE800, ls="--", lw=0.9)); l.append("True models")
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=5.8, handlelength=1.2, columnspacing=0.8)
    ghb.tidy(*axs)
    style.save(fig, ghb.FIGDIR / "sim18_fairness.pdf")


def main():
    with Pool(4) as pool:
        res = dict(pool.map(run, list(enumerate(SCENARIOS.items()))))
    json.dump(dict(reps=REPS, scenarios=SCENARIOS, fair_fate=FF, res=res),
              open(ghb.RESDIR / "sim18_fairness.json", "w"), indent=1)
    figure(res)


if __name__ == "__main__":
    import sys
    if "--redraw" in sys.argv:
        figure(json.load(open(ghb.RESDIR / "sim18_fairness.json"))["res"])
    else:
        main()
