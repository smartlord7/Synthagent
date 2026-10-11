"""S7: end to end with logistic-regression heads (layer L2 of Figure 10), beyond the Gaussian model.

24 hospitals of the sizes used in S2. Each hospital has its own logistic model on 4 standardized
features plus an intercept, theta_k = mu + u_k with u_k ~ N(0, tau^2 I). Features are clipped so that
||x|| <= sqrt(5), which bounds the sensitivity of the released estimate.

Release (requirement R2): each hospital releases its mean score at the published anchor mu_0, with
Gaussian noise calibrated by zero-concentrated DP to the sensitivity 2 sqrt(5)/n; the releases are inverted
to the parameter scale by public moment matching (ghb_logit.py). mu_0 is a published model (channel P3),
equal to mu plus N(0, 0.3^2 I).

Methods, evaluated on 4,000 fresh patients per hospital:
  Local       MAP with prior N(mu_0, I): uses the published anchor but no other hospital;
  Global      one logistic model fitted on the pooled data of all hospitals (no privacy; generous);
  GHB         ghb_logit.py: coordinatewise borrowing by gossip on the ring+2-hop graph (T = 60 rounds), the
              network estimate combined with mu_0 unless they conflict (layer L3), and each hospital's
              posterior mean with tau integrated out over its posterior (half-Cauchy(0.5) prior, Remark 8.3).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import ghb
import style
from blueprint import dp_noise_var

P, R_X, LAM, EPS, T, N_TEST, REPS = 5, np.sqrt(5.0), 0.2, 1.0, 60, 4000, 100
MU = np.array([-1.0, 0.8, -0.6, 0.4, 0.2])
TAU2S = [0.01, 0.05, 0.2, 0.5]
TAU2_MAIN = 0.05
ANCHOR_SD = 0.3
TAU2_EST = "bayes"  # half-Cauchy(0.5) posterior mean of tau^2 (Remark 8.3)


def features(rng, m):
    z = rng.standard_normal((m, P - 1))
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z * np.minimum(1.0, 2.0 / nrm)
    return np.hstack([np.ones((m, 1)), z])


def fit(X, y, prior_mean, prior_prec, iters=100):
    """MAP of logistic regression with Gaussian prior N(prior_mean, diag(1/prior_prec)), by damped Newton."""
    def obj(t):
        s = X @ t
        return np.sum(np.logaddexp(0, s) - y * s) + 0.5 * np.sum(prior_prec * (t - prior_mean) ** 2)

    th = prior_mean.copy()
    f = obj(th)
    for _ in range(iters):
        p = expit(X @ th)
        g = X.T @ (p - y) + prior_prec * (th - prior_mean)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prior_prec)
        step = np.linalg.solve(H, g)
        a = 1.0
        while a > 1e-8:
            f_new = obj(th - a * step)
            if f_new <= f - 1e-4 * a * g @ step:
                break
            a /= 2
        th, f = th - a * step, f_new
        if np.abs(a * step).max() < 1e-9:
            break
    return th


def logloss(X, y, th):
    s = X @ th
    return float(np.mean(np.logaddexp(0, s) - y * s))


def auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fisher_info(rng, mu0, m=20000):
    """Per-patient Fisher information at the anchor under the public feature distribution."""
    X = features(rng, m)
    p = expit(X @ mu0)
    return (X * (p * (1 - p))[:, None]).T @ X / len(X)


def one_rep(rng, n, tau2, eps, avg):
    import ghb_logit as GL  # imported here: ghb_logit imports fit and P from this module
    K = len(n)
    theta = MU + np.sqrt(tau2) * rng.standard_normal((K, P))
    mu0 = MU + ANCHOR_SD * rng.standard_normal(P)
    pub = GL.Public(features(rng, 20000))
    data = []
    for k in range(K):
        X = features(rng, int(n[k]))
        data.append((X, rng.binomial(1, expit(X @ theta[k])).astype(float)))
    # GHB (ghb_logit.py): privatized scores at the anchor, inverted by public moment matching, tau integrated out
    g, sd = GL.release(rng, data, n, mu0, eps)
    th, r, drop = GL.invert(pub, g, sd, n, mu0, avg, ANCHOR_SD)
    ghb_est = GL.estimate(data, th, r, drop, mu0, avg, ANCHOR_SD)
    Xall = np.vstack([X for X, _ in data]); yall = np.concatenate([y for _, y in data])
    glob = fit(Xall, yall, np.zeros(P), np.full(P, 1e-3))
    out = {m: dict(kl=[], auc=[], mse=[]) for m in ("Local", "Global", "GHB", "Oracle")}
    for k, (X, y) in enumerate(data):
        ests = {"Local": fit(X, y, mu0, np.ones(P)),
                "Global": glob,
                "GHB": ghb_est[k],
                "Oracle": theta[k]}
        Xt = features(rng, N_TEST)
        pt = expit(Xt @ theta[k])
        yt = rng.binomial(1, pt).astype(float)
        bayes = float(np.mean(-(pt * np.log(pt) + (1 - pt) * np.log1p(-pt))))
        for m, th in ests.items():
            s = Xt @ th
            exp_loss = float(np.mean(np.logaddexp(0, s) - pt * s))  # expected log-loss given x
            out[m]["kl"].append(exp_loss - bayes)
            out[m]["auc"].append(auc(s, yt))
            out[m]["mse"].append(float(np.sum((th - theta[k]) ** 2)))
    return out


def run(rng, n, tau2, eps, avg, reps=REPS):
    acc = None
    for _ in range(reps):
        o = one_rep(rng, n, tau2, eps, avg)
        if acc is None:
            acc = {m: {q: [] for q in v} for m, v in o.items()}
        for m, v in o.items():
            for q, arr in v.items():
                acc[m][q].append(arr)
    return {m: {q: np.array(a) for q, a in v.items()} for m, v in acc.items()}  # (reps, K)


EPS_GRID = [0.3, 1.0, 3.0, 10.0, None]


def summarize(res):
    loc = res["Local"]["kl"].mean(axis=0)
    out = {m: dict(kl=float(v["kl"].mean()), auc=float(v["auc"].mean())) for m, v in res.items()}
    for m in ("Global", "GHB"):
        ratio = res[m]["kl"].mean(axis=0) / loc
        out[m].update(kl_mean_ratio=float(ratio.mean()), kl_worst_ratio=float(ratio.max()))
    return out


def main():
    rng = np.random.default_rng(7)
    n = ghb.site_sizes()
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), T)
    sweep = {eps: {t: run(rng, n, t, eps, avg) for t in TAU2S} for eps in (EPS, None)}
    by_eps = {eps: (sweep[eps][TAU2_MAIN] if eps in sweep else run(rng, n, TAU2_MAIN, eps, avg)) for eps in EPS_GRID}
    table = {f"tau2={t}|eps={e}": summarize(sweep[e][t]) for e in sweep for t in TAU2S}
    table.update({f"tau2={TAU2_MAIN}|eps={e}": summarize(r) for e, r in by_eps.items()})
    for k, v in table.items():
        print(k, "  ".join(f"{m}: KL {d['kl']:.4f} AUC {d['auc']:.3f}" for m, d in v.items()))

    style.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(wspace=0.32))
    for eps, mk, lab in [(EPS, "o", r"GHB, $\epsilon$=1"), (None, "^", "GHB, no DP")]:
        r = sweep[eps][TAU2_MAIN]
        a.plot(n, r["GHB"]["kl"].mean(0) / r["Local"]["kl"].mean(0), mk, ms=3,
               color=style.TEAL700 if eps else style.BLUE600, mfc="white", mew=0.8, label=lab)
    r = sweep[EPS][TAU2_MAIN]
    a.plot(n, r["Global"]["kl"].mean(0) / r["Local"]["kl"].mean(0), "s", ms=3, color=style.RED700,
           mfc="white", mew=0.8, label="Global model (pooled, no DP)")
    a.axhline(1, color=style.SLATE800, lw=0.6)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("Hospital size $n_k$"); a.set_ylabel("Excess log-loss relative to local")
    a.set_title(rf"Each hospital, $\tau^2$={TAU2_MAIN}", fontsize=7.5)
    a.legend(frameon=False, fontsize=6.3, loc="upper left")
    for m, eps, c, ls, lab in [("Local", EPS, style.GRAY400, "-", "Local (anchor prior only)"),
                               ("Global", EPS, style.RED700, "-", "Global model"),
                               ("GHB", EPS, style.TEAL700, "-", r"GHB, $\epsilon$=1"),
                               ("GHB", None, style.BLUE600, "--", "GHB, no DP")]:
        b.plot(TAU2S, [sweep[eps][t][m]["kl"].mean() for t in TAU2S], ls, marker="o", ms=3, color=c, lw=1.3,
               label=lab)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"Heterogeneity $\tau^2$ (per coefficient)"); b.set_ylabel("Mean excess log-loss (nats)")
    b.set_title("Mean over hospitals and replicates", fontsize=7.5)
    b.legend(frameon=False, fontsize=6.3)
    ghb.tidy(a, b)
    ghb.tidy(a, b)
    style.save(fig, ghb.FIGDIR / "s7_logistic.pdf")
    json.dump(dict(n=n.tolist(), tau2=TAU2S, eps_grid=[str(e) for e in EPS_GRID], table=table),
              open(ghb.RESDIR / "s7_logistic.json", "w"), indent=1)


if __name__ == "__main__":
    main()
