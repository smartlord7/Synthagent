"""S7: end to end with logistic-regression heads (layer L2 of Figure 10), beyond the Gaussian model.

24 hospitals of the sizes used in S2. Each hospital has its own logistic model on 4 standardized
features plus an intercept, theta_k = mu + u_k with u_k ~ N(0, tau^2 I). Features are clipped so that
||x|| <= sqrt(5), which bounds the sensitivity of the released estimate.

Release (requirement R2): each hospital fits the ridge objective centred on a fixed public reference c = 0
    (1/n) sum_i loss_i(theta) + (lambda/2) ||theta - c||^2,  lambda = 0.2,
whose minimizer has L2 sensitivity 2 sqrt(5) / (n lambda) (Chaudhuri et al., 2011), and releases it
with isotropic Gaussian noise. mu_0 is a published model (channel P3), equal to mu plus N(0, 0.3^2 I).
Anyone can then undo the ridge shrinkage to first order, theta = c + (I + lambda I) I^{-1} (release - c),
with I the per-patient Fisher information at c under the public feature distribution; this is
post-processing. The variance of each de-shrunk release is public: [I^{-1}]_dd / n_j plus the
correspondingly amplified Gaussian-mechanism noise.

Methods, evaluated on 4,000 fresh patients per hospital:
  Local       MAP with prior N(mu_0, I): uses the published anchor but no other hospital;
  Global      one logistic model fitted on the pooled data of all hospitals (no privacy; generous);
  GHB         MAP with prior N(mu_tilde_k, diag(tau_hat^2 + M)), coordinatewise borrowing from the
              privatized releases by gossip on the ring+2-hop graph (T = 60 rounds); tau^2 is the posterior mean
              under a half-Cauchy(0.5) prior on tau, from all releases (Remark 8.3); the network
              estimate of mu is combined with the anchor mu_0 by precision weighting (layer L3).
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
    K = len(n)
    theta = MU + np.sqrt(tau2) * rng.standard_normal((K, P))
    mu0 = MU + ANCHOR_SD * rng.standard_normal(P)
    c = np.zeros(P)  # fixed public reference on which the private releases are centred (not the anchor)
    Info = fisher_info(rng, c)  # public: computed at c on the public feature distribution
    Iinv = np.linalg.inv(Info)
    A = (Info + LAM * np.eye(P)) @ Iinv  # undoes the ridge shrinkage to first order (post-processing)
    data = []
    for k in range(K):
        X = features(rng, int(n[k]))
        data.append((X, rng.binomial(1, expit(X @ theta[k])).astype(float)))
    # releases
    s2 = dp_noise_var(n, eps, clip=R_X / LAM)  # (2 C / n)^2 ... with C = R_X / LAM
    rel = np.array([fit(X, y, c, np.full(P, len(y) * LAM)) for X, y in data])
    rel = rel + np.sqrt(s2)[:, None] * rng.standard_normal((K, P))
    rel = c + (rel - c) @ A.T
    r = np.diag(Iinv)[None, :] / n[:, None] + s2[:, None] * np.diag(A @ A.T)[None, :]
    # network borrowing, coordinatewise (the Averager works on (replicates, K) arrays)
    t2, mu_t, M = (np.empty((K, P)) for _ in range(3))
    for d in range(P):
        a, b, c = ghb.borrow(rel[None, :, d], r[None, :, d], avg, tau2=TAU2_EST)
        t2[:, d], mu_t[:, d], M[:, d] = a[0], b[0], c[0]
    # knowledge anchor (L3): combine the network estimate with the published model mu_0, whose
    # stated uncertainty is ANCHOR_SD^2 per coefficient, by precision weighting
    prec = 1.0 / M + 1.0 / ANCHOR_SD ** 2
    mu_t = (mu_t / M + mu0[None, :] / ANCHOR_SD ** 2) / prec
    M = 1.0 / prec
    Xall = np.vstack([X for X, _ in data]); yall = np.concatenate([y for _, y in data])
    glob = fit(Xall, yall, np.zeros(P), np.full(P, 1e-3))
    out = {m: dict(kl=[], auc=[], mse=[]) for m in ("Local", "Global", "GHB", "Oracle")}
    for k, (X, y) in enumerate(data):
        ests = {"Local": fit(X, y, mu0, np.ones(P)),
                "Global": glob,
                "GHB": fit(X, y, mu_t[k], 1.0 / np.maximum(t2[k] + M[k], 1e-6)),
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
