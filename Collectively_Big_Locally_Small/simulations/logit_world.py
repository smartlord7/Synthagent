"""The logistic-regression world of Simulation 7, generalized for Simulations 8, 10 and 11.

Each hospital has its own logistic model on four standardized features and an intercept,
theta_k = mu + u_k with u_k ~ N(0, tau^2 I). Optional departures from Simulation 7:
  shift       hospital k draws its features from N(m_k, I) with m_k ~ N(0, shift^2 I) (case mix);
  anchor_err  standard deviation of the error of the published anchor mu_0 (truth: mu);
  anchor_sd   the uncertainty the anchor is *stated* to have, which GHB uses (overconfident if smaller);
  check       GHB drops the anchor when it conflicts with the network estimate (|z| > 3 in any coordinate);
              on by default.
  regions     (G, tau_r2): a shared effect per region (Simulation 15).

GHB is implemented in ghb_logit.py. Every one-shot method below that uses other hospitals consumes the same
privatized releases (GHB's score releases, inverted to the parameter scale), so all of them satisfy the
same (epsilon, delta) guarantee, except "Global (pooled, no DP)", which is a non-private reference.
releases(), network() and with_anchor() are the earlier version of GHB (ridge releases centred on a fixed
reference, first-order de-shrinking, plug-in tau^2), kept for the development note (legacy=True).
"""
import numpy as np
from scipy.special import expit

import ghb
import ghb_logit as GL
from blueprint import dp_noise_var
from exp7_logistic import LAM, MU, N_TEST, P, R_X, auc, fisher_info, fit

DITTO_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
METHODS = ("Local", "Global (pooled, no DP)", "FedAvg (DP)", "Meta-analysis (DP)", "Ditto-style (DP)", "GHB (DP)")


def features(rng, m, shift_mean=None):
    z = rng.standard_normal((m, P - 1))
    if shift_mean is not None:
        z = z + shift_mean
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z * np.minimum(1.0, 2.0 / nrm)  # ||x|| <= sqrt(5) whatever the shift
    return np.hstack([np.ones((m, 1)), z])


def world(rng, n, tau2, shift=0.0, anchor_err=0.3, regions=None):
    """Draw one world. regions = (G, tau_r2) adds a shared effect N(0, tau_r2 I) per region, hospital k in region k mod G."""
    K = len(n)
    theta = MU + np.sqrt(tau2) * rng.standard_normal((K, P))
    if regions is not None:
        G, tau_r2 = regions
        theta = theta + np.sqrt(tau_r2) * rng.standard_normal((G, P))[np.arange(K) % G]
    means = shift * rng.standard_normal((K, P - 1))
    mu0 = MU + anchor_err * rng.standard_normal(P)
    data = []
    for k in range(K):
        X = features(rng, int(n[k]), means[k])
        data.append((X, rng.binomial(1, expit(X @ theta[k])).astype(float)))
    return theta, means, mu0, data


def releases(rng, data, n, mu0, eps, deshrink="anchor", centre="zero"):
    """Private releases, de-shrunk with public Fisher information, and their public variances.

    The ridge of each release is centred on a fixed public reference c ("zero": c = 0, the default)
    or on the anchor mu_0 ("anchor"); its sensitivity does not depend on the centre. Simulation 11
    showed that centring on the anchor lets a wrong anchor corrupt every release, so all logistic
    simulations centre on c = 0 and the anchor enters only through the prior of the final fit. The
    first-order de-shrinking uses the Fisher information of the public feature distribution at c
    ("anchor") or at each hospital's own release ("release"); both are post-processing.
    """
    K = len(n)
    c = mu0 if centre == "anchor" else np.zeros(P)
    s2 = dp_noise_var(n, eps, clip=R_X / LAM)
    raw = np.array([fit(X, y, c, np.full(P, len(y) * LAM)) for X, y in data])
    raw = raw + np.sqrt(s2)[:, None] * rng.standard_normal((K, P))
    rel = np.empty_like(raw)
    r = np.empty_like(raw)
    Xpub = features(rng, 20000)
    for j in range(K):
        at = c if deshrink == "anchor" else np.clip(raw[j], -6, 6)
        p = expit(Xpub @ at)
        Info = (Xpub * (p * (1 - p))[:, None]).T @ Xpub / len(Xpub)
        Iinv = np.linalg.inv(Info)
        A = (Info + LAM * np.eye(P)) @ Iinv
        rel[j] = c + A @ (raw[j] - c)
        r[j] = np.diag(Iinv) / n[j] + s2[j] * np.diag(A @ A.T)
    return rel, r


def network(rel, r, avg, tau2="bayes"):
    K = rel.shape[0]
    t2, mu_t, M = (np.empty((K, P)) for _ in range(3))
    for d in range(P):
        a, b, c = ghb.borrow(rel[None, :, d], r[None, :, d], avg, tau2=tau2)
        t2[:, d], mu_t[:, d], M[:, d] = a[0], b[0], c[0]
    return t2, mu_t, M


def with_anchor(mu_t, M, mu0, anchor_sd, check=False):
    if anchor_sd is None:
        return mu_t, M, np.zeros(len(mu_t), bool)
    z = (mu0[None, :] - mu_t) / np.sqrt(anchor_sd ** 2 + M)
    drop = np.abs(z).max(axis=1) > 3 if check else np.zeros(len(mu_t), bool)
    prec = 1.0 / M + 1.0 / anchor_sd ** 2
    comb = (mu_t / M + mu0[None, :] / anchor_sd ** 2) / prec
    mu_c = np.where(drop[:, None], mu_t, comb)
    M_c = np.where(drop[:, None], M, 1.0 / prec)
    return mu_c, M_c, drop


def evaluate(rng, theta_k, mean_k, est):
    Xt = features(rng, N_TEST, mean_k)
    pt = expit(Xt @ theta_k)
    yt = rng.binomial(1, pt).astype(float)
    bayes = float(np.mean(-(pt * np.log(pt) + (1 - pt) * np.log1p(-pt))))
    out = {}
    for m, th in est.items():
        s = Xt @ th
        out[m] = (float(np.mean(np.logaddexp(0, s) - pt * s)) - bayes, auc(s, yt))
    return out


def one_rep(rng, n, tau2, eps, avg, shift=0.0, anchor_err=0.3, anchor_sd=0.3, check=True,
            ditto_lam=None, ghb_variants=None, baselines=True, regions=None, extra=None, oracle_tau2=False,
            legacy=False):
    """Excess log-loss and AUC of every method at every hospital for one replicate.

    GHB is ghb_logit (score releases at the anchor, moment inversion, tau integrated out). ghb_variants maps a
    name to dict(anchor_sd=..., check=...) and reuses the same releases. extra(rng, data, n, mu0, eps) may return
    further estimates, a dict name -> (K, P) array (Simulation 15). oracle_tau2 adds "GHB, oracle tau2 (DP)".
    """
    theta, means, mu0, data = world(rng, n, tau2, shift, anchor_err, regions)
    pub = GL.Public(features(rng, 20000))
    g, sd = GL.release(rng, data, n, mu0, eps)
    variants = ghb_variants or {"GHB (DP)": dict(anchor_sd=anchor_sd, check=check)}
    ests, drops, first = {}, {}, None
    for name, cfg in variants.items():
        th, r, drop = GL.invert(pub, g, sd, n, mu0, avg, cfg.get("anchor_sd"), cfg.get("check", True))
        ests[name] = GL.estimate(data, th, r, drop, mu0, avg, cfg.get("anchor_sd"))
        drops[name] = drop
        first = first or (th, r, drop, cfg)
    if oracle_tau2:
        th, r, drop, cfg = first
        t2_true = tau2 + (regions[1] if regions is not None else 0.0)
        ests["GHB, oracle tau2 (DP)"] = GL.estimate(data, th, r, drop, mu0, avg, cfg.get("anchor_sd"), oracle_tau2=t2_true)
    if legacy:  # the earlier version of GHB, on its own releases (same budget, classical Gaussian mechanism)
        rel0, r0 = releases(rng, data, n, mu0, eps)
        t2o, mu_o, M_o = network(rel0, r0, avg)
        mc, Mc, _ = with_anchor(mu_o, M_o, mu0, anchor_sd, check)
        ests["GHB, earlier version (DP)"] = np.array([fit(X, y, mc[k], 1.0 / np.maximum(t2o[k] + Mc[k], 1e-6))
                                                      for k, (X, y) in enumerate(data)])
    if baselines:
        rel, r = first[0], first[1]  # the same privatized releases, on the parameter scale
        w_n = n / n.sum()
        fedavg = (rel * w_n[:, None]).sum(axis=0)  # one-shot federated averaging of the releases
        t2_all = np.array([ghb.dersimonian_laird(rel[None, :, d], r[None, :, d], ghb.Averager())[0, 0] for d in range(P)])
        w_iv = 1.0 / (t2_all[None, :] + r)
        meta = (rel * w_iv).sum(axis=0) / w_iv.sum(axis=0)  # random-effects (DerSimonian-Laird) pooled mean
        Xall = np.vstack([X for X, _ in data]); yall = np.concatenate([y for _, y in data])
        pooled = fit(Xall, yall, np.zeros(P), np.full(P, 1e-3))
    more = extra(rng, data, n, mu0, eps) if extra is not None else {}
    res = {}
    for k, (X, y) in enumerate(data):
        est = {"Local": fit(X, y, mu0, np.ones(P))}
        for name, e in ests.items():
            est[name] = e[k]
        if baselines:
            est["Global (pooled, no DP)"] = pooled
            est["FedAvg (DP)"] = fedavg
            est["Meta-analysis (DP)"] = meta
            for lam in (DITTO_GRID if ditto_lam is None else (ditto_lam,)):
                est[f"Ditto-style (DP), lambda={lam}"] = fit(X, y, fedavg, np.full(P, lam))
        for name, th_ in more.items():
            est[name] = th_[k]
        res[k] = evaluate(rng, theta[k], means[k], est)
    methods = list(res[0])
    kl = {m: np.array([res[k][m][0] for k in range(len(n))]) for m in methods}
    au = {m: np.array([res[k][m][1] for k in range(len(n))]) for m in methods}
    return kl, au, drops


def run(rng, n, reps, **kw):
    """Stack one_rep over replicates: dict method -> (reps, K) arrays."""
    KL, AU, DR = {}, {}, {}
    for _ in range(reps):
        kl, au, dr = one_rep(rng, n, **kw)
        for m in kl:
            KL.setdefault(m, []).append(kl[m]); AU.setdefault(m, []).append(au[m])
        for m in dr:
            DR.setdefault(m, []).append(dr[m])
    return ({m: np.array(v) for m, v in KL.items()}, {m: np.array(v) for m, v in AU.items()},
            {m: np.array(v) for m, v in DR.items()})


def best_ditto(KL):
    """Name of the Ditto-style variant with the lowest mean excess log-loss (oracle tuning)."""
    cands = [m for m in KL if m.startswith("Ditto-style")]
    return min(cands, key=lambda m: KL[m].mean())
