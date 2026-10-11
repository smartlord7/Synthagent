"""GHB for logistic heads with every component switchable, for the ablation (Simulation 16) and sensitivity
(Simulation 17) studies. With the default configuration it computes the same estimator as ghb_logit.py; it is
a separate module so that the reported simulations keep using the code they were run with.

Configuration keys (defaults in DEFAULT, which is GHB as reported):
  release      'score' (mean score at the anchor) | 'xty' (mean x*y, no control variate) | 'ridge' (ridge fit
               centred on 0 with weight lam, the earlier GHB's release)
  accounting   'zcdp' | 'classical' (Gaussian mechanism with s = Delta sqrt(2 ln(1.25/delta)) / eps)
  relinearize  re-linearize each release at the hospital's leave-one-out network estimate (else pooled point only)
  tau          'integrate' (posterior quantiles, weighted by own evidence) | 'mean' | 'median' (plug-in posterior
               mean or median of tau^2) | 'dl' (plug-in DerSimonian-Laird) | 'oracle' (true tau^2)
  evidence     weight the tau components by the hospital's own marginal likelihood (else equal weights)
  Q, scale     number of posterior quantiles of tau; half-Cauchy scale of its prior
  prior_sd     weakly informative default prior N(0, prior_sd^2) on each coefficient (None: none)
  anchor_sd    stated uncertainty of the anchor (None: no anchor)
  check, z     anchor conflict check and its threshold
  weights      'iv' (inverse-variance) | 'equal' leave-one-out network mean
  exact        exact averaging (a server) instead of gossip
  lam          ridge weight of the 'ridge' release
"""
import numpy as np
from scipy.special import expit

import ghb
from exp7_logistic import P, fit
from ghb_logit import G, GRID, rho_total

DEFAULT = dict(release="score", accounting="zcdp", relinearize=True, tau="integrate", evidence=True, Q=12, scale=0.5,
               prior_sd=2.5, anchor_sd=0.3, check=True, z=3.0, weights="iv", exact=False, lam=0.2)


class Releases:
    """The statistics every release type needs, computed once per replicate, and a shared noise draw, so that
    variants of one replicate differ only by the component that is switched (paired comparisons)."""

    def __init__(self, rng, data, n, mu0, lam=0.2):
        self.n, self.mu0, self.lam = np.asarray(n, float), mu0, lam
        self.score = np.array([X.T @ (y - expit(X @ mu0)) / len(y) for X, y in data])
        self.xty = np.array([X.T @ y / len(y) for X, y in data])
        self.ridge = np.array([fit(X, y, np.zeros(P), np.full(P, len(y) * lam)) for X, y in data])
        self.Z = rng.standard_normal((len(n), P))

    def get(self, cfg, eps, delta=1e-5):
        kind = cfg["release"]
        sens = 2 * G / self.n / (cfg["lam"] if kind == "ridge" else 1.0)
        if eps is None or np.isinf(eps):
            sd = np.zeros(len(self.n))
        elif cfg["accounting"] == "zcdp":
            sd = sens / np.sqrt(2 * rho_total(eps, delta))
        else:
            sd = sens * np.sqrt(2 * np.log(1.25 / delta)) / eps
        stat = {"score": self.score, "xty": self.xty, "ridge": self.ridge}[kind]
        return stat + sd[:, None] * self.Z, sd


def _cov(pub, th, pc):
    """Covariance of x (y - pc) when y ~ Bernoulli(sigmoid(x'theta)) and pc is a fixed per-row offset."""
    pt = expit(pub.X @ th)
    E = (pub.X * (pt * (1 - pt) + (pt - pc) ** 2)[:, None]).T @ pub.X / len(pub.X)
    d = pub.X.T @ (pt - pc) / len(pub.X)
    return E - np.outer(d, d)


def linearize(pub, z, sd, n, mu0, points, cfg):
    """Release -> parameter scale at public points, with delta-method variances, for each release type."""
    th, r = np.empty_like(z), np.empty_like(z)
    kind = cfg["release"]
    if kind == "score":
        pc, mc = expit(pub.X @ mu0), pub.m(mu0)
    elif kind == "xty":
        pc, mc = np.zeros(len(pub.X)), np.zeros(P)
    for j in range(len(n)):
        a = points[j]
        I = pub.info(a)
        Ii = np.linalg.inv(I)
        if kind == "ridge":
            A = (I + cfg["lam"] * np.eye(P)) @ Ii
            th[j] = z[j] + cfg["lam"] * Ii @ z[j]          # centre c = 0
            r[j] = np.diag(Ii / n[j] + sd[j] ** 2 * A @ A.T)
        else:
            th[j] = a + Ii @ (z[j] + mc - pub.m(a))
            r[j] = np.diag(Ii @ (_cov(pub, a, pc) / n[j] + sd[j] ** 2 * np.eye(P)) @ Ii.T)
    return th, r


def pooled_point(pub, z, n, mu0, cfg):
    if cfg["release"] == "ridge":
        return np.zeros(P)
    mc = pub.m(mu0) if cfg["release"] == "score" else np.zeros(P)
    return pub.invert((n[:, None] * z).sum(0) / n.sum() + mc, np.zeros(P))


def tau_posterior(th, r, scale):
    """(P, G) posterior probabilities of tau on GRID, per coordinate (half-Cauchy prior, flat prior on mu)."""
    out = np.empty((P, len(GRID)))
    for d in range(P):
        x, rr = th[:, d], r[:, d]
        lp = []
        for t in GRID:
            V = t * t + rr
            w = 1.0 / V
            mu = (w * x).sum() / w.sum()
            lp.append(-0.5 * np.log(V).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (x - mu) ** 2).sum()
                      - np.log1p((t / scale) ** 2))
        lp = np.array(lp)
        p = np.exp(lp - lp.max()) * np.gradient(GRID)
        out[d] = p / p.sum()
    return out


def tau_components(th, r, cfg, avg, true_tau2=None):
    """List of tau^2 settings, each a (K, P) array (per hospital and coordinate)."""
    K = len(th)
    kind = cfg["tau"]
    if kind == "oracle":
        return [np.full((K, P), true_tau2)]
    if kind == "dl":
        t2 = np.empty((K, P))
        for d in range(P):
            t2[:, d] = ghb.dersimonian_laird(th[None, :, d], r[None, :, d], avg)[0]
        return [t2]
    post = tau_posterior(th, r, cfg["scale"])
    cdf = np.cumsum(post, axis=1)
    if kind == "mean":
        return [np.repeat((post * GRID[None, :] ** 2).sum(1)[None], K, 0)]
    if kind == "median":
        idx = np.argmax(cdf >= 0.5, axis=1)
        return [np.repeat((GRID[idx] ** 2)[None], K, 0)]
    Q = cfg["Q"]
    comps = []
    for q in range(Q):
        idx = np.minimum(np.argmax(cdf >= (q + 0.5) / Q, axis=1), len(GRID) - 1)
        comps.append(np.repeat((GRID[idx] ** 2)[None], K, 0))
    return comps


def network(th, r, avg, t2, weights):
    K = len(th)
    mu, M = np.empty((K, P)), np.empty((K, P))
    for d in range(P):
        _, b, c = ghb.borrow(th[None, :, d], r[None, :, d], avg, tau2=t2[None, :, d], weights=weights)
        mu[:, d], M[:, d] = b[0], c[0]
    return mu, M


def with_anchor(mu, M, mu0, anchor_sd, drop):
    if anchor_sd is None:
        return mu, M
    prec = 1.0 / M + 1.0 / anchor_sd ** 2
    comb, Mc = (mu / M + mu0[None, :] / anchor_sd ** 2) / prec, 1.0 / prec
    return np.where(drop[:, None], mu, comb), np.where(drop[:, None], M, Mc)


def conflicts(mu, M, mu0, cfg):
    if cfg["anchor_sd"] is None or not cfg["check"]:
        return np.zeros(len(mu), bool)
    return (np.abs(mu0[None, :] - mu) / np.sqrt(cfg["anchor_sd"] ** 2 + M)).max(axis=1) > cfg["z"]


def combine(X, y, means, variances, cfg):
    fits, logev = [], []
    for m, v in zip(means, variances):
        pr = 1.0 / np.maximum(v, 1e-8)
        if cfg["prior_sd"] is not None:
            pr = pr + 1.0 / cfg["prior_sd"] ** 2
            m = (m / np.maximum(v, 1e-8)) / pr
        t = fit(X, y, m, pr)
        s = X @ t
        p = expit(s)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pr)
        logev.append(np.sum(y * s - np.logaddexp(0, s)) - 0.5 * np.sum(pr * (t - m) ** 2)
                     + 0.5 * np.sum(np.log(pr)) - 0.5 * np.linalg.slogdet(H)[1])
        fits.append(t)
    if cfg["evidence"]:
        w = np.exp(np.array(logev) - max(logev))
    else:
        w = np.ones(len(fits))
    return (w[:, None] * np.array(fits)).sum(axis=0) / w.sum()


def estimate(pub, rel, data, n, mu0, eps, avg_gossip, cfg=None, true_tau2=None, delta=1e-5):
    """GHB estimates (K, P) under configuration cfg (missing keys take DEFAULT)."""
    cfg = {**DEFAULT, **(cfg or {})}
    n = np.asarray(n, float)
    avg = ghb.Averager() if cfg["exact"] else avg_gossip
    z, sd = rel.get(cfg, eps, delta)
    a0 = pooled_point(pub, z, n, mu0, cfg)
    th, r = linearize(pub, z, sd, n, mu0, np.repeat(a0[None], len(n), 0), cfg)
    comps = tau_components(th, r, cfg, avg, true_tau2)
    mid = comps[len(comps) // 2]
    mu_l, M_l = network(th, r, avg, mid, cfg["weights"])
    drop = conflicts(mu_l, M_l, mu0, cfg)
    if cfg["relinearize"] and cfg["release"] != "ridge":
        th, r = linearize(pub, z, sd, n, mu0, with_anchor(mu_l, M_l, mu0, cfg["anchor_sd"], drop)[0], cfg)
        comps = tau_components(th, r, cfg, avg, true_tau2)
    elif cfg["relinearize"]:  # ridge: undo the shrinkage at the leave-one-out point instead of at 0
        th, r = linearize(pub, z, sd, n, mu0, with_anchor(mu_l, M_l, mu0, cfg["anchor_sd"], drop)[0], cfg)
        comps = tau_components(th, r, cfg, avg, true_tau2)
    nets = [with_anchor(*network(th, r, avg, t2, cfg["weights"]), mu0, cfg["anchor_sd"], drop) + (t2,) for t2 in comps]
    return np.array([combine(X, y, [c[0][k] for c in nets], [c[1][k] + c[2][k] for c in nets], cfg)
                     for k, (X, y) in enumerate(data)])
