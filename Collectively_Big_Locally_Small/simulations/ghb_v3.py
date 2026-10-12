"""GHB v3: GHB-EB with four additions, for any number of coefficients P.

Units are hospitals, or (cancer type, institution) cells when several networks share institutions. Each unit releases
its mean score at the public point 0 once (Gaussian mechanism, zero-concentrated accounting, sensitivity 2G/n with
G = max ||x||), exactly as GHB-EB; everything below is post-processing of those releases and costs no privacy.

  (4) safety       each unit's mixture always contains a no-borrowing component (the default prior alone), so its own
                   evidence can switch borrowing off when its data disagree with the network (an atypical unit).
  (5) learned      the scale s of the default prior N(0, s^2) is learned at the network level from the releases: with
      regularization  theta_j = mu + u_j and mu ~ N(0, s^2), the releases' marginal likelihood gives a posterior over s on
                   {0.5, 1, 2.5}, which replaces the equal prior weights of GHB-EB's scale components.
  (6) correlated   besides the per-coefficient components for tau (as GHB), components with a full between-unit
      heterogeneity  covariance Sigma (method of moments from the releases, projected onto the positive semidefinite
                   cone), scaled by {0.5, 1, 2}, so that coefficients that vary together across hospitals are borrowed
                   together; the final fit uses a full prior precision matrix.
  (1) cross-cancer two-level hierarchy theta_ch = mu_pan + a_c + b_h + e_ch per coefficient: a cancer's network mean is
                   shrunk toward the pan-cancer mean of the other cancers (between-cancer variance psi^2 by
                   DerSimonian-Laird), and a cell borrows the institution effect b_h from the same institution's
                   deviations in its other cancers (variance sigma_b^2 from cross-cancer covariances of the deviations
                   of the same institution, which privacy noise does not bias because releases are independent).

Each unit's final model averages its MAP fits under every component, weighted by the Laplace evidence of its own data
and by the components' prior weights. Leave-one-out network means are exact (the limit that gossip reaches; Simulation
4 shows 60 rounds suffice).
"""
import numpy as np
from scipy.special import expit

import ghb_logit as GL

SDS = (0.5, 1.0, 2.5)
SIGMA_SCALES = (0.5, 1.0, 2.0)
Q = 12


class Public:
    def __init__(self, X):
        self.X = X
        self.P = X.shape[1]

    def m(self, th):
        return self.X.T @ expit(self.X @ th) / len(self.X)

    def info(self, th):
        p = expit(self.X @ th)
        return (self.X * (p * (1 - p))[:, None]).T @ self.X / len(self.X)

    def score_cov(self, th, c):
        pt, pc = expit(self.X @ th), expit(self.X @ c)
        E = (self.X * (pt * (1 - pt) + (pt - pc) ** 2)[:, None]).T @ self.X / len(self.X)
        d = self.m(th) - self.m(c)
        return E - np.outer(d, d)

    def invert(self, target, start, ridge, iters=60):
        th = np.array(start, float)
        for _ in range(iters):
            step = np.linalg.solve(self.info(th) + ridge * np.eye(self.P), target - self.m(th) - ridge * th)
            big = np.abs(step).max()
            if big > 1:
                step = step / big
            th = np.clip(th + step, -8, 8)
            if big < 1e-10:
                break
        return th


def fit_mat(X, y, m, Prec, iters=60):
    """MAP of logistic regression under the Gaussian prior N(m, Prec^{-1}) with a full precision matrix."""
    th = m.copy()

    def obj(t):
        s = X @ t
        d = t - m
        return np.sum(np.logaddexp(0, s) - y * s) + 0.5 * d @ Prec @ d

    f = obj(th)
    for _ in range(iters):
        p = expit(X @ th)
        g = X.T @ (p - y) + Prec @ (th - m)
        H = (X * (p * (1 - p))[:, None]).T @ X + Prec
        step = np.linalg.solve(H, g)
        a = 1.0
        while a > 1e-8:
            fn = obj(th - a * step)
            if fn <= f - 1e-4 * a * g @ step:
                break
            a /= 2
        th, f = th - a * step, fn
        if np.abs(a * step).max() < 1e-8:
            break
    return th


def combine(X, y, comps):
    """comps: list of (log prior weight, mean, covariance matrix). Evidence-weighted average of the MAP fits."""
    fits, lw = [], []
    for logw, m, C in comps:
        Prec = np.linalg.inv(C)
        t = fit_mat(X, y, m, Prec)
        s = X @ t
        p = expit(s)
        H = (X * (p * (1 - p))[:, None]).T @ X + Prec
        d = t - m
        lw.append(logw + np.sum(y * s - np.logaddexp(0, s)) - 0.5 * d @ Prec @ d
                  + 0.5 * np.linalg.slogdet(Prec)[1] - 0.5 * np.linalg.slogdet(H)[1])
        fits.append(t)
    lw = np.array(lw)
    w = np.exp(lw - lw.max())
    return (w[:, None] * np.array(fits)).sum(0) / w.sum()


def release(rng, data, eps, delta, G):
    n = np.array([len(y) for _, y in data], float)
    P = data[0][0].shape[1]
    rho = GL.rho_total(eps, delta)
    sd = np.zeros(len(n)) if np.isinf(rho) else 2 * G / n / np.sqrt(2 * rho)
    g = np.array([X.T @ (y - 0.5) / len(y) for X, y in data])  # score at the public point 0
    return g + sd[:, None] * rng.standard_normal((len(n), P)), sd


def linearize(pub, g, sd, n, points, prior_sd=GL.PRIOR_SD):
    """theta~_j and its full covariance R_j, linearized at points[j] under the default prior."""
    P = pub.P
    m0 = pub.m(np.zeros(P))
    th, R = np.empty_like(g), np.empty((len(n), P, P))
    for j in range(len(n)):
        a = points[j]
        Ii = np.linalg.inv(pub.info(a) + np.eye(P) / (n[j] * prior_sd ** 2))
        th[j] = a + Ii @ (g[j] + m0 - pub.m(a))
        R[j] = Ii @ (pub.score_cov(a, np.zeros(P)) / n[j] + sd[j] ** 2 * np.eye(P)) @ Ii.T
    return th, R


def tau2_post(x, r, grid=GL.GRID, scale=GL.SCALE):
    lp = []
    for t in grid:
        V = t * t + r
        w = 1.0 / V
        mu = (w * x).sum() / w.sum()
        lp.append(-0.5 * np.log(V).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (x - mu) ** 2).sum() - np.log1p((t / scale) ** 2))
    lp = np.array(lp)
    p = np.exp(lp - lp.max()) * np.gradient(grid)
    return p / p.sum()


def quantiles(p, grid=GL.GRID, q=Q):
    cdf = np.cumsum(p)
    return grid[np.minimum(np.searchsorted(cdf, (np.arange(q) + 0.5) / q), len(grid) - 1)] ** 2


def loo_mean(th, r, t2, j):
    """Inverse-variance leave-one-out mean of coordinate arrays (K,), its variance."""
    w = 1.0 / (t2 + r)
    w = np.delete(w, j)
    x = np.delete(th, j)
    return (w * x).sum() / w.sum(), 1.0 / w.sum()


def sigma_hat(th, R):
    """Method-of-moments between-unit covariance, projected onto the PSD cone."""
    w = 1.0 / np.array([np.trace(Rj) for Rj in R])
    mu = (w[:, None] * th).sum(0) / w.sum()
    d = th - mu
    S = (w[:, None, None] * np.einsum('ki,kj->kij', d, d)).sum(0) / w.sum() - (w[:, None, None] * R).sum(0) / w.sum()
    ev, U = np.linalg.eigh((S + S.T) / 2)
    return (U * np.maximum(ev, 0)) @ U.T


def scale_weights(th, r, t2_med, sds=SDS):
    """(5) posterior over the default-prior scale s from the releases: theta~_j ~ N(mu, t2 + r_j), mu ~ N(0, s^2)."""
    lp = []
    for s in sds:
        tot = 0.0
        for d in range(th.shape[1]):
            v = t2_med[d] + r[:, d]
            w = 1.0 / v
            A = w.sum() + 1.0 / s ** 2
            b = (w * th[:, d]).sum()
            tot += -0.5 * np.log(s ** 2 * w.sum() + 1.0) + 0.5 * b ** 2 / A  # integrating mu out, up to constants in s
        lp.append(tot)
    lp = np.array(lp)
    w = np.exp(lp - lp.max())
    return w / w.sum()


def network_stats(pub, data, rng, eps, delta, G):
    """Releases of one network, inverted and linearized as GHB-EB (anchor-free, default-prior linearization)."""
    n = np.array([len(y) for _, y in data], float)
    P = pub.P
    g, sd = release(rng, data, eps, delta, G)
    a0 = pub.invert((n[:, None] * g).sum(0) / n.sum() + pub.m(np.zeros(P)), np.zeros(P), 1.0 / (n.sum() * GL.PRIOR_SD ** 2))
    th0, R0 = linearize(pub, g, sd, n, np.repeat(a0[None], len(n), 0))
    r0 = np.array([np.diag(Rj) for Rj in R0])
    t2m = np.array([quantiles(tau2_post(th0[:, d], r0[:, d]))[Q // 2] for d in range(P)])
    pts = np.array([[loo_mean(th0[:, d], r0[:, d], t2m[d], j)[0] for d in range(P)] for j in range(len(n))])
    th, R = linearize(pub, g, sd, n, pts)  # second linearization at the leave-one-out network estimate
    return dict(n=n, th=th, R=R, r=np.array([np.diag(Rj) for Rj in R]))


def estimate(nets, pubs, rng, eps, delta, G, inst=None, cfg=None):
    """GHB v3 estimates for several networks at once.

    nets: dict name -> list of (X, y) per unit; pubs: dict name -> Public; inst: dict name -> list of institution ids
    (enables (1)); cfg: dict with booleans safety, learned_scale, fullcov, cross (all True by default).
    Returns dict name -> (K, P) estimates."""
    cfg = {"safety": True, "learned_scale": True, "fullcov": True, "cross": True, **(cfg or {})}
    stats = {c: network_stats(pubs[c], d, rng, eps, delta, G) for c, d in nets.items()}
    P = next(iter(pubs.values())).P
    names = list(nets)
    # per-network tau posteriors, medians, and covariance
    tq, t2med, Sig = {}, {}, {}
    for c in names:
        st = stats[c]
        tq[c] = np.array([quantiles(tau2_post(st["th"][:, d], st["r"][:, d])) for d in range(P)]).T  # (Q, P)
        t2med[c] = tq[c][Q // 2]
        Sig[c] = sigma_hat(st["th"], st["R"])
    # (1) cross-cancer: network means, pan-cancer shrinkage and institution effects
    use_cross = cfg["cross"] and inst is not None and len(names) > 1
    netmean, netvar = {}, {}
    for c in names:
        st = stats[c]
        w = 1.0 / (t2med[c][None, :] + st["r"])
        netmean[c] = (w * st["th"]).sum(0) / w.sum(0)
        netvar[c] = 1.0 / w.sum(0)
    if use_cross:
        M_ = np.array([netmean[c] for c in names]); V_ = np.array([netvar[c] for c in names])
        psi2 = np.array([max(0.0, np.var(M_[:, d]) - V_[:, d].mean()) for d in range(P)])
        dev = {}  # institution -> list of (cancer, deviation, variance)
        for c in names:
            st = stats[c]
            for k, h in enumerate(inst[c]):
                dev.setdefault(h, []).append((c, st["th"][k] - netmean[c], st["r"][k] + netvar[c]))
        prods = []
        for h, lst in dev.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    prods.append(lst[a][1] * lst[b][1])
        sb2 = np.maximum(np.mean(prods, axis=0), 0.0) if prods else np.zeros(P)
    out = {}
    for c in names:
        st, data = stats[c], nets[c]
        K = len(data)
        sw = scale_weights(st["th"], st["r"], t2med[c]) if cfg["learned_scale"] else np.ones(len(SDS)) / len(SDS)
        est = np.empty((K, P))
        for j, (X, y) in enumerate(data):
            comps = []
            shift = np.zeros(P)
            extra_var = np.zeros(P)
            if use_cross:
                others = [x for x in dev.get(inst[c][j], []) if x[0] != c]
                tot = np.minimum(sb2, t2med[c])
                if others:
                    dm = np.mean([o[1] for o in others], axis=0)
                    dv = np.mean([o[2] for o in others], axis=0) / len(others) + (t2med[c] - tot) / len(others)
                    k_ = tot / np.maximum(tot + dv, 1e-12)
                    shift, extra_var = k_ * dm, tot * (1 - k_)
                else:
                    extra_var = tot
            for q in range(Q):
                t2 = tq[c][q]
                mu, M = np.array([loo_mean(st["th"][:, d], st["r"][:, d], t2[d], j) for d in range(P)]).T
                if use_cross:  # shrink this cancer's leave-one-out mean toward the other cancers' means
                    om = np.array([netmean[o] for o in names if o != c]); ov = np.array([netvar[o] for o in names if o != c])
                    pan, panv = om.mean(0), ov.mean(0) / len(om) + psi2
                    wgt = panv / (panv + M)
                    mu, M = wgt * mu + (1 - wgt) * pan, 1.0 / (1.0 / M + 1.0 / panv)
                    t2 = np.maximum(t2 - np.minimum(sb2, t2), 0.0) + extra_var
                for si, s in enumerate(SDS):
                    C = np.diag(1.0 / (1.0 / (t2 + M) + 1.0 / s ** 2))
                    m = np.diag(C) * ((mu + shift) / (t2 + M))
                    comps.append((np.log(sw[si] / Q / 2 if cfg["fullcov"] else sw[si] / Q), m, C))
            if cfg["fullcov"]:
                for f in SIGMA_SCALES:
                    mu = np.array([loo_mean(st["th"][:, d], st["r"][:, d], t2med[c][d], j)[0] for d in range(P)])
                    M = np.diag([loo_mean(st["th"][:, d], st["r"][:, d], t2med[c][d], j)[1] for d in range(P)])
                    Cn = f * Sig[c] + M + 1e-6 * np.eye(P)
                    for si, s in enumerate(SDS):
                        Prec = np.linalg.inv(Cn) + np.eye(P) / s ** 2
                        C = np.linalg.inv(Prec)
                        m = C @ np.linalg.solve(Cn, mu + shift)
                        comps.append((np.log(sw[si] / len(SIGMA_SCALES) / 2), m, C))
            if cfg["safety"]:  # (4) no borrowing: the default prior alone, at each learned scale
                for si, s in enumerate(SDS):
                    comps.append((np.log(sw[si] / 3), np.zeros(P), np.eye(P) * s ** 2))
            est[j] = combine(X, y, comps)
        out[c] = est
    return out
