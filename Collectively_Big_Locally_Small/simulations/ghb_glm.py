"""GHB v4 and the literature baselines for generalized linear models other than the logistic (FLamby benchmark).

New code paths only: ghb_v4.py, ghb_v3.py, literature.py, personalized.py and ghb_sa.py are untouched. Every function
here takes a `fam` (GLM family) and mirrors its logistic original line by line, including the order of random draws,
so that with fam=LOGISTIC it reproduces the original output (checked in flamby_bench.py `check`).

Families (y is the outcome array of one hospital; a row of X is one patient, ||x|| <= G):
  LOGISTIC        y in {0, 1}; loss log(1 + e^s) - y s; information weight p (1 - p) (<= 1/4).
  POISSON         exponential survival model as a Poisson GLM with log-exposure offset: y = (d, t, c) with d the event
                  indicator and t = min(time / 3650 days, 1) the follow-up time scaled by a public constant (10 years);
                  loss t e^s - d s (the log-likelihood d s - t e^s up to terms free of theta); information weight t e^s.
                  At the origin the per-patient score is x (d - t), |d - t| <= 1, so ||score|| <= G, and the
                  per-patient information is t x x^T (Frobenius norm <= G^2); replacing one patient moves the
                  information sum by at most 2 G^2 (used conservatively; the logistic analogue is G^2 / 2 = 2 G^2 / 4).
  CLIPPED_POISSON the loss whose per-patient gradient is the Poisson gradient x (t e^s - d) clipped to norm G (DP-SGD
                  style): since t e^s - d is increasing in s, clipping the vector to norm G is capping the scalar at
                  c = G / ||x|| (third column of y; the lower side never binds since t e^s - d >= -1 >= -c), which is
                  the derivative of a convex (Huberized) loss. All literature methods use it, so their sensitivity
                  arguments (2G / n per gradient; 2G / (n lam) for strongly convex local fits) hold unchanged.

GHB (estimate) is ghb_v4.estimate restricted to what the FLamby protocol uses (one network, f = 1, pan off, no
cross/anchor); stage 1 evaluates the score only at the origin when R1 = 1. If R1 > 1 the per-patient scores are
clipped to norm G (a no-op for the logistic family and at the origin).
"""
import numpy as np
from scipy.special import expit

import ghb_logit as GL
import ghb_v3 as V3

S0 = GL.PRIOR_SD
SDS = V3.SDS
Q = V3.Q
SMAX = 50.0  # guard against overflow of e^s far outside any fitted range


class Logistic:
    name = "logistic"
    clip = False

    @staticmethod
    def nll(s, y):
        return np.logaddexp(0, s) - y * s

    @staticmethod
    def dloss(s, y):
        return expit(s) - y

    @staticmethod
    def w(s, y):
        p = expit(s)
        return p * (1 - p)

    @staticmethod
    def info_sens(G):
        return G ** 2 / 2


class Poisson:
    name = "poisson"
    clip = True  # per-patient clipping in stage 1 when scores are not taken at the origin

    @staticmethod
    def nll(s, y):
        return y[:, 1] * np.exp(np.minimum(s, SMAX)) - y[:, 0] * s

    @staticmethod
    def dloss(s, y):
        return y[:, 1] * np.exp(np.minimum(s, SMAX)) - y[:, 0]

    @staticmethod
    def w(s, y):
        return y[:, 1] * np.exp(np.minimum(s, SMAX))

    @staticmethod
    def info_sens(G):
        return 2 * G ** 2


class ClippedPoisson:
    name = "clipped_poisson"
    clip = False  # the clipping is in the loss itself

    @staticmethod
    def _star(y):
        d, t, c = y[:, 0], y[:, 1], y[:, 2]
        with np.errstate(divide="ignore"):
            return np.where(t > 0, np.log((d + c) / np.where(t > 0, t, 1.0)), np.inf)

    @classmethod
    def nll(cls, s, y):
        d, t, c = y[:, 0], y[:, 1], y[:, 2]
        st = cls._star(y)
        sm = np.minimum(np.minimum(s, st), SMAX)
        base = t * np.exp(sm) - d * sm
        return base + np.where(s > st, c * (s - np.where(np.isfinite(st), st, 0.0)), 0.0)

    @classmethod
    def dloss(cls, s, y):
        return np.minimum(y[:, 1] * np.exp(np.minimum(s, SMAX)) - y[:, 0], y[:, 2])

    @classmethod
    def w(cls, s, y):
        return np.where(s < cls._star(y), y[:, 1] * np.exp(np.minimum(s, SMAX)), 0.0)

    @staticmethod
    def info_sens(G):
        return 2 * G ** 2


LOGISTIC, POISSON, CLIPPED_POISSON = Logistic(), Poisson(), ClippedPoisson()


# ------------------------------------------------------------------ fits (mirror exp7_logistic.fit and ghb_v3.fit_mat)
def fit(fam, X, y, prior_mean, prior_prec, iters=100):
    """MAP with prior N(prior_mean, diag(1/prior_prec)), damped Newton (as exp7_logistic.fit)."""
    def obj(t):
        return np.sum(fam.nll(X @ t, y)) + 0.5 * np.sum(prior_prec * (t - prior_mean) ** 2)

    th = prior_mean.copy()
    f = obj(th)
    for _ in range(iters):
        s = X @ th
        g = X.T @ fam.dloss(s, y) + prior_prec * (th - prior_mean)
        H = (X * fam.w(s, y)[:, None]).T @ X + np.diag(prior_prec)
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


def fit_mat(fam, X, y, m, Prec, iters=60):
    """MAP under N(m, Prec^{-1}) with a full precision matrix (as ghb_v3.fit_mat)."""
    th = m.copy()

    def obj(t):
        d = t - m
        return np.sum(fam.nll(X @ t, y)) + 0.5 * d @ Prec @ d

    f = obj(th)
    if fam.clip:  # unbounded likelihood (Poisson): start from the better of m and 0 (same MAP, the objective is convex)
        z = np.zeros_like(m)
        if not np.isfinite(f) or obj(z) < f:
            th, f = z, obj(z)
    for _ in range(iters):
        s = X @ th
        g = X.T @ fam.dloss(s, y) + Prec @ (th - m)
        H = (X * fam.w(s, y)[:, None]).T @ X + Prec
        step = np.linalg.solve(H, g) if not fam.clip else np.linalg.lstsq(H, g, rcond=None)[0]
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


def combine(fam, X, y, comps):
    """Evidence-weighted (Laplace) average of the MAP fits under each Gaussian component (as ghb_v3.combine)."""
    fits, lw = [], []
    for logw, m, C in comps:
        Prec = np.linalg.inv(C)
        t = fit_mat(fam, X, y, m, Prec)
        s = X @ t
        H = (X * fam.w(s, y)[:, None]).T @ X + Prec
        d = t - m
        lw.append(logw - np.sum(fam.nll(s, y)) - 0.5 * d @ Prec @ d
                  + 0.5 * np.linalg.slogdet(Prec)[1] - 0.5 * np.linalg.slogdet(H)[1])
        fits.append(t)
    lw = np.array(lw)
    if fam.clip:  # unbounded likelihood: a component whose Laplace evidence is not finite (a prior mean so far from
        lw = np.where(np.isfinite(lw), lw, -np.inf)  # the data that the fit fails) gets weight 0; logistic unchanged
    w = np.exp(lw - lw.max())
    return (w[:, None] * np.array(fits)).sum(0) / w.sum()


def combine_loo(fam, X, y, comps):
    """ghb_v4.combine_loo (mode pbma) for a GLM family: prior weight x exp(sum of leave-one-out log predictive
    densities), with eta_-i = eta_i + h_i dloss_i / (1 - w_i h_i) (one Newton step; local data only)."""
    fits, lw = [], []
    for logw, m, C in comps:
        Prec = np.linalg.inv(C)
        t = fit_mat(fam, X, y, m, Prec)
        s = X @ t
        wv = fam.w(s, y)
        Hi = np.linalg.inv((X * wv[:, None]).T @ X + Prec)
        h = np.einsum("ij,jk,ik->i", X, Hi, X)
        el = s + h * fam.dloss(s, y) / np.maximum(1 - wv * h, 1e-6)
        v = logw - np.sum(fam.nll(el, y))
        lw.append(v if np.isfinite(v) else -np.inf)
        fits.append(t)
    lw = np.array(lw)
    w = np.exp(lw - lw.max())
    return (w[:, None] * np.array(fits)).sum(0) / w.sum()


def local_eb(fam, X, y):
    P = X.shape[1]
    return combine(fam, X, y, [(0.0, np.zeros(P), np.eye(P) * s ** 2) for s in SDS])


# ------------------------------------------------------------------------------------------------ GHB v4, one network
def _psd(H, floor):
    H = (H + H.T) / 2
    ev, U = np.linalg.eigh(H)
    return (U * np.maximum(ev, floor)) @ U.T


def _score_sum(fam, X, y, w, G):
    r = -fam.dloss(X @ w, y)
    if fam.clip:
        nrm = np.abs(r) * np.linalg.norm(X, axis=1)
        r = r * np.minimum(1.0, G / np.maximum(nrm, 1e-300))
    return X.T @ r


def stage1(fam, rng, data, rho1, G, R1=2, hshare=None):
    """ghb_v4.stage1 with the family's information at the origin and its sensitivity."""
    P = data[0][0].shape[1]
    N = sum(len(y) for _, y in data)
    hs = 1.0 / (R1 + 1) if hshare is None else hshare
    rg, rh = rho1 * (1 - hs) / R1, rho1 * hs
    sg = 0.0 if np.isinf(rg) else 2 * G / np.sqrt(2 * rg)
    sh = 0.0 if np.isinf(rh) else fam.info_sens(G) / np.sqrt(2 * rh)
    H = sum((X * fam.w(np.zeros(len(X)), y)[:, None]).T @ X for X, y in data)
    E = np.triu(rng.standard_normal((P, P))) * sh
    H0 = _psd(H + E + np.triu(E, 1).T, 1e-2 * N)
    A = np.linalg.inv(H0 + np.eye(P) / S0 ** 2)
    w = np.zeros(P)
    for _ in range(R1):
        g = sum(_score_sum(fam, X, y, w, G) for X, y in data) + sg * rng.standard_normal(P)
        w = np.clip(w + A @ (g - w / S0 ** 2), -8, 8)
    return w, H0, sg ** 2 * A @ A * R1


def estimate(fam, data, rng, eps, delta, G, cfg=None):
    """ghb_v4.estimate for one network (list of (X, y)), f = 1, no pan/cross/anchor. Returns (K, P)."""
    cfg = {"f": 1.0, "safety": True, "cross": False, "pan": False, "R1": 2, "hshare": None, "loo": True, "vdp": True,
           "scale": GL.SCALE, "safety_w": 1.0 / 3, **(cfg or {})}
    assert cfg["f"] == 1.0 and not cfg["cross"] and not cfg["pan"] and cfg.get("anchor") is None
    P = data[0][0].shape[1]
    n = np.array([len(y) for _, y in data], float)
    rho = GL.rho_total(eps, delta)
    w, H, Vdp = stage1(fam, rng, data, cfg["f"] * rho, G, cfg["R1"], cfg["hshare"])
    p = 1.0 / (1 + (GL.GRID / cfg["scale"]) ** 2) * np.gradient(GL.GRID)
    tq = np.repeat(V3.quantiles(p / p.sum())[:, None], P, 1)
    est = np.empty((len(data), P))
    for j, (X, y) in enumerate(data):
        sj = -X.T @ fam.dloss(X @ w, y)  # hospital j's exact score at w (local, no privacy cost)
        Hj0 = (X * fam.w(np.zeros(len(X)), y)[:, None]).T @ X
        Hm = _psd(H - Hj0, 1e-2 * max(n.sum() - len(y), 1)) + np.eye(P) / S0 ** 2
        V = np.linalg.inv(Hm)
        mu = w - V @ sj if cfg["loo"] else w
        V = V + Vdp if cfg["vdp"] else V
        comps = []
        ex = np.r_[cfg["free_int"] ** 2, np.zeros(P - 1)] if cfg.get("free_int") else np.zeros(P)  # v5 options:
        for q in range(Q):                                     # intercept not borrowed, own default prior int_sd
            Cn = np.diag(tq[q] + ex) + V
            Cni = np.linalg.inv(Cn)
            for s in SDS:
                D0 = np.full(P, 1.0 / s ** 2)
                if cfg.get("int_sd"):
                    D0[0] = 1.0 / cfg["int_sd"] ** 2
                C = np.linalg.inv(Cni + np.diag(D0))
                comps.append((np.log(1.0 / (Q * len(SDS))), C @ Cni @ mu, C))
        if cfg["safety"]:
            for s in SDS:
                Cs = np.eye(P) * s ** 2
                if cfg.get("int_sd"):
                    Cs[0, 0] = cfg["int_sd"] ** 2
                comps.append((np.log(cfg["safety_w"] / len(SDS)), np.zeros(P), Cs))
        est[j] = combine(fam, X, y, comps) if cfg.get("weights", "evidence") == "evidence" else combine_loo(fam, X, y, comps)
    return est


# ------------------------------------------------------------------- literature baselines (mirror real_v3.baselines)
LAM, FT_ETA = 0.2, 1.0  # as exp7_logistic.LAM and literature.FT_ETA


def grad(fam, X, y, th):
    return X.T @ fam.dloss(X @ th, y) / len(y)


def noisy_grads(fam, rng, data, thetas, sd, P):
    return np.array([grad(fam, X, y, thetas[k]) + sd[k] * rng.standard_normal(P) for k, (X, y) in enumerate(data)])


def fine_tune(fam, data, start, steps):
    out, th, done = {}, np.array(start, float).copy(), 0
    for E in sorted(steps):
        for _ in range(E - done):
            th = th - FT_ETA * np.array([grad(fam, X, y, th[k]) for k, (X, y) in enumerate(data)])
        done = E
        out[E] = th.copy()
    return out


def fedavg(fam, rng, data, n, init, rho, R, eta, G, members=None):
    P = len(init)
    idx = np.arange(len(n)) if members is None else np.asarray(members)
    w = n[idx] / n[idx].sum()
    sd = np.zeros(len(idx)) if np.isinf(rho) else 2 * G / n[idx] / np.sqrt(2 * rho / R)
    sub = [data[k] for k in idx]
    th = np.array(init, float).copy()
    for _ in range(R):
        g = noisy_grads(fam, rng, sub, np.repeat(th[None], len(idx), 0), sd, P)
        th = th - eta * (w @ g)
    return th


def fedavg_sa(fam, rng, data, n, init, rho, R, eta, G):
    P = len(init)
    n = np.asarray(n, float)
    N = n.sum()
    sd = 0.0 if np.isinf(rho) else 2 * G / np.sqrt(2 * rho / R)
    th = np.array(init, float).copy()
    for _ in range(R):
        g = sum(X.T @ fam.dloss(X @ th, y) for X, y in data)
        th = th - eta * (g + sd * rng.standard_normal(P)) / N
    return th


def private_clusters(fam, rng, data, n, rho, C, G, P, iters=25):
    raw = np.array([fit(fam, X, y, np.zeros(P), np.full(P, len(y) * LAM)) for X, y in data])
    s2 = np.zeros(len(n)) if np.isinf(rho) else (2 * G / (LAM * n)) ** 2 / (2 * rho)
    x = raw + np.sqrt(s2)[:, None] * rng.standard_normal(raw.shape)
    w = 1.0 / (s2 + 1.0 / n)
    cent = [x[np.argmax(w)]]
    for _ in range(C - 1):
        d = np.min([((x - c) ** 2).sum(1) for c in cent], axis=0) * w
        cent.append(x[rng.choice(len(n), p=d / d.sum())])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - cent[None]) ** 2).sum(2), axis=1)
        for c in range(C):
            if (lab == c).any():
                cent[c] = (w[lab == c, None] * x[lab == c]).sum(0) / w[lab == c].sum()
    return lab


def _sd(n, rho, R, G, factor=1.0):
    n = np.asarray(n, float)
    return np.zeros(len(n)) if np.isinf(rho) else factor * 2 * G / n / np.sqrt(2 * rho / R)


def _avg(rng, vecs, n, sd, P):
    w = np.asarray(n, float) / np.sum(n)
    noisy = np.array(vecs) + sd[:, None] * rng.standard_normal((len(vecs), P))
    return w @ noisy


def ditto(fam, rng, data, n, rho, R, eta, lam, G, P, steps=5, eta_p=0.5):
    sd = _sd(n, rho, R, G)
    w = np.zeros(P)
    v = np.zeros((len(data), P))
    for _ in range(R):
        g = _avg(rng, [grad(fam, X, y, w) for X, y in data], n, sd, P)
        for k, (X, y) in enumerate(data):
            for _ in range(steps):
                v[k] = v[k] - eta_p * (grad(fam, X, y, v[k]) + lam * (v[k] - w))
        w = w - eta * g
    return np.array([fit(fam, X, y, w, np.full(P, lam)) for X, y in data])


def pfedme(fam, rng, data, n, rho, R, eta, lam, G, P, S=5):
    beta = min(eta * lam, 1.0)
    sd = _sd(n, rho, R, G, factor=S * beta * np.asarray(n, float) / lam)  # sensitivity S beta 2G / lam (personalized.pfedme)
    w = np.zeros(P)
    for _ in range(R):
        deltas = []
        for X, y in data:
            wk = w.copy()
            for _ in range(S):
                th = fit(fam, X, y, wk, np.full(P, lam))
                wk = wk - beta * (wk - th)
            deltas.append(wk - w)
        w = w + _avg(rng, deltas, n, sd, P)
    return np.array([fit(fam, X, y, w, np.full(P, lam)) for X, y in data])


def apfl(fam, rng, data, n, rho, R, eta, alpha, G, P, steps=5, eta_p=0.5):
    sd = _sd(n, rho, R, G)
    w = np.zeros(P)
    v = np.zeros((len(data), P))
    for _ in range(R):
        g = _avg(rng, [grad(fam, X, y, w) for X, y in data], n, sd, P)
        for k, (X, y) in enumerate(data):
            for _ in range(steps):
                mix = alpha * v[k] + (1 - alpha) * w
                v[k] = v[k] - eta_p * alpha * grad(fam, X, y, mix)
        w = w - eta * g
    return alpha * v + (1 - alpha) * w[None, :]


def ckey(c):
    return "|".join(str(x) for x in c)


def baselines(fam, rng, train, rho, G):
    """Every configuration of real_v3.GRIDS, in the order and with the random draws of real_v3.baselines.
    Takes the zCDP budget rho directly (so that ISIC heads can each get rho / 8)."""
    import personalized as PZ  # grids only
    P = train[0][0].shape[1]
    n = np.array([len(y) for _, y in train], float)
    K = len(train)
    out = {}
    z = np.zeros(P)
    for R in (1, 3, 10, 30):
        for eta in (0.2, 0.5, 2.0):
            g = fedavg(fam, rng, train, n, z, rho, R, eta, G)
            for E, th in fine_tune(fam, train, np.repeat(g[None], K, 0), steps=(1, 5, 30)).items():
                out[ckey(("FT", R, eta, E))] = th
    for R in (3, 10, 30):
        for eta in (0.5, 2.0):
            g = fedavg_sa(fam, rng, train, n, z, rho, R, eta, G)
            for E, th in fine_tune(fam, train, np.repeat(g[None], K, 0), steps=(0, 5, 30)).items():
                out[ckey(("FTSA", R, eta, E))] = th
    for R, eta, a in PZ.GRID["APFL"]:
        out[ckey(("APFL", R, eta, a))] = apfl(fam, rng, train, n, rho, R, eta, a, G, P)
    for R, eta, lam in PZ.GRID["Ditto"]:
        out[ckey(("PDIT", R, eta, lam))] = ditto(fam, rng, train, n, rho, R, eta, lam, G, P)
    for R, eta, lam in PZ.GRID["pFedMe"]:
        out[ckey(("PME", R, eta, lam))] = pfedme(fam, rng, train, n, rho, R, eta, lam, G, P)
    for C in (2, 3):
        lab = private_clusters(fam, rng, train, n, rho / 2, C, G, P)
        for R in (1, 3, 10):
            for eta in (0.2, 0.5, 2.0):
                th0 = np.empty((K, P))
                for c in range(C):
                    mem = np.flatnonzero(lab == c)
                    if len(mem):
                        th0[mem] = fedavg(fam, rng, train, n, z, rho / 2, R, eta, G, mem)
                for E, th in fine_tune(fam, train, th0, steps=(0, 5, 30)).items():
                    out[ckey(("CLU", C, R, eta, E))] = th
    return out
