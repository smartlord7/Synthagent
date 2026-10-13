"""GHB for images: one softmax head on the features of a frozen public encoder (development on ISIC 2019, declared as
such in dev/PREREG_image.md; confirmation pre-registered there).

Why a new head. FLamby's ISIC 2019 run (dev/PREREG_flamby.md, prediction 4) used a 16-dimensional random projection of
the encoder features and eight one-vs-rest logistic heads, each with an eighth of the budget, and no correction for the
very different class mixes of the six centers. Every borrowing method then sat at the local level. Three changes, all
public or local, so none costs privacy:

  projection   PCA of the encoder features of the public ISIC 2019 Test images (unlabeled, disjoint from FLamby's
               centers), d components; per-center standardization as before, /3, clip to [-1, 1], intercept: P = d + 1.
  softmax head theta is (P, K) for K classes, one model and one budget. The per-patient score x (e_y - p)^T has
               Frobenius norm at most sqrt(2) G (||e_y - p|| <= sqrt(2)), so a replacement moves a score sum by at most
               GS = 2 sqrt(2) G; every method below (GHB and the literature) uses this bound.
  logit        each hospital predicts argmax_k s_k - log pi_k, where pi is the class mix the model implies on the
  adjustment   hospital's own training images (the mean predicted probability; local, no privacy cost). For a local
               model pi is the hospital's class mix, which is the post-hoc logit adjustment of Menon et al. (2021) for
               balanced accuracy; the same rule is applied to every method.

GHB (estimate), one network, as ghb_v4 with f = 1 and the curvature taken from public images instead of released:
  Stage 1  R1 rounds of differentially private Newton on the pooled softmax model: each round the network releases
           the secure sum of the hospitals' scores at the current w (Gaussian noise on the sum only, rho / R1 each);
           the Hessian is the public one, N E_pub[(diag p - p p^T) (x) x x^T] at w, which needs features and the
           model but no label (N is public under replacement neighbours). Vdp = sg^2 A A^T of the last round.
  Stage 2  hospital j: mu_-j = w - H_-j^{-1} s_j(w) with H_-j = H_pub(w) - H_j(w) (own exact score and information,
           local), V_-j = H_-j^{-1} + Vdp; prior components N(mu_-j, tau_q^2 I + V_-j) x N(0, s^2 I) for the Q
           half-Cauchy(0.5) quantiles of tau and s in SDS, plus the no-borrowing components with weight safety_w;
           optionally (free_int) the intercepts get an extra prior variance S0^2, so that the class mix is learned
           locally (label shift) while the image-feature coefficients are borrowed.
  Final    evidence-weighted average of the MAP fits (Laplace), as ghb_v3.combine.
"""
import numpy as np
from scipy.special import logsumexp, softmax

import ghb_logit as GL
import ghb_v3 as V3

S0 = GL.PRIOR_SD
SDS = V3.SDS
Q = V3.Q


# ---------------------------------------------------------------------------------------------------- softmax model
def onehot(y, K):
    return np.eye(K)[np.asarray(y, int)]


def nll(X, Y, th):
    S = X @ th.reshape(X.shape[1], -1)
    return float(np.sum(logsumexp(S, 1) - np.sum(Y * S, 1)))


def score(X, Y, th):
    """Sum of the per-patient scores x (e_y - p)^T, flattened (P * K)."""
    Pm = softmax(X @ th.reshape(X.shape[1], -1), 1)
    return (X.T @ (Y - Pm)).ravel()


def hess(X, th, Pm=None):
    """Information sum: sum_i (x_i x_i^T) kron (diag p_i - p_i p_i^T), index p * K + k."""
    P = X.shape[1]
    if Pm is None:
        Pm = softmax(X @ th.reshape(P, -1), 1)
    K = Pm.shape[1]
    B = (X[:, :, None] * Pm[:, None, :]).reshape(len(X), P * K)
    H = -(B.T @ B).reshape(P, K, P, K)
    for k in range(K):
        H[:, k, :, k] += (X * Pm[:, k:k + 1]).T @ X
    return H.reshape(P * K, P * K)


def fit_mat(X, Y, m, Prec, start=None, iters=50):
    """MAP under N(m, Prec^{-1}): damped Newton from `start` (warm start) or m."""
    th = (m if start is None else start).copy()

    def obj(t):
        d = t - m
        return nll(X, Y, t) + 0.5 * d @ Prec @ d

    f = obj(th)
    for _ in range(iters):
        g = -score(X, Y, th) + Prec @ (th - m)
        H = hess(X, th) + Prec
        step = np.linalg.solve(H, g)
        a = 1.0
        while a > 1e-8:
            fn = obj(th - a * step)
            if fn <= f - 1e-4 * a * g @ step:
                break
            a /= 2
        th, f = th - a * step, fn
        if np.abs(a * step).max() < 1e-7:
            break
    return th


def fit(X, Y, m, prec, start=None):
    """MAP under N(m, diag(1/prec)) (prec a scalar or a vector)."""
    return fit_mat(X, Y, m, np.diag(np.broadcast_to(prec, m.shape).astype(float)), start)


def combine(X, Y, comps):
    """Evidence-weighted (Laplace) average of the MAP fits under each Gaussian component (as ghb_v3.combine)."""
    fits, lw, start = [], [], None
    for logw, m, C in comps:
        Prec = np.linalg.inv(C)
        t = fit_mat(X, Y, m, Prec, start)
        H = hess(X, t) + Prec
        d = t - m
        lw.append(logw - nll(X, Y, t) - 0.5 * d @ Prec @ d
                  + 0.5 * np.linalg.slogdet(Prec)[1] - 0.5 * np.linalg.slogdet(H)[1])
        fits.append(t)
        start = t
    lw = np.array(lw)
    w = np.exp(lw - lw.max())
    return (w[:, None] * np.array(fits)).sum(0) / w.sum()


def local_eb(X, Y):
    D = X.shape[1] * Y.shape[1]
    return combine(X, Y, [(0.0, np.zeros(D), np.eye(D) * s ** 2) for s in SDS])


# ------------------------------------------------------------------------------------------------------------- GHB
def _psd(H, floor):
    H = (H + H.T) / 2
    ev, U = np.linalg.eigh(H)
    return (U * np.maximum(ev, floor)) @ U.T


def stage1(rng, data, Xpub, rho1, GS, R1, damp=1.0):
    """data: list of (X, Y one-hot). Damped Newton w <- w + damp A (g - w / S0^2), A from the public information at w
    (undamped Newton overshoots on the softmax and diverges: development finding). Returns w, the public information
    sum at w, and Vdp, the stationary covariance of the noise in the damped iteration, damp / (2 - damp) sg^2 A A^T."""
    P, K = data[0][0].shape[1], data[0][1].shape[1]
    D = P * K
    N = sum(len(Y) for _, Y in data)
    sg = 0.0 if np.isinf(rho1) else GS / np.sqrt(2 * rho1 / R1)
    w = np.zeros(D)
    for _ in range(R1):
        A = np.linalg.inv(N / len(Xpub) * hess(Xpub, w) + np.eye(D) / S0 ** 2)
        g = sum(score(X, Y, w) for X, Y in data) + sg * rng.standard_normal(D)
        w = np.clip(w + damp * A @ (g - w / S0 ** 2), -8, 8)
    return w, N / len(Xpub) * hess(Xpub, w), damp / (2 - damp) * sg ** 2 * A @ A.T


def estimate(data, Xpub, rng, eps, delta, G, cfg=None):
    """data: list of (X, y int); Xpub: public feature rows (with intercept). Returns (K_hosp, P * K)."""
    cfg = {"R1": 3, "damp": 1.0, "safety": True, "safety_w": 1.0 / 3, "free_int": False, "scale": GL.SCALE, **(cfg or {})}
    K = int(max(y.max() for _, y in data)) + 1 if "K" not in cfg else cfg["K"]
    P = data[0][0].shape[1]
    D = P * K
    dY = [(X, onehot(y, K)) for X, y in data]
    N = sum(len(y) for _, y in data)
    rho = GL.rho_total(eps, delta)
    w, H, Vdp = stage1(rng, dY, Xpub, rho, 2 * np.sqrt(2) * G, cfg["R1"], cfg["damp"])
    p = 1.0 / (1 + (GL.GRID / cfg["scale"]) ** 2) * np.gradient(GL.GRID)
    tq = V3.quantiles(p / p.sum())
    extra = np.zeros(D)
    if cfg["free_int"]:
        extra[:K] = S0 ** 2  # row 0 of theta (the intercepts) is the first K entries
    est = np.empty((len(data), D))
    for j, (X, Y) in enumerate(dY):
        sj = score(X, Y, w)
        Hm = _psd(H - hess(X, w), 1e-3 * max(N - len(Y), 1)) + np.eye(D) / S0 ** 2
        V = np.linalg.inv(Hm)
        mu = w - V @ sj
        V = V + Vdp
        comps = []
        for q in range(Q):
            Cni = np.linalg.inv(V + np.diag(tq[q] + extra))
            for s in SDS:
                C = np.linalg.inv(Cni + np.eye(D) / s ** 2)
                comps.append((np.log(1.0 / (Q * len(SDS))), C @ Cni @ mu, C))
        if cfg["safety"]:
            for s in SDS:
                comps.append((np.log(cfg["safety_w"] / len(SDS)), np.zeros(D), np.eye(D) * s ** 2))
        est[j] = combine(X, Y, comps)
    return est


# ---------------------------------------------------------------- literature baselines (mirror ghb_glm.baselines)
LAM, FT_ETA = 0.2, 1.0


def grad(X, Y, th):
    return -score(X, Y, th) / len(Y)


def fine_tune(data, start, steps):
    out, th, done = {}, np.array(start, float).copy(), 0
    for E in sorted(steps):
        for _ in range(E - done):
            th = th - FT_ETA * np.array([grad(X, Y, th[k]) for k, (X, Y) in enumerate(data)])
        done = E
        out[E] = th.copy()
    return out


def _sd(n, rho, R, GS, factor=1.0):
    n = np.asarray(n, float)
    return np.zeros(len(n)) if np.isinf(rho) else factor * GS / n / np.sqrt(2 * rho / R)


def _avg(rng, vecs, n, sd, D):
    w = np.asarray(n, float) / np.sum(n)
    return w @ (np.array(vecs) + sd[:, None] * rng.standard_normal((len(vecs), D)))


def fedavg(rng, data, n, init, rho, R, eta, GS, members=None):
    D = len(init)
    idx = np.arange(len(n)) if members is None else np.asarray(members)
    w = n[idx] / n[idx].sum()
    sd = _sd(n[idx], rho, R, GS)
    th = np.array(init, float).copy()
    for _ in range(R):
        g = np.array([grad(*data[k], th) + sd[i] * rng.standard_normal(D) for i, k in enumerate(idx)])
        th = th - eta * (w @ g)
    return th


def fedavg_sa(rng, data, n, init, rho, R, eta, GS):
    D = len(init)
    N = np.sum(n)
    sd = 0.0 if np.isinf(rho) else GS / np.sqrt(2 * rho / R)
    th = np.array(init, float).copy()
    for _ in range(R):
        g = -sum(score(X, Y, th) for X, Y in data)
        th = th - eta * (g + sd * rng.standard_normal(D)) / N
    return th


def private_clusters(rng, data, n, rho, C, GS, D, iters=25):
    raw = np.array([fit(X, Y, np.zeros(D), len(Y) * LAM) for X, Y in data])
    s2 = np.zeros(len(n)) if np.isinf(rho) else (GS / (LAM * n)) ** 2 / (2 * rho)
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


def ditto(rng, data, n, rho, R, eta, lam, GS, D):
    sd = _sd(n, rho, R, GS)
    w = np.zeros(D)
    for _ in range(R):
        w = w - eta * _avg(rng, [grad(X, Y, w) for X, Y in data], n, sd, D)
    return np.array([fit(X, Y, w, lam) for X, Y in data])


def pfedme(rng, data, n, rho, R, eta, lam, GS, D, S=5):
    beta = min(eta * lam, 1.0)
    sd = _sd(n, rho, R, GS, factor=S * beta / lam)
    w = np.zeros(D)
    th = [None] * len(data)
    for _ in range(R):
        deltas = []
        for k, (X, Y) in enumerate(data):
            wk = w.copy()
            for _ in range(S):
                th[k] = fit(X, Y, wk, lam, th[k])
                wk = wk - beta * (wk - th[k])
            deltas.append(wk - w)
        w = w + _avg(rng, deltas, n, sd, D)
    return np.array([fit(X, Y, w, lam, th[k]) for k, (X, Y) in enumerate(data)])


def apfl(rng, data, n, rho, R, eta, alpha, GS, D, steps=5, eta_p=0.5):
    sd = _sd(n, rho, R, GS)
    w = np.zeros(D)
    v = np.zeros((len(data), D))
    for _ in range(R):
        g = _avg(rng, [grad(X, Y, w) for X, Y in data], n, sd, D)
        for k, (X, Y) in enumerate(data):
            for _ in range(steps):
                v[k] = v[k] - eta_p * alpha * grad(X, Y, alpha * v[k] + (1 - alpha) * w)
        w = w - eta * g
    return alpha * v + (1 - alpha) * w[None, :]


def ckey(c):
    return "|".join(str(x) for x in c)


def baselines(rng, train, rho, G, K, keys=None):
    """The configurations of real_v3.GRIDS (all, or only `keys`), as ghb_glm.baselines, with the multinomial
    sensitivity GS = 2 sqrt(2) G in place of 2 G."""
    import personalized as PZ
    GS = 2 * np.sqrt(2) * G
    P = train[0][0].shape[1]
    D = P * K
    data = [(X, onehot(y, K)) for X, y in train]
    n = np.array([len(y) for _, y in train], float)
    H = len(train)
    want = (lambda k: True) if keys is None else (lambda k: k in keys)
    out = {}
    z = np.zeros(D)
    for R in (1, 3, 10, 30):
        for eta in (0.2, 0.5, 2.0):
            if any(want(ckey(("FT", R, eta, E))) for E in (1, 5, 30)):
                g = fedavg(rng, data, n, z, rho, R, eta, GS)
                for E, th in fine_tune(data, np.repeat(g[None], H, 0), steps=(1, 5, 30)).items():
                    out[ckey(("FT", R, eta, E))] = th
    for R in (3, 10, 30):
        for eta in (0.5, 2.0):
            if any(want(ckey(("FTSA", R, eta, E))) for E in (0, 5, 30)):
                g = fedavg_sa(rng, data, n, z, rho, R, eta, GS)
                for E, th in fine_tune(data, np.repeat(g[None], H, 0), steps=(0, 5, 30)).items():
                    out[ckey(("FTSA", R, eta, E))] = th
    for R, eta, a in PZ.GRID["APFL"]:
        if want(ckey(("APFL", R, eta, a))):
            out[ckey(("APFL", R, eta, a))] = apfl(rng, data, n, rho, R, eta, a, GS, D)
    for R, eta, lam in PZ.GRID["Ditto"]:
        if want(ckey(("PDIT", R, eta, lam))):
            out[ckey(("PDIT", R, eta, lam))] = ditto(rng, data, n, rho, R, eta, lam, GS, D)
    for R, eta, lam in PZ.GRID["pFedMe"]:
        if want(ckey(("PME", R, eta, lam))):
            out[ckey(("PME", R, eta, lam))] = pfedme(rng, data, n, rho, R, eta, lam, GS, D)
    for C in (2, 3):
        if not any(want(ckey(("CLU", C, R, eta, E))) for R in (1, 3, 10) for eta in (0.2, 0.5, 2.0) for E in (0, 5, 30)):
            continue
        lab = private_clusters(rng, data, n, rho / 2, C, GS, D)
        for R in (1, 3, 10):
            for eta in (0.2, 0.5, 2.0):
                th0 = np.empty((H, D))
                for c in range(C):
                    mem = np.flatnonzero(lab == c)
                    if len(mem):
                        th0[mem] = fedavg(rng, data, n, z, rho / 2, R, eta, GS, mem)
                for E, th in fine_tune(data, th0, steps=(0, 5, 30)).items():
                    out[ckey(("CLU", C, R, eta, E))] = th
    return out


# ---------------------------------------------------------------------------------------------------- prediction
def predict(th, X, Xtr=None, adjust="model"):
    """Class predictions; adjust: None (argmax s), "model" (s - log of the class mix the model implies on the
    hospital's training images Xtr) or a vector of class frequencies."""
    S = X @ th.reshape(X.shape[1], -1)
    if adjust is None:
        return S.argmax(1)
    pi = softmax(Xtr @ th.reshape(X.shape[1], -1), 1).mean(0) if isinstance(adjust, str) else np.asarray(adjust)
    return (S - np.log(np.maximum(pi, 1e-12))).argmax(1)


def balanced_accuracy(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))
