"""GHB with secure aggregation (GHB-SA): privacy noise added once to network sums instead of once per hospital.

With a secure-aggregation protocol, the network learns only sums over hospitals, never a hospital's own release, and
each hospital adds a share of Gaussian noise so that the sum carries noise of the required scale (distributed DP).
The sensitivity of a sum over patients is set by one patient, not by the size of the hospital that holds them, so
small hospitals no longer pay for being small. Two rounds of secure sums (zero-concentrated DP, budget split rho/3,
2 rho/3):

  1. S = sum over all patients of x (y - sigmoid(x' mu0)), the score at the public point mu0; sensitivity 2G.
     The pooled point a0 solves m(a0) = S / N + m(mu0) (under the default prior, as in real data).
  2. Each hospital computes, from its own data and without noise, its deviation from a0 on the parameter scale,
     delta_j = I(a0)^{-1} (g_j + m(mu0) - m(a0)), clipped coordinatewise to [-D, D]. The network sums
     U = sum_j n_j delta_j and T = sum_j n_j delta_j^2 (coordinatewise). Replacing one patient moves delta_j by at
     most c / n_j with c = 2 G ||I(a0)^{-1}||_2, so (U, T) has L2 sensitivity c sqrt(1 + 4 D^2), whatever n_j.

Hospital j knows its own contribution, so its leave-one-out network mean is exact up to the noise of U:
mu_{-j} = a0 + (U - n_j delta_j) / (N - n_j), with error variance M_j = s^2 / (N - n_j) + sigma_U^2 / (N - n_j)^2
+ tau^2 sum_{i != j} n_i^2 / (N - n_j)^2, where s^2 is the per-patient sampling variance on the parameter scale (public).
Heterogeneity: Q = T - U^2 / N has expectation (K - 1) s^2 + tau^2 (N - sum n^2 / N); a posterior over tau on GRID
(half-Cauchy(0.5) prior, Gaussian likelihood for the noisy Q) is summarized by Q_TAU quantiles, and each hospital
weights them, and the scales of the default prior, by the evidence of its own data, as GHB-EB does.

Trust: the network must run a secure-aggregation protocol (pairwise masks or secret sharing, which can run over the
agreement graph without a server); the guarantee assumes the parties do not collude to remove other hospitals'
noise shares. DP-FedAvg with secure aggregation (fedavg_sa) is the matching baseline.
"""
import numpy as np
from scipy.special import expit

import ghb_logit as GL
import literature as LIT
from exp7_logistic import P

G = np.sqrt(5.0)
D_CLIP = 0.5
Q_TAU = 12


def estimate(rng, pub, data, n, mu0, eps, delta=1e-5, anchor_sd=None, prior_sd=GL.PRIOR_SDS, d_clip=D_CLIP):
    n = np.asarray(n, float)
    N, K = n.sum(), len(n)
    rho = GL.rho_total(eps, delta)
    noisy = not np.isinf(rho)
    # round 1: the pooled score
    S = sum(X.T @ (y - expit(X @ mu0)) for X, y in data)
    if noisy:
        S = S + 2 * G / np.sqrt(2 * rho / 3) * rng.standard_normal(P)
    a0 = pub.invert(S / N + pub.m(mu0), np.zeros(P), ridge=1.0 / (N * GL.PRIOR_SD ** 2))
    Ii = np.linalg.inv(pub.info(a0) + np.eye(P) / (N * GL.PRIOR_SD ** 2))
    s2 = np.diag(Ii @ pub.score_cov(a0, mu0) @ Ii.T)                 # per-patient sampling variance, public
    # round 2: clipped deviations, summed
    dm = pub.m(mu0) - pub.m(a0)
    dev = np.array([np.clip(Ii @ (X.T @ (y - expit(X @ mu0)) / len(y) + dm), -d_clip, d_clip) for X, y in data])
    U, T = (n[:, None] * dev).sum(0), (n[:, None] * dev ** 2).sum(0)
    sig2 = 0.0
    if noisy:
        c = 2 * G * np.linalg.norm(Ii, 2)
        sig = c * np.sqrt(1 + 4 * d_clip ** 2) / np.sqrt(2 * (2 * rho / 3))
        U, T, sig2 = U + sig * rng.standard_normal(P), T + sig * rng.standard_normal(P), sig ** 2
    # posterior over tau per coordinate from the noisy Q
    Qs = T - U ** 2 / N
    eff = N - (n ** 2).sum() / N
    t2q = np.empty((Q_TAU, P))
    for d in range(P):
        mean = (K - 1) * s2[d] + GL.GRID ** 2 * eff
        var = sig2 * (1 + 4 * (U[d] / N) ** 2) + 2 * (K - 1) * (s2[d] + GL.GRID ** 2 * n.mean()) ** 2
        lp = -0.5 * (Qs[d] - mean) ** 2 / var - 0.5 * np.log(var) - np.log1p((GL.GRID / GL.SCALE) ** 2)
        p = np.exp(lp - lp.max()) * np.gradient(GL.GRID)
        cdf = np.cumsum(p / p.sum())
        t2q[:, d] = GL.GRID[np.minimum(np.searchsorted(cdf, (np.arange(Q_TAU) + 0.5) / Q_TAU), len(GL.GRID) - 1)] ** 2
    out = np.empty((K, P))
    sq = (n ** 2).sum()
    for j, (X, y) in enumerate(data):
        rest = N - n[j]
        mu = a0 + (U - n[j] * dev[j]) / rest
        means, variances = [], []
        for q in range(Q_TAU):
            M = s2 / rest + sig2 / rest ** 2 + t2q[q] * (sq - n[j] ** 2) / rest ** 2
            m_, v_ = mu, t2q[q] + M
            if anchor_sd is not None:  # precision-weighted with the published model, as GHB's layer L3
                pr = 1.0 / v_ + 1.0 / anchor_sd ** 2
                m_, v_ = (m_ / v_ + mu0 / anchor_sd ** 2) / pr, 1.0 / pr
            means.append(m_); variances.append(v_)
        out[j] = GL.combine(X, y, means, variances, prior_sd)
    return out


def fedavg_sa(rng, data, n, init, rho, R, eta):
    """DP-FedAvg with secure aggregation: Gaussian noise on the summed gradient (sensitivity 2G per round)."""
    n = np.asarray(n, float)
    N = n.sum()
    sd = 0.0 if np.isinf(rho) else 2 * G / np.sqrt(2 * rho / R)
    th = np.array(init, float).copy()
    for _ in range(R):
        g = sum(X.T @ (expit(X @ th) - y) for X, y in data)
        th = th - eta * (g + sd * rng.standard_normal(P)) / N
    return th


def fedavg_sa_ft(rng, data, n, init, rho, R, eta, steps=(0, 1, 5, 30)):
    g = fedavg_sa(rng, data, n, init, rho, R, eta)
    return LIT.fine_tune(data, np.repeat(g[None], len(data), 0), steps=steps)


# ---------------------------------------------------------------------------------------------------------------
# Version 2 (after the development run of version 1, dev/sa_dev.json, which over-borrowed at high heterogeneity):
#  (1, 3) clip each PATIENT's influence on the parameter scale, phi_i = I(a0)^{-1} (x_i (y_i - sigmoid(x_i' mu0)) + dm),
#         to L2 norm C, instead of clipping each hospital's deviation: real between-hospital differences survive,
#         and the sensitivity of the sums is set by C (U: 2C; T: 4C^2), not by a worst-case bound;
#  (2)    each hospital's mixture always includes a no-borrowing component and two high-heterogeneity components,
#         so its own evidence can switch borrowing off when the network's estimate of tau is wrong;
#  (4)    optionally a third secure sum with inverse-variance weights w_j = 1 / (s^2 / n_j + tau_hat^2), tau_hat the
#         posterior median from round 2, so that large hospitals do not dominate the network mean under heterogeneity.
# The budget split, C and the use of round 3 are chosen on development data only (dev/sa_dev2.py).
# ---------------------------------------------------------------------------------------------------------------
EXTRA_TAU2 = (1.0, 4.0)   # high-heterogeneity components always offered (fix 2)


def _clipped_phi(Ii, X, y, mu0, dm, C):
    phi = (X * (y - expit(X @ mu0))[:, None]) @ Ii.T + Ii @ dm
    nrm = np.linalg.norm(phi, axis=1, keepdims=True)
    return phi * np.minimum(1.0, C / np.maximum(nrm, 1e-12))


def _public_var(pub, Ii, a0, mu0, dm, C):
    """Per-patient variance of the clipped influence under the public feature distribution at a0 (public)."""
    p0 = expit(pub.X @ a0)
    f1 = _clipped_phi(Ii, pub.X, np.ones(len(pub.X)), mu0, dm, C)
    f0 = _clipped_phi(Ii, pub.X, np.zeros(len(pub.X)), mu0, dm, C)
    m = (p0[:, None] * f1 + (1 - p0[:, None]) * f0).mean(0)
    e2 = (p0[:, None] * f1 ** 2 + (1 - p0[:, None]) * f0 ** 2).mean(0)
    return np.maximum(e2 - m ** 2, 1e-8)


def estimate_v2(rng, pub, data, n, mu0, eps, delta=1e-5, anchor_sd=None, prior_sd=GL.PRIOR_SDS, C=4.0,
                split=(0.25, 0.35, 0.40), iv=True):
    n = np.asarray(n, float)
    N, K = n.sum(), len(n)
    rho = GL.rho_total(eps, delta)
    noisy = not np.isinf(rho)
    sp = np.asarray(split if iv else (split[0], split[1] + split[2], 0.0), float)
    sp = sp / sp.sum()
    # round 1: pooled score at mu0 (sensitivity 2G)
    S = sum(X.T @ (y - expit(X @ mu0)) for X, y in data)
    if noisy:
        S = S + 2 * G / np.sqrt(2 * rho * sp[0]) * rng.standard_normal(P)
    ridge = 1.0 / (N * GL.PRIOR_SD ** 2)
    a0 = pub.invert(S / N + pub.m(mu0), np.zeros(P), ridge=ridge)
    Ii = np.linalg.inv(pub.info(a0) + ridge * np.eye(P))
    dm = pub.m(mu0) - pub.m(a0)
    s2 = _public_var(pub, Ii, a0, mu0, dm, C)
    # round 2: U = sum_j sum_i phi_i, T = sum_j n_j delta_j^2 (each patient clipped to norm C)
    dev = np.array([_clipped_phi(Ii, X, y, mu0, dm, C).mean(0) for X, y in data])
    U, T = (n[:, None] * dev).sum(0), (n[:, None] * dev ** 2).sum(0)
    sU2 = sT2 = 0.0
    if noisy:
        sU, sT = 2 * C / np.sqrt(2 * rho * sp[1] / 2), 4 * C ** 2 / np.sqrt(2 * rho * sp[1] / 2)
        U, T, sU2, sT2 = U + sU * rng.standard_normal(P), T + sT * rng.standard_normal(P), sU ** 2, sT ** 2
    Qs, eff = T - U ** 2 / N, N - (n ** 2).sum() / N
    t2q, med = np.empty((Q_TAU, P)), np.empty(P)
    for d in range(P):
        mean = (K - 1) * s2[d] + GL.GRID ** 2 * eff
        var = sT2 + (2 * U[d] / N) ** 2 * sU2 + 2 * (K - 1) * (s2[d] + GL.GRID ** 2 * n.mean()) ** 2
        lp = -0.5 * (Qs[d] - mean) ** 2 / var - 0.5 * np.log(var) - np.log1p((GL.GRID / GL.SCALE) ** 2)
        pw = np.exp(lp - lp.max()) * np.gradient(GL.GRID)
        cdf = np.cumsum(pw / pw.sum())
        t2q[:, d] = GL.GRID[np.minimum(np.searchsorted(cdf, (np.arange(Q_TAU) + 0.5) / Q_TAU), len(GL.GRID) - 1)] ** 2
        med[d] = GL.GRID[min(np.searchsorted(cdf, 0.5), len(GL.GRID) - 1)] ** 2
    # network mean: sample-size weights from U, or inverse-variance weights from a third secure sum
    if iv:
        w = 1.0 / (s2[None, :] / n[:, None] + med[None, :])            # (K, P), public given round 2
        Vs = (w * dev).sum(0)
        sV2 = 0.0
        if noisy:
            sens = 2 * C * np.max(w / n[:, None])
            sV = sens / np.sqrt(2 * rho * sp[2])
            Vs, sV2 = Vs + sV * rng.standard_normal(P), sV ** 2
        Wt = w.sum(0)
    else:
        w = np.repeat(n[:, None], P, 1)
        Vs, sV2, Wt = U, sU2, n.sum() * np.ones(P)
    comps = [t2q[q] for q in range(Q_TAU)] + [np.full(P, t) for t in EXTRA_TAU2]
    out = np.empty((K, P))
    for j, (X, y) in enumerate(data):
        rest = Wt - w[j]
        mu = a0 + (Vs - w[j] * dev[j]) / rest
        wr = np.delete(w, j, axis=0)
        nr = np.delete(n, j)
        means, variances = [], []
        for t2 in comps:
            M = ((wr ** 2) * (s2[None, :] / nr[:, None] + t2[None, :])).sum(0) / rest ** 2 + sV2 / rest ** 2
            m_, v_ = mu, t2 + M
            if anchor_sd is not None:
                pr = 1.0 / v_ + 1.0 / anchor_sd ** 2
                m_, v_ = (m_ / v_ + mu0 / anchor_sd ** 2) / pr, 1.0 / pr
            means.append(m_); variances.append(v_)
        means.append(np.zeros(P) if anchor_sd is None else mu0)          # no borrowing (fix 2)
        variances.append(np.full(P, 1e12) if anchor_sd is None else np.full(P, anchor_sd ** 2 + 1.0))
        out[j] = GL.combine(X, y, means, variances, prior_sd)
    return out
