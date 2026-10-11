"""GHB for logistic-regression heads (layer L2 of Algorithm 1): release, inversion, borrowing, combination.

Release (the only thing that leaves a hospital). Each hospital releases its mean score at the published
anchor mu_0,
    g_k = (1/n_k) sum_i x_i (y_i - sigmoid(x_i' mu_0)) + N(0, s_k^2 I),
Per-patient terms have norm at most ||x|| <= sqrt(5), so replacing one patient moves g_k by at most
Delta_k = 2 sqrt(5)/n_k. The noise is calibrated with zero-concentrated DP, s_k = Delta_k / sqrt(2 rho), with rho
chosen so that rho + 2 sqrt(rho ln(1/delta)) = epsilon (Bun and Steinke, 2016). Evaluating the score at mu_0
acts as a control variate: when mu_0 is close to the truth the release is nearly efficient. When mu_0 is
wrong the release stays unbiased and only becomes noisier.

Inversion (post-processing with public information only). Under the public feature distribution,
E[g_k] = m(theta_k) - m(mu_0) with m(theta) = E[x sigmoid(x' theta)], so theta_k is recovered by moment matching,
linearized at a public point a_k:
    theta~_k = a_k + I(a_k)^{-1} (g_k + m(mu_0) - m(a_k)).
Its variance follows by the delta method from the covariance of the per-patient score and the noise. The
first point is the pooled moment inversion, which does not use the anchor. The second is each hospital's
leave-one-out network estimate, combined with the anchor only where the anchor passes the conflict check
against that anchor-free estimate. The anchor therefore never shapes a release it conflicts with.

Borrowing. Coordinatewise, the posterior of tau^2 given the releases under a half-Cauchy(0.5) prior on tau
(Remark 8.3) is summarized by Q equally weighted quantiles. For each quantile, gossip gives the
leave-one-out inverse-variance mean mu~_k and its error M_k, combined with the anchor by precision
weighting unless the hospital's single conflict decision dropped it.

Combination (inside the hospital, no privacy cost). Each component q defines a Gaussian prior
N(mu~_k(q), tau_q^2 + M_k(q)), multiplied by the weakly informative default N(0, 2.5^2) for logistic
coefficients (Gelman et al., 2008). The hospital fits its own data under each component and averages the
fits, weighted by the Laplace approximation of its own marginal likelihood. This is the posterior mean of
theta_k with tau integrated out: a hospital whose data agree with the network borrows strongly, and one
whose data conflict with it puts its weight on wide components and borrows little.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

import ghb
from exp7_logistic import P, fit

G = np.sqrt(5.0)      # bound on the norm of a per-patient score term
GRID = np.r_[0.0, np.logspace(-3.5, 1, 160)]  # grid of tau for its posterior
Q = 12                # posterior quantiles of tau used as mixture components
SCALE = 0.5           # half-Cauchy scale of the prior on tau (Remark 8.3)
PRIOR_SD = 2.5        # weakly informative default prior on each coefficient
PRIOR_SDS = (0.5, 1.0, 2.5)  # scales integrated out by the evidence-weighted default prior (revision after real data)
Z_CONFLICT = 3.0      # anchor conflict threshold


def rho_total(eps, delta):
    """zCDP budget whose (epsilon, delta) conversion equals epsilon (inf when there is no privacy)."""
    if eps is None or np.isinf(eps):
        return np.inf
    return brentq(lambda r: r + 2 * np.sqrt(r * np.log(1 / delta)) - eps, 1e-12, 50)


class Public:
    """Moments of the public feature distribution, estimated once from m draws."""

    def __init__(self, X):
        self.X = X

    def m(self, th):
        return self.X.T @ expit(self.X @ th) / len(self.X)

    def info(self, th):
        p = expit(self.X @ th)
        return (self.X * (p * (1 - p))[:, None]).T @ self.X / len(self.X)

    def score_cov(self, th, c):
        """Covariance of x (y - sigmoid(x'c)) when y ~ Bernoulli(sigmoid(x'theta))."""
        pt, pc = expit(self.X @ th), expit(self.X @ c)
        E = (self.X * (pt * (1 - pt) + (pt - pc) ** 2)[:, None]).T @ self.X / len(self.X)
        d = self.m(th) - self.m(c)
        return E - np.outer(d, d)

    def invert(self, target, start, iters=60, ridge=0.0):
        """Solve m(theta) + ridge * theta = target by damped Newton steps (ridge > 0: under a Gaussian prior at 0)."""
        th = np.array(start, float)
        for _ in range(iters):
            step = np.linalg.solve(self.info(th) + ridge * np.eye(P), target - self.m(th) - ridge * th)
            big = np.abs(step).max()
            if big > 1:
                step = step / big
            th = np.clip(th + step, -8, 8)
            if big < 1e-10:
                break
        return th


def release(rng, data, n, mu0, eps, delta=1e-5):
    """Privatized mean scores at the anchor and their noise standard deviations."""
    rho = rho_total(eps, delta)
    sd = np.zeros(len(n)) if np.isinf(rho) else 2 * G / np.asarray(n, float) / np.sqrt(2 * rho)
    g = np.array([X.T @ (y - expit(X @ mu0)) / len(y) for X, y in data])
    return g + sd[:, None] * rng.standard_normal(g.shape), sd


def linearize(pub, g, sd, n, mu0, points, prior_sd=None):
    """theta~_j = a_j + I(a_j)^{-1}(g_j + m(mu_0) - m(a_j)) and its delta-method variances.

    prior_sd: if given, I(a_j) is replaced by I(a_j) + I / (n_j prior_sd^2), the information under the default prior,
    which keeps the inverse finite at extreme points (real data)."""
    th, r, m0 = np.empty_like(g), np.empty_like(g), pub.m(mu0)
    for j in range(len(n)):
        a = points[j]
        extra = 0.0 if prior_sd is None else 1.0 / (n[j] * prior_sd ** 2)
        Ii = np.linalg.inv(pub.info(a) + extra * np.eye(P))
        th[j] = a + Ii @ (g[j] + m0 - pub.m(a))
        r[j] = np.diag(Ii @ (pub.score_cov(a, mu0) / n[j] + sd[j] ** 2 * np.eye(P)) @ Ii.T)
    return th, r


def network(th, r, avg, tau2):
    """Coordinatewise gossip borrowing at a given tau^2 per coordinate: leave-one-out mean and its error."""
    K = len(th)
    mu, M = np.empty((K, P)), np.empty((K, P))
    for d in range(P):
        _, b, c = ghb.borrow(th[None, :, d], r[None, :, d], avg, tau2=tau2[d])
        mu[:, d], M[:, d] = b[0], c[0]
    return mu, M


def anchor(mu, M, mu0, anchor_sd, drop):
    """Precision-weighted combination with the anchor, except for hospitals whose decision dropped it."""
    if anchor_sd is None:
        return mu, M
    prec = 1.0 / M + 1.0 / anchor_sd ** 2
    comb, Mc = (mu / M + mu0[None, :] / anchor_sd ** 2) / prec, 1.0 / prec
    return np.where(drop[:, None], mu, comb), np.where(drop[:, None], M, Mc)


def conflicts(mu, M, mu0, anchor_sd, check):
    """Hospitals for which the anchor conflicts with the anchor-free network estimate."""
    if anchor_sd is None or not check:
        return np.zeros(len(mu), bool)
    return (np.abs(mu0[None, :] - mu) / np.sqrt(anchor_sd ** 2 + M)).max(axis=1) > Z_CONFLICT


def tau2_quantiles(th, r, q=Q):
    """(q, P) array: equally weighted posterior quantiles of tau^2 per coordinate (half-Cauchy prior on tau)."""
    out = np.empty((q, P))
    for d in range(P):
        x, rr = th[:, d], r[:, d]
        lp = []
        for t in GRID:
            V = t * t + rr
            w = 1.0 / V
            mu = (w * x).sum() / w.sum()
            lp.append(-0.5 * np.log(V).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (x - mu) ** 2).sum()
                      - np.log1p((t / SCALE) ** 2))
        lp = np.array(lp)
        p = np.exp(lp - lp.max()) * np.gradient(GRID)
        cdf = np.cumsum(p / p.sum())
        out[:, d] = GRID[np.minimum(np.searchsorted(cdf, (np.arange(q) + 0.5) / q), len(GRID) - 1)] ** 2
    return out


def combine(X, y, means, variances, prior_sd=PRIOR_SD):
    """Posterior mean of theta_k under a mixture of Gaussian priors, by Laplace approximation per component.

    prior_sd: scale of the default prior, or a tuple of scales, which are then integrated out like tau: every
    (component, scale) pair is a mixture component with equal prior weight, weighted by the hospital's own evidence."""
    sds = prior_sd if isinstance(prior_sd, tuple) else (prior_sd,)
    fits, logev = [], []
    for m, v, sd_ in [(m, v, q) for m, v in zip(means, variances) for q in sds]:
        pr = 1.0 / np.maximum(v, 1e-8) + 1.0 / sd_ ** 2
        m = (m / np.maximum(v, 1e-8)) / pr
        t = fit(X, y, m, pr)
        s = X @ t
        p = expit(s)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pr)
        logev.append(np.sum(y * s - np.logaddexp(0, s)) - 0.5 * np.sum(pr * (t - m) ** 2)
                     + 0.5 * np.sum(np.log(pr)) - 0.5 * np.linalg.slogdet(H)[1])
        fits.append(t)
    w = np.exp(np.array(logev) - max(logev))
    return (w[:, None] * np.array(fits)).sum(axis=0) / w.sum()


def invert(pub, g, sd, n, mu0, avg, anchor_sd=0.3, check=True, pool_prior_sd=None):
    """Releases on the parameter scale: (theta~, variances, anchor decisions). Post-processing only.

    pool_prior_sd: if given, the pooled starting point is the posterior mode under N(0, pool_prior_sd^2) for the
    pooled sample, and both linearizations use the information under that prior, which keeps them finite when noisy
    releases fall outside the range of the moment map (real data). None (the default) is the reported simulations.
    """
    n = np.asarray(n, float)
    ridge = 0.0 if pool_prior_sd is None else 1.0 / (n.sum() * pool_prior_sd ** 2)
    a0 = pub.invert((n[:, None] * g).sum(0) / n.sum() + pub.m(mu0), np.zeros(P), ridge=ridge)
    th0, r0 = linearize(pub, g, sd, n, mu0, np.repeat(a0[None], len(n), 0), pool_prior_sd)
    mu_l, M_l = network(th0, r0, avg, tau2_quantiles(th0, r0)[Q // 2])
    drop = conflicts(mu_l, M_l, mu0, anchor_sd, check)
    th, r = linearize(pub, g, sd, n, mu0, anchor(mu_l, M_l, mu0, anchor_sd, drop)[0], pool_prior_sd)
    return th, r, drop


def estimate(data, th, r, drop, mu0, avg, anchor_sd=0.3, oracle_tau2=None, prior_sd=PRIOR_SD):
    """GHB estimates (K, P) from inverted releases. With oracle_tau2, tau^2 is known (a single component).
    prior_sd: scale of the default prior in each hospital's final fit (PRIOR_SD as reported)."""
    t2 = tau2_quantiles(th, r) if oracle_tau2 is None else np.full((1, P), oracle_tau2)
    comps = [anchor(*network(th, r, avg, t2[q]), mu0, anchor_sd, drop) + (t2[q],) for q in range(len(t2))]
    return np.array([combine(X, y, [c[0][k] for c in comps], [c[1][k] + c[2] for c in comps], prior_sd)
                     for k, (X, y) in enumerate(data)])
