"""Prototype: GHB with privatized sufficient statistics (X'y/n) released under zCDP and inverted by public post-processing."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'scripts'))
import numpy as np
from scipy.special import expit
import ghb, logit_world as L, literature as LIT
from exp7_logistic import P, fit

class Pub:
    """Public feature distribution: moments m(theta)=E[sigma(x'theta) x], Fisher I, Cov(x y)."""
    def __init__(self, rng, m=20000):
        self.X = L.features(rng, m)
    def m(self, th):
        return self.X.T @ expit(self.X @ th) / len(self.X)
    def I(self, th):
        p = expit(self.X @ th); return (self.X * (p * (1 - p))[:, None]).T @ self.X / len(self.X)
    def Vxy(self, th):
        p = expit(self.X @ th); E = (self.X * p[:, None]).T @ self.X / len(self.X); mm = self.m(th); return E - np.outer(mm, mm)
    def Vscore(self, th, c):
        """Covariance of x (y - sigma(x'c)) when y ~ Bernoulli(sigma(x'theta)): the data part of a score release at c."""
        pt, pc = expit(self.X @ th), expit(self.X @ c)
        w = pt * (1 - pt) + (pt - pc) ** 2
        E = (self.X * w[:, None]).T @ self.X / len(self.X); d = self.m(th) - self.m(c)
        return E - np.outer(d, d)
    def inv(self, s, th0, iters=50):
        th = th0.copy()
        for _ in range(iters):
            step = np.linalg.solve(self.I(th), s - self.m(th))
            nrm = np.abs(step).max()
            if nrm > 1: step /= nrm
            th = np.clip(th + step, -8, 8)
            if nrm < 1e-10: break
        return th

def release_stats(rng, data, n, eps, delta=1e-5):
    rho = LIT.rho_total(eps, delta)
    sd = np.zeros(len(n)) if np.isinf(rho) else 2 * LIT.G / n / np.sqrt(2 * rho)
    s = np.array([X.T @ y / len(y) for X, y in data]) + sd[:, None] * rng.standard_normal((len(n), P))
    return s, sd

def linearize(pub, s, sd, n, points):
    """theta_j = a_j + I(a_j)^{-1}(s_j - m(a_j)) at public points a_j; variances by the delta method."""
    th = np.empty_like(s); r = np.empty_like(s)
    for j in range(len(n)):
        a = points[j]; Ii = np.linalg.inv(pub.I(a))
        th[j] = a + Ii @ (s[j] - pub.m(a))
        r[j] = np.diag(Ii @ (pub.Vxy(a) / n[j] + sd[j] ** 2 * np.eye(P)) @ Ii.T)
    return th, r

def releases2(rng, pub, data, n, eps, refine=1, avg=None, tau2="bayes"):
    s, sd = release_stats(rng, data, n, eps)
    S = (n[:, None] * s).sum(0) / n.sum()       # sample-size weighted mean statistic (gossip)
    mu0 = pub.inv(S, np.zeros(P))                 # pooled linearization point
    th, r = linearize(pub, s, sd, n, np.repeat(mu0[None], len(n), 0))
    for _ in range(refine):                       # relinearize each release at its own GHB estimate (post-processing)
        t2, mu_t, M = L.network(th, r, avg, tau2=tau2)
        a = (t2 + M) / (t2 + M + r); pts = a * th + (1 - a) * mu_t
        th, r = linearize(pub, s, sd, n, pts)
    return th, r

if __name__ == "__main__":
    n = ghb.site_sizes(); avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), 60)
    for eps in (0.3, 1.0, np.inf):
        for tau2 in (0.05, 0.5):
            for refine in (0, 1):
                rng = np.random.default_rng(101); Z = []; Zo = []
                for _ in range(40):
                    theta, means, mu0, data = L.world(rng, n, tau2)
                    pub = Pub(rng, 20000)
                    th, r = releases2(rng, pub, data, n, eps, refine, avg)
                    Z.append((th - theta) ** 2 / r)
                    rel, ro = L.releases(rng, data, n, mu0, eps)
                    Zo.append((rel - theta) ** 2 / ro)
                Z = np.array(Z).mean(0); Zo = np.array(Zo).mean(0)
                sm, lg = slice(0, 8), slice(16, 24)
                print(f"eps={eps} tau2={tau2} refine={refine}: new small {Z[sm].mean():.2f} large {Z[lg].mean():.2f} | old small {Zo[sm].mean():.2f} large {Zo[lg].mean():.2f}", flush=True)


def release_score(rng, data, n, eps, c, delta=1e-5):
    """Privatized mean score at the public point c (sensitivity 2 sqrt(5)/n), zCDP-calibrated."""
    rho = LIT.rho_total(eps, delta)
    sd = np.zeros(len(n)) if np.isinf(rho) else 2 * LIT.G / n / np.sqrt(2 * rho)
    g = np.array([X.T @ (y - expit(X @ c)) / len(y) for X, y in data]) + sd[:, None] * rng.standard_normal((len(n), P))
    return g, sd

def linearize_score(pub, g, sd, n, c, points):
    """theta_j = a_j + I(a_j)^{-1}(g_j + m(c) - m(a_j)); variance by the delta method with the score covariance."""
    th = np.empty_like(g); r = np.empty_like(g); mc = pub.m(c)
    for j in range(len(n)):
        a = points[j]; Ii = np.linalg.inv(pub.I(a))
        th[j] = a + Ii @ (g[j] + mc - pub.m(a))
        r[j] = np.diag(Ii @ (pub.Vscore(a, c) / n[j] + sd[j] ** 2 * np.eye(P)) @ Ii.T)
    return th, r
