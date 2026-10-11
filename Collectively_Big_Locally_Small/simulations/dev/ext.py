"""Prototypes of three GHB extensions on development seeds (900+)."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.special import expit
from multiprocessing import Pool
import ghb, logit_world as L, ghb_logit as GL
from exp7_logistic import P, fit

def units_consecutive(n, g, which="small"):
    """Partition hospitals (sorted by size) into release units: consecutive groups of g among those at or below
    the median size ('small') or among all hospitals ('all'); the rest release alone."""
    K = len(n); idx = np.arange(K)
    pool = idx[n <= np.median(n)] if which == "small" else idx
    units, i = [], 0
    pool = list(pool)
    while pool:
        units.append(pool[:g]); pool = pool[g:]
    rest = [k for k in idx if all(k not in u for u in units)]
    return units + [[k] for k in rest]

def unit_graph(units, A):
    U = len(units); B = np.zeros((U, U))
    where = {k: u for u, mem in enumerate(units) for k in mem}
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] and where[i] != where[j]:
                B[where[i], where[j]] = 1
    return B

def joint(rng, pub, data, n, mu0, eps, units, Wu, T=60, delta=1e-5, anchor_sd=0.3):
    rho = GL.rho_total(eps, delta); U = len(units)
    Nu = np.array([n[m].sum() for m in units], float)
    cu = np.array([((n[m] / n[m].sum()) ** 2).sum() for m in units])
    sd = np.zeros(U) if np.isinf(rho) else 2 * GL.G / Nu / np.sqrt(2 * rho)
    g = np.array([sum(X.T @ (y - expit(X @ mu0)) for X, y in (data[k] for k in m)) / Nu[u] for u, m in enumerate(units)])
    g = g + sd[:, None] * rng.standard_normal(g.shape)
    avg = ghb.Averager(Wu, T) if U > 2 else ghb.Averager()
    a0 = pub.invert((Nu[:, None] * g).sum(0) / Nu.sum() + pub.m(mu0), np.zeros(P))
    th0, r0 = GL.linearize(pub, g, sd, Nu, mu0, np.repeat(a0[None], U, 0))
    # unit-level heterogeneity tau^2 c_u: rescale so that GL's machinery (tau^2 + r) applies: divide by sqrt(c_u)
    s = np.sqrt(cu)[:, None]
    def pipeline(th, r):
        ths, rs = th / s, r / s ** 2  # theta_u / sqrt(c_u) has between-unit variance tau^2 ... approximation below
        return ths, rs
    # tau posterior with unit-specific scaling: likelihood x_u ~ N(mu, tau^2 c_u + r_u)
    def tau_q(th, r):
        out = np.empty((GL.Q, P))
        for d in range(P):
            x, rr = th[:, d], r[:, d]; lp = []
            for t in GL.GRID:
                V = t * t * cu + rr; w = 1 / V; mu = (w * x).sum() / w.sum()
                lp.append(-0.5 * np.log(V).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (x - mu) ** 2).sum() - np.log1p((t / GL.SCALE) ** 2))
            lp = np.array(lp); p = np.exp(lp - lp.max()) * np.gradient(GL.GRID); cdf = np.cumsum(p / p.sum())
            out[:, d] = GL.GRID[np.minimum(np.searchsorted(cdf, (np.arange(GL.Q) + 0.5) / GL.Q), len(GL.GRID) - 1)] ** 2
        return out
    def net(th, r, t2):
        mu, M = np.empty((U, P)), np.empty((U, P))
        for d in range(P):
            _, b, c = ghb.borrow(th[None, :, d], r[None, :, d], avg, tau2=(t2[d] * cu)[None])
            mu[:, d], M[:, d] = b[0], c[0]
        return mu, M
    tq0 = tau_q(th0, r0)
    mu_l, M_l = net(th0, r0, tq0[GL.Q // 2])
    drop = GL.conflicts(mu_l, M_l, mu0, anchor_sd, True)
    th, r = GL.linearize(pub, g, sd, Nu, mu0, GL.anchor(mu_l, M_l, mu0, anchor_sd, drop)[0])
    tq = tau_q(th, r)
    comps = [GL.anchor(*net(th, r, tq[q]), mu0, anchor_sd, drop) + (tq[q],) for q in range(GL.Q)]
    unit_of = {k: u for u, mem in enumerate(units) for k in mem}
    return np.array([GL.combine(X, y, [c[0][unit_of[k]] for c in comps], [c[1][unit_of[k]] + c[2] for c in comps])
                     for k, (X, y) in enumerate(data)])

n = ghb.site_sizes(); A = ghb.topologies()["Ring + 2-hop (degree 4)"]; W = ghb.metropolis_hastings(A); avg = ghb.Averager(W, 60)

def run(args):
    tau2, eps, seed = args
    rng = np.random.default_rng([902, seed, int(tau2 * 1000), int(min(eps, 99) * 10)])
    theta, means, mu0, data = L.world(rng, n, tau2)
    pub = GL.Public(L.features(rng, 20000))
    est = {"Local": np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])}
    gg, sd = GL.release(rng, data, n, mu0, eps)
    th, r, drop = GL.invert(pub, gg, sd, n, mu0, avg)
    est["GHB"] = GL.estimate(data, th, r, drop, mu0, avg)
    for which in ("small", "all"):
        for g in (2, 3, 4):
            units = units_consecutive(n, g, which)
            Wu = ghb.metropolis_hastings(unit_graph(units, A))
            est[f"joint {which} g={g}"] = joint(rng, pub, data, n, mu0, eps, units, Wu)
    kl = {m: np.array([L.evaluate(rng, theta[k], means[k], {m: e[k]})[m][0] for k in range(len(n))]) for m, e in est.items()}
    return (tau2, eps), kl

if __name__ == "__main__":
    jobs = [(t, e, s) for t, e in ((0.05, 0.3), (0.01, 0.3), (0.05, 1.0), (0.2, 1.0), (0.5, 0.3)) for s in range(20)]
    with Pool(3) as pool:
        out = pool.map(run, jobs)
    res = {}
    for key, kl in out:
        for m, v in kl.items():
            res.setdefault(key, {}).setdefault(m, []).append(v)
    sm = np.argsort(n)[:8]
    for key, d in res.items():
        print(key, "  ".join(f"{m}={np.mean(v):.4f}(s{np.mean(np.array(v)[:, sm]):.4f})/w{(np.mean(v,0)/np.mean(d['Local'],0)).max():.2f}" for m, v in d.items()), flush=True)
