import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from ghb2 import *
from multiprocessing import Pool
GRID = np.r_[0.0, np.logspace(-3.5, 1, 160)]
n = ghb.site_sizes(); avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), 60)

def post_tau(x, r, scale=0.5):
    lp = []
    for t in GRID:
        V = t * t + r; w = 1 / V; mu = (w * x).sum() / w.sum()
        lp.append(-0.5 * np.log(V).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (x - mu) ** 2).sum() - np.log1p((t / scale) ** 2))
    lp = np.array(lp); p = np.exp(lp - lp.max()) * np.gradient(GRID); return p / p.sum()

def borrow_all(th, r, how):
    """Per coordinate: network mean mu, its MSE M, and the heterogeneity t2 to use in the prior variance t2 + M."""
    K = len(th); mu, M, T2 = (np.empty((K, P)) for _ in range(3))
    for d in range(P):
        x, rr = th[:, d], r[:, d]
        if how == "DL":
            t2, m_, M_ = ghb.borrow(x[None], rr[None], avg)
        elif how == "mean":
            t2, m_, M_ = ghb.borrow(x[None], rr[None], avg, tau2="bayes")
        else:
            p = post_tau(x, rr)
            tt = GRID[np.searchsorted(np.cumsum(p), 0.5)] ** 2
            t2, m_, M_ = ghb.borrow(x[None], rr[None], avg, tau2=tt)
            if how == "precision":
                keep = p > 1e-5; acc = 0
                for g, pg in zip(GRID[keep], p[keep]):
                    a2, b2, c2 = ghb.borrow(x[None], rr[None], avg, tau2=g * g); acc = acc + pg / (a2[0] + c2[0])
                t2 = (p[keep].sum() / acc - M_[0])[None]
        mu[:, d], M[:, d], T2[:, d] = m_[0], M_[0], np.maximum(t2[0], 0)
    return mu, M, T2

from scipy.stats import gamma as _gamma
NU = 3.0
LAMS = _gamma.ppf((np.arange(12) + 0.5) / 12, NU / 2, scale=2 / NU)  # precision multipliers of a t_nu prior (equal weights)

def fit_t(X, y, m, prec, lams=LAMS):
    """Posterior mean under a Student-t prior written as a scale mixture of normals, by Laplace approximation per component."""
    th, lev = [], []
    for lam in lams:
        pr = prec * lam
        t = fit(X, y, m, pr); s_ = X @ t; p = expit(s_)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pr)
        ll = np.sum(y * s_ - np.logaddexp(0, s_)) - 0.5 * np.sum(pr * (t - m) ** 2) + 0.5 * np.sum(np.log(pr)) - 0.5 * np.linalg.slogdet(H)[1]
        th.append(t); lev.append(ll)
    lev = np.array(lev); w = np.exp(lev - lev.max()); w /= w.sum()
    return (w[:, None] * np.array(th)).sum(0)

Q = 12

def tau_quantiles(th, r):
    """Per coordinate: Q quantiles of the posterior of tau^2 from the releases (equal weights)."""
    out = np.empty((Q, P))
    for d in range(P):
        p = post_tau(th[:, d], r[:, d]); cdf = np.cumsum(p)
        out[:, d] = GRID[np.minimum(np.searchsorted(cdf, (np.arange(Q) + 0.5) / Q), len(GRID) - 1)] ** 2
    return out

def fit_fullbayes(X, y, m_list, var_list):
    """Posterior mean of theta_k averaging over tau (components: posterior quantiles of tau), weights from the
    network posterior (equal) times the Laplace evidence of the hospital's own data."""
    th, lev = [], []
    for m, v in zip(m_list, var_list):
        # combine the component prior N(m, v) with the weakly informative default N(0, 2.5^2) (Gelman et al., 2008)
        pr0 = 1 / 2.5 ** 2
        pr = 1 / np.maximum(v, 1e-6) + pr0
        m = (m / np.maximum(v, 1e-6)) / pr
        t = fit(X, y, m, pr); s_ = X @ t; pp = expit(s_)
        H = (X * (pp * (1 - pp))[:, None]).T @ X + np.diag(pr)
        ll = np.sum(y * s_ - np.logaddexp(0, s_)) - 0.5 * np.sum(pr * (t - m) ** 2) + 0.5 * np.sum(np.log(pr)) - 0.5 * np.linalg.slogdet(H)[1]
        th.append(t); lev.append(ll)
    lev = np.array(lev); w = np.exp(lev - lev.max()); w /= w.sum()
    return (w[:, None] * np.array(th)).sum(0)

LAM4 = _gamma.ppf((np.arange(4) + 0.5) / 4, NU / 2, scale=2 / NU)

def fullbayes(th, rr, data, mu0, tail=False, check=True, drop=None):
    tq = tau_quantiles(th, rr)
    def net(tau2_vec):
        t2, mu, M = (np.empty((len(n), P)) for _ in range(3))
        for d in range(P):
            a, b, c = ghb.borrow(th[None, :, d], rr[None, :, d], avg, tau2=tau2_vec[d])
            t2[:, d], mu[:, d], M[:, d] = a[0], b[0], c[0]
        return t2, mu, M
    # one conflict decision per hospital, at the posterior median of tau
    if drop is None:
        _, mu_med, M_med = net(tq[Q // 2])
        _, _, drop = L.with_anchor(mu_med, M_med, mu0, 0.3, check=check)
    comps_m, comps_v = [], []
    for q in range(Q):
        t2, mu, M = net(tq[q])
        mc, Mc, _ = L.with_anchor(mu, M, mu0, 0.3)
        mc = np.where(drop[:, None], mu, mc); Mc = np.where(drop[:, None], M, Mc)
        comps_m.append(mc); comps_v.append(t2 + Mc)
    return [fit_fullbayes(X, y, [cm[k] for cm in comps_m], [cv[k] for cv in comps_v]) for k, (X, y) in enumerate(data)]

def run(args):
    tau2, eps, seed = args
    rng = np.random.default_rng([900, seed, int(tau2 * 1000), int(min(eps, 99) * 10)])
    theta, means, mu0, data = L.world(rng, n, tau2)
    pub = Pub(rng, 20000)
    est = {"Local": [fit(X, y, mu0, np.ones(P)) for X, y in data]}
    # V0: current GHB
    rel, r = L.releases(rng, data, n, mu0, eps if np.isfinite(eps) else np.inf)
    t2, mu_t, M = L.network(rel, r, avg)
    mc, Mc, _ = L.with_anchor(mu_t, M, mu0, 0.3)
    est["V0 ridge+mean"] = [fit(X, y, mc[k], 1 / np.maximum(t2[k] + Mc[k], 1e-6)) for k, (X, y) in enumerate(data)]

    s, sd = release_stats(rng, data, n, eps)
    S = (n[:, None] * s).sum(0) / n.sum(); a0 = pub.inv(S, np.zeros(P))
    th0, r0 = linearize(pub, s, sd, n, np.repeat(a0[None], len(n), 0))
    # LOO relinearization point: network estimate excluding own release (post-processing, independent of own noise)
    _, mu_l, M_l = L.network(th0, r0, avg)
    mu_l, _, _ = L.with_anchor(mu_l, M_l, mu0, 0.3)
    th1, r1 = linearize(pub, s, sd, n, mu_l)
    g, sdg = release_score(rng, data, n, eps, mu0)
    thg0, rg0 = linearize_score(pub, g, sdg, n, mu0, np.repeat(mu0[None], len(n), 0))
    _, mu_g, M_g = L.network(thg0, rg0, avg); mu_g, _, _ = L.with_anchor(mu_g, M_g, mu0, 0.3)
    thg1, rg1 = linearize_score(pub, g, sdg, n, mu0, mu_g)
    for tag, (th, rr) in {"score+LOO": (thg1, rg1)}.items():
        est[f"{tag}+fullbayes"] = fullbayes(th, rr, data, mu0)
        est[f"{tag}+fullbayes+t"] = fullbayes(th, rr, data, mu0, tail=True)
        for how in ("mean",):
            mu, Mm, T2 = borrow_all(th, rr, how)
            mc2, Mc2, _ = L.with_anchor(mu, Mm, mu0, 0.3)
            est[f"{tag}+{how}"] = [fit(X, y, mc2[k], 1 / np.maximum(T2[k] + Mc2[k], 1e-6)) for k, (X, y) in enumerate(data)]
            if how == "mean":
                est[f"{tag}+{how}+t"] = [fit_t(X, y, mc2[k], 1 / np.maximum(T2[k] + Mc2[k], 1e-6)) for k, (X, y) in enumerate(data)]
    kl = {}
    for m, e in est.items():
        kl[m] = np.array([L.evaluate(rng, theta[k], means[k], {m: e[k]})[m][0] for k in range(len(n))])
    return (tau2, eps), kl

if __name__ == "__main__":
    jobs = [(t, e, s) for t in (0.01, 0.05, 0.2, 0.5) for e in (0.3, 1.0, np.inf) for s in range(24)]
    with Pool(4) as pool:
        out = pool.map(run, jobs)
    res = {}
    for key, kl in out:
        for m, v in kl.items():
            res.setdefault(key, {}).setdefault(m, []).append(v)
    for key, d in res.items():
        loc = np.mean(d["Local"])
        print(key, "\n   ".join(f"{m}={np.mean(v):.4f}/w{(np.mean(v,0)/np.mean(d['Local'],0)).max():.2f}" for m, v in d.items()), flush=True)
