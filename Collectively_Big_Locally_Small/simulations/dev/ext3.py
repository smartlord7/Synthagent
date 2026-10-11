import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from ext import *
TG = np.r_[0.0, np.logspace(-2.5, 0.5, 19)]  # grid of tau_r and tau_w (sd)

def two_level(data, th, r, drop, mu0, labels, rng_s, anchor_sd=0.3, ncomp=16):
    K = len(th); G = labels.max() + 1; same = (labels[:, None] == labels[None, :]).astype(float)
    # joint posterior of (tau_r, tau_w) per coordinate, REML-type marginal likelihood, half-Cauchy(0.5) priors
    samples = []
    for d in range(P):
        x, rr = th[:, d], r[:, d]; lp = np.empty((len(TG), len(TG)))
        for i, tr in enumerate(TG):
            for j, tw in enumerate(TG):
                S = tr * tr * same + np.diag(tw * tw + rr); Si = np.linalg.inv(S); one = np.ones(K)
                a = one @ Si @ one; mu = one @ Si @ x / a; e = x - mu
                lp[i, j] = -0.5 * np.linalg.slogdet(S)[1] - 0.5 * np.log(a) - 0.5 * e @ Si @ e - np.log1p((tr / 0.5) ** 2) - np.log1p((tw / 0.5) ** 2)
        p = np.exp(lp - lp.max()) * np.outer(np.gradient(TG), np.gradient(TG)); p = p.ravel() / p.sum()
        # deterministic stratified draws from the grid posterior
        cdf = np.cumsum(p); pick = np.minimum(np.searchsorted(cdf, (np.arange(ncomp) + 0.5) / ncomp), len(p) - 1)
        samples.append([(TG[q // len(TG)], TG[q % len(TG)]) for q in pick])
    comps_m = np.empty((ncomp, K, P)); comps_v = np.empty((ncomp, K, P))
    for d in range(P):
        x, rr = th[:, d], r[:, d]
        for c, (tr, tw) in enumerate(samples[d]):
            for k in range(K):
                o = np.arange(K) != k
                S = tr * tr * same[np.ix_(o, o)] + np.diag(tw * tw + rr[o]); Si = np.linalg.inv(S); one = np.ones(K - 1)
                use_anchor = anchor_sd is not None and not drop[k]
                a = one @ Si @ one + (1 / anchor_sd ** 2 if use_anchor else 0.0)
                mu = (one @ Si @ x[o] + (mu0[d] / anchor_sd ** 2 if use_anchor else 0.0)) / a
                cvec = tr * tr * same[k, o]
                pred = mu + cvec @ Si @ (x[o] - mu)
                var = tr * tr - cvec @ Si @ cvec + (1 - one @ Si @ cvec) ** 2 / a
                comps_m[c, k, d], comps_v[c, k, d] = pred, var + tw * tw
    return np.array([GL.combine(X, y, comps_m[:, k], comps_v[:, k]) for k, (X, y) in enumerate(data)])

def run3(args):
    tau2, eps, regions, seed = args
    rng = np.random.default_rng([904, seed, int(tau2 * 1000), int(min(eps, 99) * 10), 0 if regions is None else regions[0]])
    theta, means, mu0, data = L.world(rng, n, tau2, regions=regions)
    pub = GL.Public(L.features(rng, 20000))
    est = {"Local": np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])}
    gg, sd = GL.release(rng, data, n, mu0, eps)
    th, r, drop = GL.invert(pub, gg, sd, n, mu0, avg)
    est["GHB"] = GL.estimate(data, th, r, drop, mu0, avg)
    G = 3 if regions is None else regions[0]
    est["two-level"] = two_level(data, th, r, drop, mu0, np.arange(len(n)) % G, rng)
    kl = {m: np.array([L.evaluate(rng, theta[k], means[k], {m: e[k]})[m][0] for k in range(len(n))]) for m, e in est.items()}
    return (tau2, eps, regions), kl

if __name__ == "__main__":
    jobs = [(t, e, rg, s) for t, e, rg in ((0.02, 1.0, (3, 0.1)), (0.02, 0.3, (3, 0.1)), (0.005, 1.0, (3, 0.05)), (0.05, 1.0, None), (0.05, 0.3, None), (0.5, 1.0, None)) for s in range(16)]
    with Pool(3) as pool:
        out = pool.map(run3, jobs)
    res = {}
    for key, kl in out:
        for m, v in kl.items():
            res.setdefault(key, {}).setdefault(m, []).append(v)
    for key, d in res.items():
        print(key, "  ".join(f"{m}={np.mean(v):.4f}/w{(np.mean(v,0)/np.mean(d['Local'],0)).max():.2f}" for m, v in d.items()), flush=True)
