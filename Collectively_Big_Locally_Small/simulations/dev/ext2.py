import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from ext import *

def two_round(rng, pub, data, n, mu0, eps, frac=0.5, delta=1e-5, anchor_sd=0.3):
    """Round 1: score at mu0 with budget frac*rho; round 2: score at each hospital's public network point a_k
    with the rest. Scores are combined on the m(theta) scale by noise precision and linearized at a_k."""
    rho = GL.rho_total(eps, delta); K = len(n)
    s1 = np.zeros(K) if np.isinf(rho) else 2 * GL.G / n / np.sqrt(2 * rho * frac)
    s2 = np.zeros(K) if np.isinf(rho) else 2 * GL.G / n / np.sqrt(2 * rho * (1 - frac))
    g1 = np.array([X.T @ (y - expit(X @ mu0)) / len(y) for X, y in data]) + s1[:, None] * rng.standard_normal((K, P))
    th, r, drop = GL.invert(pub, g1, s1, n, mu0, avg)
    mu_l, M_l = GL.network(th, r, avg, GL.tau2_quantiles(th, r)[GL.Q // 2])
    a = GL.anchor(mu_l, M_l, mu0, anchor_sd, drop)[0]          # public points
    g2 = np.array([X.T @ (y - expit(X @ a[k])) / len(y) for k, (X, y) in enumerate(data)]) + s2[:, None] * rng.standard_normal((K, P))
    m0 = pub.m(mu0); thc, rc = np.empty((K, P)), np.empty((K, P))
    for k in range(K):
        if np.isinf(rho):
            w1, w2 = 0.0, 1.0
        else:
            w1, w2 = 1 / s1[k] ** 2, 1 / s2[k] ** 2; w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
        mk = pub.m(a[k]); target = w1 * (g1[k] + m0) + w2 * (g2[k] + mk)
        Ii = np.linalg.inv(pub.info(a[k]))
        thc[k] = a[k] + Ii @ (target - mk)
        V = w1 * pub.score_cov(a[k], mu0) + w2 * pub.score_cov(a[k], a[k])
        noise = (w1 * s1[k]) ** 2 + (w2 * s2[k]) ** 2
        rc[k] = np.diag(Ii @ (V / n[k] + noise * np.eye(P)) @ Ii.T)
    return GL.estimate(data, thc, rc, drop, mu0, avg, anchor_sd)

def run2(args):
    tau2, eps, seed = args
    rng = np.random.default_rng([903, seed, int(tau2 * 1000), int(min(eps, 99) * 10)])
    theta, means, mu0, data = L.world(rng, n, tau2)
    pub = GL.Public(L.features(rng, 20000))
    est = {"Local": np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])}
    gg, sd = GL.release(rng, data, n, mu0, eps)
    th, r, drop = GL.invert(pub, gg, sd, n, mu0, avg)
    est["GHB"] = GL.estimate(data, th, r, drop, mu0, avg)
    for frac in (0.3, 0.5):
        est[f"2 rounds f={frac}"] = two_round(rng, pub, data, n, mu0, eps, frac)
    kl = {m: np.array([L.evaluate(rng, theta[k], means[k], {m: e[k]})[m][0] for k in range(len(n))]) for m, e in est.items()}
    return (tau2, eps), kl

if __name__ == "__main__":
    jobs = [(t, e, s) for t, e in ((0.05, 0.3), (0.01, 0.3), (0.05, 1.0), (0.5, 1.0), (0.5, np.inf), (0.2, np.inf)) for s in range(20)]
    with Pool(3) as pool:
        out = pool.map(run2, jobs)
    res = {}
    for key, kl in out:
        for m, v in kl.items():
            res.setdefault(key, {}).setdefault(m, []).append(v)
    for key, d in res.items():
        print(key, "  ".join(f"{m}={np.mean(v):.4f}/w{(np.mean(v,0)/np.mean(d['Local'],0)).max():.2f}" for m, v in d.items()), flush=True)
