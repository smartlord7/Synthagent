import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from dev import *

def ridge_release(rng, data, n, eps, c, lam, delta=1e-5):
    rho = LIT.rho_total(eps, delta)
    sd = np.zeros(len(n)) if np.isinf(rho) else 2 * LIT.G / (n * lam) / np.sqrt(2 * rho)
    t = np.array([fit(X, y, c, np.full(P, len(y) * lam)) for X, y in data]) + sd[:, None] * rng.standard_normal((len(n), P))
    return t, sd

def deshrink(pub, t, sd, n, c, lam, points):
    th = np.empty_like(t); r = np.empty_like(t)
    for j in range(len(n)):
        a = points[j]; I = pub.I(a); Ii = np.linalg.inv(I); A = (I + lam * np.eye(P)) @ Ii
        th[j] = t[j] + lam * Ii @ (t[j] - c)
        r[j] = np.diag(Ii / n[j] + sd[j] ** 2 * A @ A.T)
    return th, r

def two_stage(pub, rel_fn, mu0, a0):
    """Linearize at the anchor-free pooled point a0; decide, per hospital, whether the anchor conflicts with the
    anchor-free leave-one-out network estimate; relinearize at that estimate, combined with the anchor where it
    does not conflict. Returns releases, variances and the conflict decisions."""
    th0, r0 = rel_fn(np.repeat(a0[None], len(n), 0))
    _, mu_l, M_l = L.network(th0, r0, avg)
    pt, _, drop = L.with_anchor(mu_l, M_l, mu0, 0.3, check=True)
    th, r = rel_fn(pt)
    return th, r, drop

def run2(args):
    tau2, eps, shift, aerr, seed = args
    rng = np.random.default_rng([901, seed, int(tau2 * 1000), int(min(eps, 99) * 10), int(shift * 10), int(aerr * 10)])
    theta, means, mu0, data = L.world(rng, n, tau2, shift, aerr)
    pub = Pub(rng, 20000)
    est = {"Local": [fit(X, y, mu0, np.ones(P)) for X, y in data]}
    rel, r = L.releases(rng, data, n, mu0, eps)
    t2, mu_t, M = L.network(rel, r, avg); mc, Mc, _ = L.with_anchor(mu_t, M, mu0, 0.3, check=aerr > 0.5)
    est["V0"] = [fit(X, y, mc[k], 1 / np.maximum(t2[k] + Mc[k], 1e-6)) for k, (X, y) in enumerate(data)]
    g, sdg = release_score(rng, data, n, eps, mu0)
    G = (n[:, None] * g).sum(0) / n.sum()
    a0 = pub.inv(G + pub.m(mu0), np.zeros(P))  # pooled moment inversion: anchor-free
    cands = {"score@anchor": two_stage(pub, lambda pts: linearize_score(pub, g, sdg, n, mu0, pts), mu0, a0)}
    for lam, cen in ((1.0, "anchor"), (0.2, "anchor"), (0.2, "zero")):
        c = mu0 if cen == "anchor" else np.zeros(P)
        t, sdt = ridge_release(rng, data, n, eps, c, lam)
        cands[f"ridge{lam}@{cen}"] = two_stage(pub, lambda pts, t=t, sdt=sdt, c=c, lam=lam: deshrink(pub, t, sdt, n, c, lam, pts), mu0, a0)
    for name, (th, rr, drop) in cands.items():
        est[name] = fullbayes(th, rr, data, mu0, drop=drop)
    kl = {m: np.array([L.evaluate(rng, theta[k], means[k], {m: e[k]})[m][0] for k in range(len(n))]) for m, e in est.items()}
    return (tau2, eps, shift, aerr), kl

if __name__ == "__main__":
    settings = [(0.05, 0.3, 0, 0.3), (0.05, 1.0, 0, 0.3), (0.05, np.inf, 0, 0.3), (0.5, 1.0, 0, 0.3), (0.05, 1.0, 1.0, 0.3), (0.05, 1.0, 0, 2.0)]
    jobs = [s + (k,) for s in settings for k in range(20)]
    with Pool(4) as pool:
        out = pool.map(run2, jobs)
    res = {}
    for key, kl in out:
        for m, v in kl.items():
            res.setdefault(key, {}).setdefault(m, []).append(v)
    for key, d in res.items():
        print(key, "  ".join(f"{m}={np.mean(v):.4f}/w{(np.mean(v,0)/np.mean(d['Local'],0)).max():.2f}" for m, v in d.items()), flush=True)
