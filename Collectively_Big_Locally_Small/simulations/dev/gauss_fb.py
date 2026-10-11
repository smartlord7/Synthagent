import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import numpy as np, ghb
GRID = np.r_[0.0, np.logspace(-3.5, 1, 160)]

def tau_post(x, r, scale=0.5):
    x = np.atleast_2d(x); r = np.broadcast_to(r, x.shape); lp = []
    for t in GRID:
        V = t * t + r; w = 1 / V; mu = (w * x).sum(1, keepdims=True) / w.sum(1, keepdims=True)
        lp.append(-0.5 * np.log(V).sum(1) - 0.5 * np.log(w.sum(1)) - 0.5 * (w * (x - mu) ** 2).sum(1) - np.log1p((t / scale) ** 2))
    lp = np.array(lp); p = np.exp(lp - lp.max(0)) * np.gradient(GRID)[:, None]; return p / p.sum(0)  # (G, R)

def ghb_fb(theta_hat, x, n, s2, avg, Q=12):
    v = 1.0 / np.asarray(n, float); r = v + s2
    p = tau_post(x, r); cdf = np.cumsum(p, 0)
    est = np.zeros_like(theta_hat); lw = []; ests = []
    for q in range(Q):
        idx = np.argmax(cdf >= (q + 0.5) / Q, axis=0); tq = GRID[idx] ** 2
        t2, mu, M = ghb.borrow(x, r, avg, tau2=np.broadcast_to(tq[:, None], x.shape))
        V = t2 + M
        lw.append(-0.5 * np.log(V + v) - 0.5 * (theta_hat - mu) ** 2 / (V + v))
        a = V / (V + v); ests.append(a * theta_hat + (1 - a) * mu)
    lw = np.array(lw); w = np.exp(lw - lw.max(0)); w /= w.sum(0)
    return (w * np.array(ests)).sum(0)

if __name__ == "__main__":
    n = ghb.site_sizes(); v = 1 / n
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), 60)
    rng = np.random.default_rng(424242)
    for eps in (None, 1.0, 0.3):
        for t2 in (1e-4, 1e-3, 0.01, 0.02, 0.05, 0.2, 1.0):
            theta, th = ghb.draw(rng, n, t2, 3000); x, s2 = ghb.release(rng, th, n, eps)
            out = {"DL": ghb.ghb(th, x, n, s2, avg=avg), "bayes-mean": ghb.ghb(th, x, n, s2, avg=avg, tau2="bayes"),
                   "fullbayes": ghb_fb(th, x, n, s2, avg), "oracle": ghb.ghb(th, x, n, s2, tau2=t2)}
            print(f"eps={eps} tau2={t2:g}: " + "  ".join(f"{k} {(ghb.risk(e, theta)[0]/v).mean():.3f}/{(ghb.risk(e, theta)[0]/v).max():.2f}" for k, e in out.items()), flush=True)
