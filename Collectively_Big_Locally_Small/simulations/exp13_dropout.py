"""Simulation 13: unreliable links during gossip.

Ring with two-hop links, 24 hospitals (tau^2 = 0.02, epsilon = 1). In every round each link fails
independently with probability p; the Metropolis-Hastings weights are recomputed on the links that
work in that round, so every round's matrix is still symmetric and doubly stochastic and the network
average is preserved. Left: consensus error against rounds. Right: mean risk of plug-in GHB relative
to local estimation after T rounds.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R, TAU2, EPS, SEQS = 3000, 0.02, 1.0, 30
PS = [0.0, 0.2, 0.5, 0.8]
TS = [1, 2, 5, 10, 20, 40, 60, 100, 160]
COLS = [style.TEAL700, style.BLUE600, style.AMBER600, style.RED700]


class Product:
    """Averaging by a fixed product of round matrices (one realization of link failures)."""

    def __init__(self, P):
        self.P = P

    def __call__(self, y):
        return y @ self.P.T


def round_products(rng, A, p, T):
    """Products W_t ... W_1 for t = 1..T under independent link failures."""
    K = len(A)
    iu = np.triu_indices(K, 1)
    P = np.eye(K)
    out = []
    for _ in range(T):
        keep = rng.random(len(iu[0])) >= p
        At = np.zeros((K, K))
        At[iu[0][keep], iu[1][keep]] = A[iu][keep]
        At = At + At.T
        P = ghb.metropolis_hastings(At) @ P
        out.append(P.copy())
    return out


def main():
    rng = np.random.default_rng(13)
    n = ghb.site_sizes()
    v = 1.0 / n
    A = ghb.topologies()["Ring + 2-hop (degree 4)"]
    Tmax = max(TS)
    cons, risk = {}, {}
    theta, th = ghb.draw(rng, n, TAU2, R)
    x, s2 = ghb.release(rng, th, n, EPS)
    exact = float((ghb.risk(ghb.ghb(th, x, n, s2), theta)[0] / v).mean())
    x0 = rng.standard_normal(len(n))
    for p in PS:
        errs = np.zeros(Tmax)
        rk = {t: [] for t in TS}
        for s in range(SEQS):
            prods = round_products(rng, A, p, Tmax)
            for t in range(Tmax):
                xt = prods[t] @ x0
                errs[t] += np.linalg.norm(xt - x0.mean()) / np.linalg.norm(x0 - x0.mean()) / SEQS
            if s < 6:  # risk is expensive; six failure sequences per point
                for t in TS:
                    est = ghb.ghb(th, x, n, s2, avg=Product(prods[t - 1]))
                    rk[t].append(float((ghb.risk(est, theta)[0] / v).mean()))
        cons[p] = errs.tolist()
        risk[p] = {t: float(np.mean(r)) for t, r in rk.items()}
        print(f"p={p}: risk at T=60 {risk[p][60]:.3f}, T=20 {risk[p][20]:.3f}; consensus error at 60 {errs[59]:.2e}", flush=True)
    style.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(wspace=0.3))
    for p, c in zip(PS, COLS):
        a.plot(np.arange(1, Tmax + 1), cons[p], color=c, lw=1.2, label=f"links fail with $p$={p}")
        b.plot(TS, [risk[p][t] for t in TS], "-o", ms=3, color=c, lw=1.2)
    a.set_yscale("log"); a.set_xlabel("Gossip rounds $t$"); a.set_ylabel("Relative consensus error")
    a.legend(frameon=False, fontsize=6.3, loc="lower left")
    b.axhline(exact, color=style.SLATE800, lw=0.7, ls="--", label="exact network average")
    b.axhline(1, color=style.SLATE800, lw=0.6)
    b.set_xscale("log"); b.set_xlabel("Gossip rounds $T$"); b.set_ylabel(r"Mean $R_k/v_k$")
    b.legend(frameon=False, fontsize=6.3, loc="upper right")
    ghb.tidy(a, b)
    style.save(fig, ghb.FIGDIR / "sim13_dropout.pdf")
    json.dump(dict(p=PS, T=TS, exact=exact, consensus={str(k): v for k, v in cons.items()},
                   risk={str(k): {str(t): r for t, r in v.items()} for k, v in risk.items()}),
              open(ghb.RESDIR / "sim13_dropout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
