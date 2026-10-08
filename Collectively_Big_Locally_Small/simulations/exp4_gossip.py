"""S4: how many gossip rounds GHB needs on each topology of Figure 9 (Proposition 8.4).

Mean risk relative to local estimation of plug-in GHB as a function of the number of gossip
rounds T, for 24 hospitals at tau^2 = 0.02 and epsilon = 1; dashed: exact network average.
Markers: the predicted number of rounds log(1/gamma)/(1-lambda) for gamma = 0.01.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R, TAU2, EPS = 5000, 0.02, 1.0
TS = np.unique(np.round(np.logspace(0, np.log10(400), 28)).astype(int))
COLORS = [style.GRAY400, style.AMBER600, style.BLUE600, style.TEAL700]


def main():
    rng = np.random.default_rng(4)
    n = ghb.site_sizes()
    v = 1.0 / n
    theta, th = ghb.draw(rng, n, TAU2, R)
    x, s2 = ghb.release(rng, th, n, EPS)
    exact = float((ghb.risk(ghb.ghb(th, x, n, s2), theta)[0] / v).mean())
    local = 1.0
    res = {}
    for name, A in ghb.topologies().items():
        W = ghb.metropolis_hastings(A)
        lam = ghb.slem(W)
        curve = [float((ghb.risk(ghb.ghb(th, x, n, s2, avg=ghb.Averager(W, t)), theta)[0] / v).mean())
                 for t in TS]
        res[name] = dict(lam=lam, curve=curve, t_pred=float(np.log(100) / (1 - lam)))
    style.setup()
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for (name, r), c in zip(res.items(), COLORS):
        ax.plot(TS, r["curve"], color=c, lw=1.3, label=f"{name}, $\\lambda$={r['lam']:.3f}")
        tp = r["t_pred"]
        ax.plot([tp], [np.interp(tp, TS, r["curve"])], "o", color=c, ms=3.5)
    ax.axhline(exact, color=style.SLATE800, lw=0.7, ls="--", label="exact network average")
    ax.axhline(local, color=style.SLATE800, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Gossip rounds $T$"); ax.set_ylabel(r"Mean $R_k/v_k$")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    style.save(fig, ghb.FIGDIR / "s4_gossip.pdf")
    # rounds needed to come within 2% of the exact-average risk
    for name, r in res.items():
        c = np.array(r["curve"])
        ok = np.where(c <= exact * 1.02)[0]
        r["t_within_2pct"] = int(TS[ok[0]]) if len(ok) else None
        print(f"{name:30s} lambda={r['lam']:.3f} predicted T={r['t_pred']:.0f} within 2%: T={r['t_within_2pct']}")
    json.dump(dict(T=TS.tolist(), exact=exact, res=res), open(ghb.RESDIR / "s4_gossip.json", "w"), indent=1)


if __name__ == "__main__":
    main()
