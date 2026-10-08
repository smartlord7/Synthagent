"""S2: fully plug-in GHB against local estimation and a single global model, across heterogeneity.

24 hospitals of skewed sizes (16 to 479 patients, log-normal around 100). GHB estimates tau^2 by DerSimonian-Laird on
the privatized releases, forms an inverse-variance leave-one-out mean by gossip on the ring+2-hop
graph of Figure 9 (T = 60 rounds), and shrinks the local estimate (Propositions 8.1, 8.2, 8.4).
Risks are reported relative to local estimation (v_k = sigma^2/n_k): below 1 means better.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R, T = 20000, 60
TAU2 = np.logspace(-4, 0, 17)


def methods(n, A):
    gossip = ghb.Averager(ghb.metropolis_hastings(A), T)
    return {
        "Global model (full pooling)": lambda th, x, s2, t2: ghb.full_pooling(th, n),
        "GHB, oracle $\\tau^2$, exact mean": lambda th, x, s2, t2: ghb.ghb(th, x, n, s2, tau2=t2),
        "GHB, plug-in, exact mean": lambda th, x, s2, t2: ghb.ghb(th, x, n, s2),
        f"GHB, plug-in, gossip T={T}": lambda th, x, s2, t2: ghb.ghb(th, x, n, s2, avg=gossip),
    }


def sweep(rng, n, eps, A):
    v = 1.0 / n
    out = {}
    for t2 in TAU2:
        theta, th = ghb.draw(rng, n, t2, R)
        x, s2 = ghb.release(rng, th, n, eps)
        for name, f in methods(n, A).items():
            rk, _ = ghb.risk(f(th, x, s2, t2), theta)
            out.setdefault(name, []).append(rk / v)
    return {k: np.array(vv) for k, vv in out.items()}


def main():
    rng = np.random.default_rng(2)
    n = ghb.site_sizes()
    A = ghb.topologies()["Ring + 2-hop (degree 4)"]
    res = {eps: sweep(rng, n, eps, A) for eps in (None, 1.0)}
    style.setup()
    colors = [style.RED700, style.GRAY400, style.TEAL700, style.BLUE600]
    styles = ["-", ":", "-", "--"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.38))
    panels = [(None, "mean", "No privacy: mean over hospitals"),
              (1.0, "mean", r"$\epsilon$=1: mean over hospitals"),
              (1.0, "max", r"$\epsilon$=1: worst hospital")]
    for ax, (eps, agg, title) in zip(axes, panels):
        for (name, arr), c, ls in zip(res[eps].items(), colors, styles):
            y = arr.mean(axis=1) if agg == "mean" else arr.max(axis=1)
            ax.plot(TAU2, y, color=c, ls=ls, lw=1.3, label=name)
        ax.axhline(1, color=style.SLATE800, lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Heterogeneity $\tau^2$")
        ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel(r"Risk relative to local, $R_k/v_k$")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    ghb.tidy(*axes)
    style.save(fig, ghb.FIGDIR / "s2_heterogeneity.pdf")

    summ = {}
    for eps in (None, 1.0):
        for name, arr in res[eps].items():
            summ[f"{name} | eps={eps}"] = dict(max_mean=float(arr.mean(1).max()), min_mean=float(arr.mean(1).min()),
                                                 max_worst=float(arr.max(1).max()))
    for k, v in summ.items():
        print(f"{k:55s} mean∈[{v['min_mean']:.3f},{v['max_mean']:.3f}] worst≤{v['max_worst']:.2f}")
    json.dump(dict(n=n.tolist(), tau2=TAU2.tolist(), summary=summ,
                   curves={str(e): {k: a.tolist() for k, a in r.items()} for e, r in res.items()}),
              open(ghb.RESDIR / "s2_heterogeneity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
