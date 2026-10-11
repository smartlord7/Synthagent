"""Simulation 17: sensitivity of GHB to its own settings and to the conditions of the network.

Part A varies each setting of GHB (ghb_variants.py) over a wide range, the others at their reported values, in
three base settings of the logistic world (tau^2 = 0.05 at epsilon = 1 and 0.3; tau^2 = 0.2 at epsilon = 1):
  Q          number of posterior quantiles of tau (1 is the plug-in posterior median)
  scale      half-Cauchy scale of the prior on tau
  prior_sd   scale of the weakly informative default prior (None: none)
  z          anchor conflict threshold
  anchor_sd  stated uncertainty of the anchor (the true anchor error is 0.3)
  T          gossip rounds
  m          size of the public sample used for the moments of the inversion
Within a replicate every value uses the same data and releases (paired). Result: mean excess log-loss
relative to GHB with its reported settings.

Part B varies the conditions GHB does not control, at tau^2 = 0.05 and epsilon = 1 unless varied: the number of
hospitals K, the median hospital size, the privacy budget epsilon, delta, and the topology of the agreement
graph. Result: GHB and GHB with known tau^2, relative to local estimation, and the worst hospital.
"""
import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import ghb
import ghb_logit as GL
import ghb_variants as V
import logit_world as L
import style
from exp7_logistic import P, fit

REPS = 40
GRIDS = {
    "Q": [1, 3, 6, 12, 24],
    "scale": [0.1, 0.25, 0.5, 1.0, 2.5],
    "prior_sd": [1.0, 2.5, 5.0, 10.0, None],
    "z": [2.0, 2.5, 3.0, 4.0, 6.0],
    "anchor_sd": [0.1, 0.3, 0.6, 1.0, 2.0],
    "T": [5, 10, 20, 40, 60, 120],
    "m": [2000, 5000, 20000, 100000],
}
DEFAULTS = {"Q": 12, "scale": 0.5, "prior_sd": 2.5, "z": 3.0, "anchor_sd": 0.3, "T": 60, "m": 20000}
BASES = [("tau2=0.05,eps=1", dict(tau2=0.05, eps=1.0)), ("tau2=0.05,eps=0.3", dict(tau2=0.05, eps=0.3)),
         ("tau2=0.2,eps=1", dict(tau2=0.2, eps=1.0))]
CONDITIONS = {
    "K": [6, 12, 24, 48, 96],
    "median": [25, 50, 100, 200, 400],
    "eps": [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf],
    "delta": [1e-3, 1e-5, 1e-7, 1e-9],
    "topology": ["Ring (degree 2)", "Regional clusters + bridges", "Sparse random (p=0.15)", "Ring + 2-hop (degree 4)"],
}


def gossip(K, T, topo="Ring + 2-hop (degree 4)"):
    return ghb.Averager(ghb.metropolis_hastings(ghb.topologies(K)[topo]), T)


def kl_of(rng, theta, means, est):
    res = [L.evaluate(rng, theta[k], means[k], {m: e[k] for m, e in est.items()}) for k in range(len(theta))]
    return {m: np.array([res[k][m][0] for k in range(len(theta))]) for m in est}


def part_a(item):
    i, (label, kw) = item
    rng = np.random.default_rng([17, i])
    n = ghb.site_sizes()
    avgs = {T: gossip(24, T) for T in GRIDS["T"]}
    acc = {}
    for _ in range(REPS):
        theta, means, mu0, data = L.world(rng, n, kw["tau2"])
        pubs = {m: GL.Public(L.features(rng, m)) for m in GRIDS["m"]}
        rel = V.Releases(rng, data, n, mu0)
        est = {}
        for par, values in GRIDS.items():
            for val in values:
                cfg, avg, pub = {}, avgs[60], pubs[20000]
                if par == "T":
                    avg = avgs[val]
                elif par == "m":
                    pub = pubs[val]
                else:
                    cfg = {par: val}
                est[(par, str(val))] = V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, cfg)
        est[("local", "")] = np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])
        for key, v in kl_of(rng, theta, means, est).items():
            acc.setdefault(key, []).append(v.mean())
    ref = np.array(acc[("Q", "12")])
    out = {}
    for (par, val), v in acc.items():
        v = np.array(v)
        d = v / ref
        out[f"{par}|{val}"] = dict(kl=float(v.mean()), ratio=float(v.mean() / ref.mean()),
                                   ratio_se=float(d.std(ddof=1) / np.sqrt(REPS)))
    print("A", label, {k: round(o["ratio"], 3) for k, o in out.items()}, flush=True)
    return label, out


def part_b(item):
    i, (cond, val) = item
    rng = np.random.default_rng([171, i])
    kw = dict(tau2=0.05, eps=1.0, delta=1e-5)
    K, median, topo = 24, 100, "Ring + 2-hop (degree 4)"
    if cond == "K":
        K = val
    elif cond == "median":
        median = val
    elif cond == "topology":
        topo = val
    else:
        kw[cond] = val
    n = ghb.site_sizes(K=K, median=median)
    avg = gossip(K, 60, topo)
    acc = {}
    for _ in range(REPS):
        theta, means, mu0, data = L.world(rng, n, kw["tau2"])
        pub = GL.Public(L.features(rng, 20000))
        rel = V.Releases(rng, data, n, mu0)
        est = {"GHB": V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, delta=kw["delta"]),
               "GHB, tau known": V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, {"tau": "oracle"}, true_tau2=kw["tau2"], delta=kw["delta"]),
               "Local": np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])}
        for m, v in kl_of(rng, theta, means, est).items():
            acc.setdefault(m, []).append(v)
    acc = {m: np.array(v) for m, v in acc.items()}
    loc = acc["Local"].mean(axis=0)
    out = {m: dict(kl=float(v.mean()), rel=float(v.mean() / acc["Local"].mean()),
                   rel_se=float((v.mean(1) / acc["Local"].mean(1)).std(ddof=1) / np.sqrt(REPS)),
                   worst=float((v.mean(axis=0) / loc).max())) for m, v in acc.items()}
    print("B", cond, val, {m: round(o["rel"], 3) for m, o in out.items()}, flush=True)
    return f"{cond}|{val}", out


NAMES = {"Q": "Quantiles of $\\tau$, $Q$", "scale": "Half-Cauchy scale of $\\tau$", "prior_sd": "Default prior s.d.",
         "z": "Conflict threshold $z$", "anchor_sd": "Stated anchor s.d. (true 0.3)", "T": "Gossip rounds $T$",
         "m": "Public sample size"}


def figures(A, B):
    style.setup()
    fig, axs = plt.subplots(2, 4, figsize=(6.8, 3.6), gridspec_kw=dict(wspace=0.45, hspace=0.75))
    cols = [style.TEAL700, style.AMBER600, style.BLUE600]
    for ax, par in zip(axs.ravel(), GRIDS):
        vals = GRIDS[par]
        xs = np.arange(len(vals))
        for (lab, _), c in zip(BASES, cols):
            y = [A[lab][f"{par}|{v}"]["ratio"] for v in vals]
            e = [2 * A[lab][f"{par}|{v}"]["ratio_se"] for v in vals]
            ax.errorbar(xs, y, yerr=e, color=c, marker="o", ms=2.5, lw=1, elinewidth=0.6,
                        label=lab.replace("tau2", r"$\tau^2$").replace("eps", r"$\epsilon$").replace(",", ", "))
        ax.axhline(1, color=style.SLATE800, lw=0.6)
        ax.set_xticks(xs, ["none" if v is None else (f"{v:g}" if not isinstance(v, int) or v < 10000 else f"{v // 1000}k") for v in vals], fontsize=5.5)
        ax.set_title(NAMES[par], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=5.5)
    axs.ravel()[-1].axis("off")
    axs.ravel()[0].set_ylabel("Error relative to\nreported settings", fontsize=6)
    axs.ravel()[4].set_ylabel("Error relative to\nreported settings", fontsize=6)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    axs.ravel()[-1].legend(h, l, frameon=False, fontsize=6, loc="center")
    ghb.tidy(*axs.ravel()[:-1])
    style.save(fig, ghb.FIGDIR / "sim17_settings.pdf")

    fig, axs = plt.subplots(1, 5, figsize=(6.8, 1.9), gridspec_kw=dict(wspace=0.5))
    titles = {"K": "Hospitals $K$", "median": "Median size", "eps": r"Budget $\epsilon$", "delta": r"$\delta$", "topology": "Topology"}
    for ax, cond in zip(axs, CONDITIONS):
        vals = CONDITIONS[cond]
        xs = np.arange(len(vals))
        for m, c, mk in (("GHB", style.TEAL700, "o"), ("GHB, tau known", "#14b8a6", "*")):
            y = [B[f"{cond}|{v}"][m]["rel"] for v in vals]
            e = [2 * B[f"{cond}|{v}"][m]["rel_se"] for v in vals]
            ax.errorbar(xs, y, yerr=e, color=c, marker=mk, ms=3, lw=1, elinewidth=0.6, label="GHB" if m == "GHB" else r"GHB, $\tau^2$ known")
        ax.plot(xs, [B[f"{cond}|{v}"]["GHB"]["worst"] for v in vals], color=style.RED700, ls=":", marker="v", ms=2.5, lw=0.9, label="GHB, worst hospital")
        ax.axhline(1, color=style.SLATE800, lw=0.6)
        short = {"Ring (degree 2)": "ring", "Regional clusters + bridges": "regional", "Sparse random (p=0.15)": "random", "Ring + 2-hop (degree 4)": "ring+2"}
        ax.set_xticks(xs, [short.get(v, "no DP" if v == np.inf else f"{v:g}") for v in vals], fontsize=5.2, rotation=45 if cond in ("topology", "delta") else 0)
        ax.set_title(titles[cond], fontsize=6.5)
        ax.set_ylim(0, 1.3)
        ax.tick_params(axis="y", labelsize=5.5)
    axs[0].set_ylabel("Error relative to local", fontsize=6)
    axs[0].legend(frameon=False, fontsize=5.2, loc="lower left")
    ghb.tidy(*axs)
    style.save(fig, ghb.FIGDIR / "sim17_conditions.pdf")


def main():
    with Pool(4) as pool:
        A = dict(pool.map(part_a, list(enumerate(BASES))))
        items = [(c, v) for c, vals in CONDITIONS.items() for v in vals]
        B = dict(pool.map(part_b, list(enumerate(items))))
    json.dump(dict(reps=REPS, A=A, B={k: v for k, v in B.items()}), open(ghb.RESDIR / "sim17_sensitivity.json", "w"), indent=1)
    figures(A, B)


if __name__ == "__main__":
    import sys
    if "--redraw" in sys.argv:
        d = json.load(open(ghb.RESDIR / "sim17_sensitivity.json"))
        figures(d["A"], d["B"])
    else:
        main()
