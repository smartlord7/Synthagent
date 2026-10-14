"""Simulation 17B: GHB as released across network conditions (the conditions of Simulation 17, part B), with GHB-lite
and local estimation on the same worlds.

tau^2 = 0.05, epsilon = 1, delta = 1e-5, 24 hospitals of median size 100 on the ring with two-hop links unless
varied; one condition varied at a time: number of hospitals, median size, epsilon, delta, topology. GHB runs with the
published anchor (stated s.d. 0.3), as in Simulation 15; GHB-lite as reported.
    python3 exp17b_ghb.py            (results/sim17_ghb.json, figures/sim17_ghb_conditions.pdf)
    python3 exp17b_ghb.py --redraw
"""
import hashlib
import json
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import ghb
import ghb_logit as GL
import ghb_v4 as V4
import ghb_variants as V
import logit_world as L
import style
from exp7_logistic import P, R_X, fit
from exp17_sensitivity import CONDITIONS, gossip, kl_of

REPS = 40


def part(item):
    i, (cond, val) = item
    rng = np.random.default_rng([172, i])
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
        seed = int.from_bytes(hashlib.sha256(data[0][0].tobytes() + data[-1][1].tobytes()).digest()[:8], "little")
        cfg = dict(L.V4_CFG, anchor=(mu0, 0.3))
        est = {"GHB": V4.estimate({"net": data}, np.random.default_rng(seed), kw["eps"], kw["delta"], R_X, cfg=cfg)["net"],
               "GHB-lite": V.estimate(pub, rel, data, n, mu0, kw["eps"], avg, delta=kw["delta"]),
               "Local": np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])}
        for m, v in kl_of(rng, theta, means, est).items():
            acc.setdefault(m, []).append(v)
    acc = {m: np.array(v) for m, v in acc.items()}
    loc = acc["Local"].mean(axis=0)
    out = {m: dict(kl=float(v.mean()), rel=float(v.mean() / acc["Local"].mean()),
                   rel_se=float((v.mean(1) / acc["Local"].mean(1)).std(ddof=1) / np.sqrt(REPS)),
                   worst=float((v.mean(axis=0) / loc).max())) for m, v in acc.items()}
    print(cond, val, {m: round(o["rel"], 3) for m, o in out.items()}, flush=True)
    return f"{cond}|{val}", out


TITLES = {"K": "Number of hospitals", "median": "Median hospital size", "eps": r"Budget $\epsilon$",
          "delta": r"$\delta$", "topology": "Topology"}


def figure(B):
    style.setup()
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.0), sharey=True)
    for ax, cond in zip(axs, CONDITIONS):
        vals = CONDITIONS[cond]
        xs = np.arange(len(vals))
        for m, c, mk in (("GHB", "#7c2d12", "s"), ("GHB-lite", style.TEAL700, "o")):
            y = [B[f"{cond}|{v}"][m]["rel"] for v in vals]
            e = [2 * B[f"{cond}|{v}"][m]["rel_se"] for v in vals]
            ax.errorbar(xs, y, yerr=e, color=c, marker=mk, ms=3, lw=1, elinewidth=0.6, label=m)
        ax.plot(xs, [B[f"{cond}|{v}"]["GHB"]["worst"] for v in vals], color="#7c2d12", ls=":", marker="v", ms=2.5,
                lw=0.9, label="GHB, worst hospital")
        ax.axhline(1, color="0.5", lw=0.6, ls="--")
        lab = [("no DP" if v == np.inf else f"{v:g}") if not isinstance(v, str) else v.split(" (")[0].replace("Regional clusters + bridges", "Regional").replace("Sparse random", "Random") for v in vals]
        ax.set_xticks(xs, lab, fontsize=5.5, rotation=40, ha="right")
        ax.set_title(TITLES[cond], fontsize=6.5)
        ax.tick_params(labelsize=5.5)
    axs[0].set_ylabel("Error relative to local", fontsize=6)
    axs[0].legend(fontsize=5, frameon=False)
    style.save(fig, ghb.FIGDIR / "sim17_ghb_conditions.pdf")


def main():
    items = [(c, v) for c, vals in CONDITIONS.items() for v in vals]
    with Pool(4) as pool:
        B = dict(pool.map(part, list(enumerate(items))))
    json.dump(dict(reps=REPS, B=B), open(ghb.RESDIR / "sim17_ghb.json", "w"), indent=1)
    figure(B)


if __name__ == "__main__":
    if "--redraw" in sys.argv:
        figure(json.load(open(ghb.RESDIR / "sim17_ghb.json"))["B"])
    else:
        main()
