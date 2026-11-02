"""Simulation 16: ablation of GHB, one component at a time.

Logistic heads (the world of Simulations 7, 8 and 15: 24 hospitals, ring with two-hop links, T = 60 gossip
rounds, delta = 1e-5). In every replicate all variants use the same data and the same standard-normal noise
draw (scaled to each variant's noise level), so differences between variants are paired. Each variant switches
one component of the reported GHB (ghb_variants.py):
  release      score at the anchor -> sufficient statistic X'y (no control variate) or a ridge fit (lambda=0.2)
  linearization  re-linearizing at the leave-one-out estimate -> pooled point only
  tau          integrated out with own evidence -> plug-in posterior mean, median or DerSimonian-Laird; or the
               components without own evidence; or known (oracle, a reference)
  anchor       with conflict check -> without the check, or no anchor at all
  default prior  N(0, 2.5^2) -> none
  network mean   inverse-variance -> equal weights
  averaging      gossip -> exact (a server, a reference)
  accounting     zCDP -> the classical Gaussian mechanism
The earlier version of GHB (logit_world.releases/network/with_anchor) and local estimation are references.
Settings: tau^2 in {0.01, 0.05, 0.2, 0.5} at epsilon = 1; epsilon in {0.3, no DP} at tau^2 = 0.05; an anchor
wrong by N(0, I); a case-mix shift of one standard deviation.

Part B repeats the ablation of the tau rule, the weights and the averaging in the Gaussian model, in closed
form over 20,000 replicates per setting.
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

REPS, T = 80, 60
VARIANTS = [
    ("GHB (reported)", {}),
    ("Release: X'y instead of the score", {"release": "xty"}),
    ("Release: ridge fit instead of the score", {"release": "ridge"}),
    ("No re-linearization", {"relinearize": False}),
    ("tau: plug-in posterior mean", {"tau": "mean"}),
    ("tau: plug-in posterior median", {"tau": "median"}),
    ("tau: plug-in DerSimonian-Laird", {"tau": "dl"}),
    ("tau: components without own evidence", {"evidence": False}),
    ("Anchor without conflict check", {"check": False}),
    ("No anchor", {"anchor_sd": None}),
    ("No default prior", {"prior_sd": None}),
    ("Equal weights in the network mean", {"weights": "equal"}),
    ("Classical Gaussian mechanism", {"accounting": "classical"}),
    ("Exact averaging (server; reference)", {"exact": True}),
    ("tau known (oracle; reference)", {"tau": "oracle"}),
]
SETTINGS = [("tau2=0.01", dict(tau2=0.01, eps=1.0)), ("tau2=0.05", dict(tau2=0.05, eps=1.0)),
            ("tau2=0.2", dict(tau2=0.2, eps=1.0)), ("tau2=0.5", dict(tau2=0.5, eps=1.0)),
            ("eps=0.3", dict(tau2=0.05, eps=0.3)), ("eps=inf", dict(tau2=0.05, eps=np.inf)),
            ("anchor=1.0", dict(tau2=0.05, eps=1.0, anchor_err=1.0)), ("shift=1.0", dict(tau2=0.05, eps=1.0, shift=1.0))]
LABELS = {"tau2=0.01": r"$\tau^2$=0.01", "tau2=0.05": r"$\tau^2$=0.05", "tau2=0.2": r"$\tau^2$=0.2",
          "tau2=0.5": r"$\tau^2$=0.5", "eps=0.3": r"$\epsilon$=0.3", "eps=inf": "no DP", "anchor=1.0": "wrong anchor",
          "shift=1.0": "case-mix shift"}


def one_rep(rng, n, avg, tau2, eps, anchor_err=0.3, shift=0.0):
    theta, means, mu0, data = L.world(rng, n, tau2, shift, anchor_err)
    pub = GL.Public(L.features(rng, 20000))
    rel = V.Releases(rng, data, n, mu0)
    est = {name: V.estimate(pub, rel, data, n, mu0, eps, avg, cfg, true_tau2=tau2) for name, cfg in VARIANTS}
    # references: the earlier version of GHB (its own releases) and local estimation
    rel0, r0 = L.releases(rng, data, n, mu0, eps)
    t2o, mu_o, M_o = L.network(rel0, r0, avg)
    mc, Mc, _ = L.with_anchor(mu_o, M_o, mu0, 0.3, True)
    est["GHB, earlier version"] = np.array([fit(X, y, mc[k], 1.0 / np.maximum(t2o[k] + Mc[k], 1e-6)) for k, (X, y) in enumerate(data)])
    est["Local"] = np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])
    res = [L.evaluate(rng, theta[k], means[k], {m: e[k] for m, e in est.items()}) for k in range(len(n))]
    return {m: np.array([res[k][m][0] for k in range(len(n))]) for m in est}


def setting(item):
    i, (label, kw) = item
    rng = np.random.default_rng([16, i])
    n = ghb.site_sizes()
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), T)
    KL = {}
    for _ in range(REPS):
        for m, v in one_rep(rng, n, avg, **kw).items():
            KL.setdefault(m, []).append(v)
    KL = {m: np.array(v) for m, v in KL.items()}  # (reps, K)
    full = KL["GHB (reported)"].mean(axis=1)
    loc = KL["Local"].mean(axis=0)
    small, large = np.argsort(n)[:8], np.argsort(n)[-8:]
    out = {}
    for m, v in KL.items():
        per = v.mean(axis=1)
        d = per - full
        out[m] = dict(kl=float(per.mean()), se=float(per.std(ddof=1) / np.sqrt(REPS)),
                      diff=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(REPS)),
                      ratio=float(per.mean() / full.mean()), worst=float((v.mean(axis=0) / loc).max()),
                      small=float(v[:, small].mean()), large=float(v[:, large].mean()))
    print(label, {m: round(o["ratio"], 3) for m, o in out.items()}, flush=True)
    return label, out


def gaussian_part():
    """Ablation of the tau rule, network weights and averaging in the Gaussian model (closed form)."""
    rng = np.random.default_rng(161)
    n = ghb.site_sizes()
    v = 1.0 / n
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), T)
    avg10 = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), 10)
    methods = {
        "tau integrated out (reported)": lambda th, x, s2, t2: ghb.ghb_full_bayes(th, x, n, s2, avg=avg),
        "tau: plug-in posterior mean": lambda th, x, s2, t2: ghb.ghb(th, x, n, s2, avg=avg, tau2="bayes"),
        "tau: plug-in DerSimonian-Laird": lambda th, x, s2, t2: ghb.ghb(th, x, n, s2, avg=avg),
        "tau: components without own evidence": lambda th, x, s2, t2: ghb.ghb_full_bayes(th, x, n, s2, avg=avg, evidence=False),
        "Equal weights in the network mean": lambda th, x, s2, t2: ghb.ghb_full_bayes(th, x, n, s2, avg=avg, weights="equal"),
        "Exact averaging (server)": lambda th, x, s2, t2: ghb.ghb_full_bayes(th, x, n, s2, avg=None),
        "Gossip with 10 rounds": lambda th, x, s2, t2: ghb.ghb_full_bayes(th, x, n, s2, avg=avg10),
        "tau known (oracle)": lambda th, x, s2, t2: ghb.ghb(th, x, n, s2, avg=avg, tau2=t2),
    }
    out = {}
    for eps in (0.3, 1.0, None):
        for t2 in (0.001, 0.02, 0.2):
            theta, th = ghb.draw(rng, n, t2, 20000)
            x, s2 = ghb.release(rng, th, n, eps)
            key = f"tau2={t2}|eps={eps}"
            out[key] = {}
            for name, f in methods.items():
                rk = ghb.risk(f(th, x, s2, t2), theta)[0] / v
                out[key][name] = dict(mean=float(rk.mean()), worst=float(rk.max()))
            print("gauss", key, {m: round(o["mean"], 3) for m, o in out[key].items()}, flush=True)
    return out


def figure(res):
    style.setup()
    rows = [m for m, _ in VARIANTS] + ["GHB, earlier version", "Local"]
    cols = [lab for lab, _ in SETTINGS]
    Z = np.array([[res[c][m]["ratio"] for c in cols] for m in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    lz = np.log2(Z)
    im = ax.imshow(lz, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            r = res[cols[j]][rows[i]]
            sig = abs(r["diff"]) > 2 * r["diff_se"]
            ax.text(j, i, f"{Z[i, j]:.2f}" + ("" if sig or i == 0 else " ·"), ha="center", va="center",
                    fontsize=5.6, color="white" if abs(lz[i, j]) > 0.6 else "black", fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(cols)), [LABELS[c] for c in cols], fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [r.replace("GHB (reported)", "GHB-lite (reported)").replace("GHB, earlier version", "GHB-lite, earlier version") for r in rows], fontsize=6)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(["0.5", "0.71", "1", "1.41", "2"], fontsize=6)
    cb.set_label("Error relative to reported GHB", fontsize=6.5)
    for s in ax.spines.values():
        s.set_visible(False)
    style.save(fig, ghb.FIGDIR / "sim16_ablation.pdf")


def main():
    with Pool(4) as pool:
        res = dict(pool.map(setting, list(enumerate(SETTINGS))))
    gauss = gaussian_part()
    json.dump(dict(reps=REPS, res=res, gaussian=gauss), open(ghb.RESDIR / "sim16_ablation.json", "w"), indent=1)
    figure(res)


if __name__ == "__main__":
    import sys
    if "--redraw" in sys.argv:
        figure(json.load(open(ghb.RESDIR / "sim16_ablation.json"))["res"])
    else:
        main()
