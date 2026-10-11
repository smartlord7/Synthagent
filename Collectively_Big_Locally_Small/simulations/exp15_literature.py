"""Simulation 15: GHB against iterative baselines from the federated-learning literature (logistic heads).

The world and GHB are those of Simulation 8 (24 hospitals, ring + 2-hop agreements, T = 60 gossip
rounds, delta = 1e-5). The baselines are in literature.py: DP-FedAvg, DP-FedAvg with local
fine-tuning, decentralized DP-SGD (swarm-style, no server) and clustered DP-FedAvg (IFCA-style with a
private cluster assignment); the Ditto-style model of Simulation 8 is kept. Every private method spends
the same (epsilon, delta) per hospital; the baselines are accounted with zCDP, which is tighter than
GHB's classical Gaussian mechanism. Each baseline's hyperparameters are tuned on TUNE separate
replicates with oracle access to the excess log-loss (generous to the baselines); GHB is not tuned.

Settings: heterogeneity tau^2 in {0.01, 0.05, 0.2, 0.5} at epsilon = 1; privacy epsilon in
{0.3, 1, no DP} at tau^2 = 0.05; and a regional world (4 regions, between-region variance 0.2,
within-region 0.02), where clustering should help. Mean excess log-loss over 24 hospitals and REPS
replicates, with Monte Carlo standard errors, and the same for the 8 smallest and 8 largest hospitals.
"""
import json
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import ghb
import literature as LIT
import logit_world as L
import style

REPS, TUNE, T, DELTA = 40, 10, 60, 1e-5
FAMILIES = {
    "DP-FedAvg": lambda m: m.startswith("DP-FedAvg|") and m.endswith("|E=0"),
    "DP-FedAvg + fine-tuning": lambda m: m.startswith("DP-FedAvg|"),
    "Decentralized DP-SGD": lambda m: m.startswith("Decentralized DP-SGD|"),
    "Clustered DP-FedAvg": lambda m: m.startswith("Clustered DP-FedAvg|"),
}
SHOW = ("Local", "GHB (DP)", "Ditto-style (DP)", "DP-FedAvg", "DP-FedAvg + fine-tuning", "Decentralized DP-SGD",
        "Clustered DP-FedAvg", "Global (pooled, no DP)")
COL = {"Local": style.GRAY400, "GHB (DP)": style.TEAL700, "Ditto-style (DP)": style.BLUE600,
       "DP-FedAvg": style.AMBER600, "DP-FedAvg + fine-tuning": "#92400e", "Decentralized DP-SGD": style.VIOLET600,
       "Clustered DP-FedAvg": style.SLATE800, "Global (pooled, no DP)": style.RED700}
MK = {"Local": "o", "GHB (DP)": "o", "Ditto-style (DP)": "D", "DP-FedAvg": "<", "DP-FedAvg + fine-tuning": ">",
      "Decentralized DP-SGD": "s", "Clustered DP-FedAvg": "^", "Global (pooled, no DP)": "x"}
SETTINGS = [("tau2=0.01", dict(tau2=0.01, eps=1.0)), ("tau2=0.05", dict(tau2=0.05, eps=1.0)),
            ("tau2=0.2", dict(tau2=0.2, eps=1.0)), ("tau2=0.5", dict(tau2=0.5, eps=1.0)),
            ("eps=0.3", dict(tau2=0.05, eps=0.3)), ("eps=inf", dict(tau2=0.05, eps=np.inf)),
            ("regional", dict(tau2=0.02, eps=1.0, regions=(4, 0.2)))]


def graph():
    return ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"])


def setting(item):
    i, (label, kw) = item
    rng = np.random.default_rng([15, i])
    n, W = ghb.site_sizes(), graph()
    avg = ghb.Averager(W, T)
    extra = lambda keep: (lambda r, data, nn, mu0, eps: LIT.estimates(r, data, nn, mu0, eps, DELTA, W, keep))
    tune, _, _ = L.run(rng, n, TUNE, avg=avg, extra=extra(None), **kw)
    lam = float(L.best_ditto(tune).split("=")[-1])
    chosen = {f: min((m for m in tune if sel(m)), key=lambda m: tune[m].mean()) for f, sel in FAMILIES.items()}
    KL, AU, _ = L.run(rng, n, REPS, avg=avg, ditto_lam=lam, extra=extra(set(chosen.values())), **kw)
    ditto = [m for m in KL if m.startswith("Ditto")][0]
    KL["Ditto-style (DP)"], AU["Ditto-style (DP)"] = KL.pop(ditto), AU.pop(ditto)
    for f, m in chosen.items():
        KL[f], AU[f] = KL[m], AU[m]
    small, large = np.argsort(n)[:8], np.argsort(n)[-8:]
    loc = KL["Local"].mean(axis=0)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v)))
    out = dict(label=label, kw={k: (str(v) if k == "eps" else v) for k, v in kw.items()}, ditto_lambda=lam, chosen=chosen,
               kl={m: float(KL[m].mean()) for m in SHOW}, kl_se={m: se(KL[m].mean(axis=1)) for m in SHOW},
               kl_small={m: float(KL[m][:, small].mean()) for m in SHOW},
               kl_small_se={m: se(KL[m][:, small].mean(axis=1)) for m in SHOW},
               kl_large={m: float(KL[m][:, large].mean()) for m in SHOW},
               auc={m: float(AU[m].mean()) for m in SHOW},
               worst_ratio={m: float((KL[m].mean(axis=0) / loc).max()) for m in SHOW},
               share_better={m: float((KL[m].mean(axis=0) < loc).mean()) for m in SHOW})
    print(label, chosen, {m: round(v, 4) for m, v in out["kl"].items()}, flush=True)
    return label, out


def main():
    with Pool(min(len(SETTINGS), 7)) as pool:
        res = dict(pool.map(setting, list(enumerate(SETTINGS))))
    json.dump(dict(reps=REPS, tune=TUNE, res=res), open(ghb.RESDIR / "sim15_literature.json", "w"), indent=1)
    figure(res)


def figure(res):
    style.setup()
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.55), gridspec_kw=dict(wspace=0.32, width_ratios=[1.25, 1, 1.1]))
    tau = [0.01, 0.05, 0.2, 0.5]
    a = axs[0]
    for m in SHOW:
        y = [res[f"tau2={t}"]["kl"][m] for t in tau]
        e = [2 * res[f"tau2={t}"]["kl_se"][m] for t in tau]
        a.errorbar(tau, y, yerr=e, color=COL[m], marker=MK[m], ms=2.8, lw=1.1, elinewidth=0.6,
                   ls="--" if m.startswith("Global") else "-", label=m)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"Heterogeneity $\tau^2$ ($\epsilon$=1)")
    a.set_ylabel("Mean excess log-loss (nats)")
    a.set_title("(a) All hospitals", fontsize=7.5)
    # (b) privacy budget, smallest third of hospitals
    b = axs[1]
    eps_keys, xs = ["eps=0.3", "tau2=0.05", "eps=inf"], [0, 1, 2]
    for m in SHOW:
        b.errorbar(xs, [res[k]["kl_small"][m] for k in eps_keys], yerr=[2 * res[k]["kl_small_se"][m] for k in eps_keys],
                   color=COL[m], marker=MK[m], ms=2.8, lw=1.1, elinewidth=0.6, ls="--" if m.startswith("Global") else "-")
    b.set_xticks(xs, ["0.3", "1", "no DP"])
    b.set_yscale("log")
    b.set_xlabel(r"Privacy budget $\epsilon$ ($\tau^2$=0.05)")
    b.set_title("(b) The 8 smallest hospitals", fontsize=7.5)
    # (c) regional world: relative to local, all and smallest
    c = axs[2]
    r = res["regional"]
    order = [m for m in SHOW if m != "Local"]
    yy = np.arange(len(order))[::-1]
    for y, m in zip(yy, order):
        c.scatter(r["kl"][m] / r["kl"]["Local"], y, color=COL[m], marker=MK[m], s=16, zorder=3)
        c.scatter(r["kl_small"][m] / r["kl_small"]["Local"], y, facecolors="none", edgecolors=COL[m], marker=MK[m], s=16, zorder=3)
    c.axvline(1, color=style.SLATE800, lw=0.7)
    c.set_yticks(yy, [m.replace(" (DP)", "").replace("Global (pooled, no DP)", "Pooled, no DP") for m in order], fontsize=6)
    c.set_xscale("log")
    c.set_xlabel("Error relative to local")
    c.set_title("(c) Regional world (filled: all; open: smallest 8)", fontsize=7)
    ghb.tidy(a, b, c)
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=5.8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.13))
    style.save(fig, ghb.FIGDIR / "sim15_literature.pdf")


if __name__ == "__main__":
    main()
