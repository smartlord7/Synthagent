"""Simulations 8 and 10: GHB against federated and meta-analytic baselines (logistic heads).

Simulation 8 varies the coefficient heterogeneity tau^2; Simulation 10 fixes tau^2 = 0.05 and
shifts each hospital's case mix (feature means m_k ~ N(0, shift^2 I)). All private methods use the
same releases (epsilon = 1): one-shot federated averaging (n-weighted mean of the releases), the
random-effects meta-analytic mean (one model for all hospitals), a Ditto-style personalized model
(local fit with a proximal term lambda/2 ||theta - theta_FedAvg||^2, lambda tuned on separate
replicates from {0.3, ..., 100}) and GHB. The pooled non-private model is a reference that no
private method can reach. Mean excess log-loss over 24 hospitals and REPS replicates.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import logit_world as L
import style

REPS, TUNE, EPS, T = 40, 10, 1.0, 60
TAU2S = [0.01, 0.05, 0.2, 0.5]
SHIFTS = [0.0, 0.25, 0.5, 1.0]
COL = {"Local": style.GRAY400, "Global (pooled, no DP)": style.RED700, "FedAvg (DP)": style.AMBER600,
       "Meta-analysis (DP)": style.VIOLET600, "Ditto-style (DP)": style.BLUE600, "GHB (DP)": style.TEAL700}
LS = {"Global (pooled, no DP)": "--"}


def setting(rng, n, avg, **kw):
    tune, _, _ = L.run(rng, n, TUNE, eps=EPS, avg=avg, **kw)
    lam = float(L.best_ditto(tune).split("=")[-1])
    KL, AU, _ = L.run(rng, n, REPS, eps=EPS, avg=avg, ditto_lam=lam, **kw)
    ditto = [m for m in KL if m.startswith("Ditto")][0]
    KL["Ditto-style (DP)"], AU["Ditto-style (DP)"] = KL.pop(ditto), AU.pop(ditto)
    loc = KL["Local"].mean(axis=0)
    return dict(lam=lam, kl={m: float(v.mean()) for m, v in KL.items()},
                kl_se={m: float(v.mean(axis=1).std(ddof=1) / np.sqrt(REPS)) for m, v in KL.items()},
                auc={m: float(v.mean()) for m, v in AU.items()},
                worst_ratio={m: float((v.mean(axis=0) / loc).max()) for m, v in KL.items()},
                share_better={m: float((v.mean(axis=0) < loc).mean()) for m, v in KL.items()})


def main():
    rng = np.random.default_rng(8)
    n = ghb.site_sizes()
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), T)
    het = {t: setting(rng, n, avg, tau2=t) for t in TAU2S}
    for t, r in het.items():
        print("tau2", t, "lambda", r["lam"], {m: round(v, 4) for m, v in r["kl"].items()}, flush=True)
    shift = {s: (het[0.05] if s == 0.0 else setting(rng, n, avg, tau2=0.05, shift=s)) for s in SHIFTS}
    for s, r in shift.items():
        print("shift", s, "lambda", r["lam"], {m: round(v, 4) for m, v in r["kl"].items()}, flush=True)

    style.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(wspace=0.3))
    for ax, res, xs, xl, xlog in ((a, het, TAU2S, r"Heterogeneity $\tau^2$ (per coefficient)", True),
                                   (b, shift, SHIFTS, "Case-mix shift between hospitals (s.d. of feature means)", False)):
        for m, c in COL.items():
            y = [res[x]["kl"][m] for x in xs]
            e = [2 * res[x]["kl_se"][m] for x in xs]
            ax.errorbar(xs, y, yerr=e, color=c, ls=LS.get(m, "-"), marker="o", ms=2.8, lw=1.2, elinewidth=0.6,
                        label=m)
        if xlog:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
    a.set_ylabel("Mean excess log-loss (nats)")
    a.set_title(r"Simulation 8: baselines, $\epsilon$=1", fontsize=7.5)
    b.set_title(r"Simulation 10: shifted case mix, $\tau^2$=0.05", fontsize=7.5)
    b.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
    ghb.tidy(a, b)
    style.save(fig, ghb.FIGDIR / "sim8_baselines.pdf")
    json.dump(dict(n=n.tolist(), het={str(k): v for k, v in het.items()}, shift={str(k): v for k, v in shift.items()}),
              open(ghb.RESDIR / "sim8_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
