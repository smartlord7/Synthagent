"""Simulation 11: a wrong knowledge anchor (layer L3).

Logistic world of Simulation 7 (tau^2 = 0.05, epsilon = 1). The published model mu_0 differs from
the truth by N(0, e^2 I) with e from 0 to 3. GHB combines the network estimate with mu_0 by
precision weighting using the uncertainty the anchor is *stated* to have:
  honest         stated = e (floored at 0.05);
  overconfident  stated = 0.3 whatever the true error;
  + check        overconfident, but the anchor is dropped when it conflicts with the network
                 estimate (|mu_0 - mu_tilde| / sqrt(stated^2 + M) > 3 in any coordinate);
  no anchor      the network estimate alone.
The local model uses N(mu_0, I) as its prior, as in Simulation 7, so it also degrades. Each release is
the hospital's score evaluated at the anchor (ghb_logit.py), so a wrong anchor also makes the releases
noisier; that is part of what is tested. The conflict check is GHB's default in the other simulations.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import logit_world as L
import style

REPS, TAU2, EPS = 40, 0.05, 1.0
ERRS = [0.0, 0.3, 1.0, 2.0, 3.0]


def variants(e):
    return {"GHB, anchor uncertainty stated honestly": dict(anchor_sd=max(e, 0.05), check=False),
            "GHB, anchor overconfident (stated 0.3)": dict(anchor_sd=0.3, check=False),
            "GHB, overconfident + conflict check": dict(anchor_sd=0.3, check=True),
            "GHB, no anchor": dict(anchor_sd=None, check=False)}


def main():
    rng = np.random.default_rng(11)
    n = ghb.site_sizes()
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), 60)
    res = {}
    for e in ERRS:
        KL, AU, DR = L.run(rng, n, REPS, tau2=TAU2, eps=EPS, avg=avg, anchor_err=e,
                           ghb_variants=variants(e), baselines=False)
        res[e] = dict(kl={m: float(v.mean()) for m, v in KL.items()},
                      se={m: float(v.mean(axis=1).std(ddof=1) / np.sqrt(REPS)) for m, v in KL.items()},
                      dropped=float(DR["GHB, overconfident + conflict check"].mean()))
        print(e, {m: round(v, 4) for m, v in res[e]["kl"].items()}, "dropped", round(res[e]["dropped"], 2), flush=True)
    style.setup()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    cols = [style.TEAL700, style.RED700, style.AMBER600, style.BLUE600, style.GRAY400]
    names = list(variants(0.0)) + ["Local"]
    for m, c in zip(names, cols):
        ax.errorbar(ERRS, [res[e]["kl"][m] for e in ERRS], yerr=[2 * res[e]["se"][m] for e in ERRS], color=c,
                    marker="o", ms=3, lw=1.2, elinewidth=0.6, ls="--" if m == "Local" else "-",
                    label="Local, prior centred on the anchor" if m == "Local" else m.replace("GHB, ", "GHB: "))
    ax.set_yscale("log")
    ax.set_xlabel(r"Error of the published anchor (s.d. per coefficient)")
    ax.set_ylabel("Mean excess log-loss (nats)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ghb.tidy(ax)
    style.save(fig, ghb.FIGDIR / "sim11_anchor.pdf")
    json.dump(dict(errs=ERRS, res={str(k): v for k, v in res.items()}), open(ghb.RESDIR / "sim11_anchor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
