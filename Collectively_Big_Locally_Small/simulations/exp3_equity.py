"""S3: who gains (requirement R4) and how many hospitals the plug-in needs (Remark 8.3).

Left: effective sample size n_eff = sigma^2 / R_k relative to n_k for each of the 24 hospitals of
S2 at tau^2 = 0.02 and epsilon = 1, for plug-in GHB with gossip and for a single global model;
the line is n_k + B from Equation (5) with the oracle M.
Right: mean relative risk as the number of hospitals K grows: oracle tau^2, DerSimonian-Laird, and
the posterior mean of tau^2 under a half-Cauchy(0.5) prior on tau.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R, T, TAU2, EPS = 20000, 60, 0.02, 1.0
KS = [3, 4, 6, 8, 12, 16, 24, 48]


def main():
    rng = np.random.default_rng(3)
    n = ghb.site_sizes()
    A = ghb.topologies()["Ring + 2-hop (degree 4)"]
    gossip = ghb.Averager(ghb.metropolis_hastings(A), T)
    theta, th = ghb.draw(rng, n, TAU2, R)
    x, s2 = ghb.release(rng, th, n, EPS)
    r_ghb, se_ghb = ghb.risk(ghb.ghb(th, x, n, s2, avg=gossip), theta)
    r_pool, _ = ghb.risk(ghb.full_pooling(th, n), theta)
    tot = 1.0 / (TAU2 + 1.0 / n + s2)
    M_or = 1.0 / (tot.sum() - tot)
    B = 1.0 / (TAU2 + M_or)

    curves = {}
    for K in KS:
        nK = ghb.site_sizes(K=K, seed=100 + K)
        th_, h_ = ghb.draw(rng, nK, TAU2, R)
        x_, s_ = ghb.release(rng, h_, nK, EPS)
        v = 1.0 / nK
        for name, est in [("oracle", ghb.ghb(h_, x_, nK, s_, tau2=TAU2)), ("plug-in", ghb.ghb(h_, x_, nK, s_)),
                          ("bayes", ghb.ghb(h_, x_, nK, s_, tau2="bayes"))]:
            rk, _ = ghb.risk(est, th_)
            curves.setdefault(name, []).append(float((rk / v).mean()))

    style.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(wspace=0.32))
    a.plot(n, (n + B) / n, color=style.TEAL700, lw=1.0, label=r"Theory $1+B/n_k$ (Eq. 5)")
    a.errorbar(n, (1.0 / r_ghb) / n, yerr=2 * se_ghb / r_ghb ** 2 / n, fmt="o", ms=3, color=style.TEAL700,
               mfc="white", mew=0.8, elinewidth=0.6, label=rf"GHB plug-in, gossip, $\epsilon$=1")
    a.plot(n, (1.0 / r_pool) / n, "s", ms=3, color=style.RED700, mfc="white", mew=0.8,
           label="Global model (full pooling)")
    a.axhline(1, color=style.SLATE800, lw=0.6)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_yticks([0.2, 0.5, 1, 2, 4]); a.set_yticklabels(["0.2", "0.5", "1", "2", "4"])
    a.set_xticks([20, 50, 100, 200, 500]); a.set_xticklabels(["20", "50", "100", "200", "500"])
    a.minorticks_off()
    a.set_xlabel("Hospital size $n_k$"); a.set_ylabel(r"$n^{\mathrm{eff}}_k / n_k$")
    a.set_title(r"Each hospital, $\tau^2$=0.02", fontsize=7.5)
    a.legend(frameon=False, fontsize=6.3, loc="lower left")

    b.plot(KS, curves["oracle"], "-o", ms=3, color=style.GRAY400, lw=1.2, label=r"oracle $\tau^2$")
    b.plot(KS, curves["plug-in"], "-o", ms=3, color=style.TEAL700, lw=1.2, label=r"DerSimonian-Laird $\hat\tau^2$")
    b.plot(KS, curves["bayes"], "--o", ms=3, color=style.VIOLET600, lw=1.2,
           label=r"posterior mean, half-Cauchy prior on $\tau$")
    b.axhline(1, color=style.SLATE800, lw=0.6)
    b.set_xscale("log"); b.set_xticks(KS); b.set_xticklabels([str(k) for k in KS]); b.minorticks_off()
    b.set_xlabel("Number of hospitals $K$"); b.set_ylabel(r"Mean $R_k/v_k$")
    b.set_title(r"Estimating $\tau^2$ with few hospitals, $\epsilon$=1", fontsize=7.5)
    b.legend(frameon=False, fontsize=6.3)
    style.save(fig, ghb.FIGDIR / "s3_equity.pdf")

    small, large = n <= 60, n >= 300
    summ = dict(B=float(B.mean()), gain_small=float(((1 / r_ghb) / n)[small].mean()),
                gain_large=float(((1 / r_ghb) / n)[large].mean()),
                pool_gain_large=float(((1 / r_pool) / n)[large].mean()),
                pool_gain_small=float(((1 / r_pool) / n)[small].mean()),
                min_gain=float(((1 / r_ghb) / n).min()), K=KS, curves=curves)
    print(summ)
    json.dump(dict(n=n.tolist(), summary=summ), open(ghb.RESDIR / "s3_equity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
