"""S6: structured heterogeneity, exchangeable GHB against the GMRF prior of Equation (7).

24 hospitals in 4 regions of 6. theta_k = mu + r_region(k) + u_k with r ~ N(0, tau_r^2) and
u ~ N(0, tau_w^2 = 0.002). The similarity graph links hospitals in the same region. GMRF: Jacobi
iterations on the privatized releases (100 sweeps), then each hospital's final update with its own
noise-free estimate; anchor mu_0 = mu + N(0, 0.05^2) (channel P3). beta and gamma are chosen on a
grid using independent tuning replicates, then evaluated on fresh ones. epsilon = 1.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R_TUNE, R, EPS, TAU2_W, SWEEPS = 1000, 8000, 1.0, 0.002, 100
TAU2_R = np.logspace(-3.5, -0.5, 9)
BETAS = np.logspace(0, 4, 13)
GAMMAS = np.array([0.0, 1.0, 10.0, 100.0])
REGIONS = 4


def network():
    n = ghb.site_sizes()
    rng = np.random.default_rng(66)
    n = rng.permutation(n)  # sizes mixed across regions
    region = np.repeat(np.arange(REGIONS), len(n) // REGIONS)
    A = (region[:, None] == region[None, :]).astype(float) - np.eye(len(n))
    return n, region, A


def draw(rng, n, region, tau2_r, Rr):
    r = np.sqrt(tau2_r) * rng.standard_normal((Rr, REGIONS))
    theta = r[:, region] + np.sqrt(TAU2_W) * rng.standard_normal((Rr, len(n)))
    th = theta + np.sqrt(1.0 / n) * rng.standard_normal(theta.shape)
    x, s2 = ghb.release(rng, th, n, EPS)
    mu0 = 0.05 * rng.standard_normal((Rr, 1))
    return theta, th, x, s2, mu0


def gmrf(th, x, s2, mu0, n, A, beta, gamma):
    v = 1.0 / n
    r = v + s2
    d = A.sum(axis=1)
    z = x.copy()
    for _ in range(SWEEPS):
        z = (x / r + beta * z @ A.T + gamma * mu0) / (1 / r + beta * d + gamma)
    return (th / v + beta * z @ A.T + gamma * mu0) / (1 / v + beta * d + gamma)


def main():
    rng = np.random.default_rng(6)
    n, region, A = network()
    v = 1.0 / n
    out = dict(tau2_r=TAU2_R.tolist(), local=[], ghb=[], gmrf=[], gmrf_worst=[], ghb_worst=[], best=[])
    for t2 in TAU2_R:
        tune = draw(rng, n, region, t2, R_TUNE)
        best, bval = None, np.inf
        for b in BETAS:
            for g in GAMMAS:
                val = (ghb.risk(gmrf(*tune[1:], n, A, b, g), tune[0])[0] / v).mean()
                if val < bval:
                    best, bval = (float(b), float(g)), val
        theta, th, x, s2, mu0 = draw(rng, n, region, t2, R)
        rg = ghb.risk(ghb.ghb(th, x, n, s2), theta)[0] / v
        rm = ghb.risk(gmrf(th, x, s2, mu0, n, A, *best), theta)[0] / v
        out["ghb"].append(float(rg.mean())); out["gmrf"].append(float(rm.mean()))
        out["ghb_worst"].append(float(rg.max())); out["gmrf_worst"].append(float(rm.max()))
        out["best"].append(best)
        print(f"tau2_r={t2:.4f} GHB {rg.mean():.3f} (worst {rg.max():.2f})  GMRF {rm.mean():.3f} "
              f"(worst {rm.max():.2f}) beta,gamma={best}")
    style.setup()
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(TAU2_R, out["ghb"], "-o", ms=3, color=style.TEAL700, lw=1.3, label="GHB, exchangeable prior")
    ax.plot(TAU2_R, out["gmrf"], "-s", ms=3, color=style.VIOLET600, lw=1.3, label="GHB, GMRF prior (Eq. 7)")
    ax.plot(TAU2_R, out["ghb_worst"], ":", color=style.TEAL700, lw=1.0, label="worst hospital")
    ax.plot(TAU2_R, out["gmrf_worst"], ":", color=style.VIOLET600, lw=1.0)
    ax.axhline(1, color=style.SLATE800, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Between-region heterogeneity $\tau_r^2$  ($\tau_w^2$=0.002)")
    ax.set_ylabel(r"$R_k/v_k$, mean over hospitals")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    ghb.tidy(ax)
    style.save(fig, ghb.FIGDIR / "s6_gmrf.pdf")
    json.dump(dict(n=n.tolist(), region=region.tolist(), **out), open(ghb.RESDIR / "s6_gmrf.json", "w"), indent=1)


if __name__ == "__main__":
    main()
