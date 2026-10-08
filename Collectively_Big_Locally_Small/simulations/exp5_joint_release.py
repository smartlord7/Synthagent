"""S5: joint release by small hospitals through secure aggregation (Section 8.3, second consequence).

Two networks: 16 hospitals of 20 patients with 8 of 200 (left), and 24 hospitals of 20 (right). Small hospitals release either individually or in
groups of g whose sum is computed by secure aggregation, so that the Gaussian-mechanism noise is
calibrated to the group mean (sensitivity 2C/N_G). The population estimate is the inverse-variance
weighted mean of all releases. Lines: closed form; markers: simulation. We report the borrowed
patients B = sigma^2/(tau^2 + M) of Equation (5), which every hospital receives.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style
from blueprint import dp_noise_var

R, TAU2 = 20000, 0.02
N_SMALL, N_LARGE = 20, 200
NETWORKS = {"mixed": (16, 8), "small": (24, 0)}
K_SMALL, K_LARGE = NETWORKS["mixed"]
EPS = np.logspace(-1.5, 0.5, 9)
GROUPS = [1, 2, 4, 8]
COLORS = {1: style.RED700, 2: style.AMBER600, 4: style.BLUE600, 8: style.TEAL700}


def design(g):
    """Release units: (sizes of members) for each release."""
    units = [[N_SMALL] * g for _ in range(K_SMALL // g)] + [[N_LARGE] for _ in range(K_LARGE)]
    return [np.array(u, dtype=float) for u in units]


def theory_M(g, eps):
    var = []
    for u in design(g):
        N = u.sum()
        var.append(TAU2 * (u ** 2).sum() / N ** 2 + 1.0 / N + float(dp_noise_var(N, eps)))
    return 1.0 / np.sum(1.0 / np.array(var))


def simulate_M(rng, g, eps):
    units = design(g)
    n = np.concatenate(units)
    _, th = ghb.draw(rng, n, TAU2, R)
    xs, var, i = [], [], 0
    for u in units:
        N = u.sum()
        m = (th[:, i:i + len(u)] * u).sum(axis=1) / N
        i += len(u)
        s2 = float(dp_noise_var(N, eps))
        xs.append(m + np.sqrt(s2) * rng.standard_normal(R))
        var.append(TAU2 * (u ** 2).sum() / N ** 2 + 1.0 / N + s2)
    w = 1.0 / np.array(var)
    est = (np.array(xs).T * w).sum(axis=1) / w.sum()
    se = est ** 2
    return float(se.mean()), float(se.std() / np.sqrt(R))


def run(rng):
    res = {}
    for g in GROUPS:
        th = [float(theory_M(g, e)) for e in EPS]
        sim = [simulate_M(rng, g, e) for e in EPS]
        res[g] = dict(theory=th, sim=[s[0] for s in sim], se=[s[1] for s in sim])
    return res


def panel(ax, res, title):
    for g in GROUPS:
        r = res[g]
        B = 1.0 / (TAU2 + np.array(r["theory"]))
        Bs = 1.0 / (TAU2 + np.array(r["sim"]))
        lab = "individual release" if g == 1 else f"groups of {g} (secure aggregation)"
        ax.plot(EPS, B, color=COLORS[g], lw=1.3, label=lab)
        ax.plot(EPS, Bs, "o", ms=3, color=COLORS[g], mfc="white", mew=0.8)
    ax.axhline(1 / TAU2, color=style.SLATE800, lw=0.6, ls="--")
    ax.text(EPS[0], 1 / TAU2 * 1.04, r"limit $\sigma^2/\tau^2$", fontsize=6.5, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Privacy budget $\epsilon$")
    ax.set_title(title, fontsize=7.5)


def main():
    global K_SMALL, K_LARGE
    rng = np.random.default_rng(5)
    style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(wspace=0.25))
    out = {}
    titles = {"mixed": "16 hospitals of 20 patients, 8 of 200", "small": "24 hospitals of 20 patients"}
    for ax, (net, (ks, kl)) in zip(axes, NETWORKS.items()):
        K_SMALL, K_LARGE = ks, kl
        res = run(rng)
        panel(ax, res, titles[net])
        i01, i1 = int(np.argmin(abs(EPS - 0.1))), int(np.argmin(abs(EPS - 1.0)))
        summ = {g: dict(B_eps01=1 / (TAU2 + res[g]["theory"][i01]), B_eps1=1 / (TAU2 + res[g]["theory"][i1]))
                for g in GROUPS}
        z = max(abs(s - t) / e for g in GROUPS for s, t, e in zip(res[g]["sim"], res[g]["theory"], res[g]["se"]))
        out[net] = dict(res=res, summary=summ, max_abs_z=z)
        print(net, {g: {k: round(v, 2) for k, v in d.items()} for g, d in summ.items()}, "max|z|", round(z, 2))
    axes[0].set_ylabel("Borrowed patients $B$")
    axes[1].legend(frameon=False, fontsize=6.3, loc="lower right")
    ghb.tidy(*axes)
    style.save(fig, ghb.FIGDIR / "s5_joint_release.pdf")
    json.dump(dict(eps=EPS.tolist(), out=out), open(ghb.RESDIR / "s5_joint_release.json", "w"), indent=1)


if __name__ == "__main__":
    main()
