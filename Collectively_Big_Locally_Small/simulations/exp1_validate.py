"""S1: Monte Carlo check of Propositions 8.1 and 8.2 in the setting of Figure 8.

Left: per-site risk of local estimation, full pooling and hierarchical borrowing as a function of
n_k, simulated (markers, +/- 2 standard errors) against the closed forms (lines).
Right: simulated M of the privatized leave-one-out mean against Equation (6).
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style
from blueprint import loo_mse, risk_full_pooling, risk_hb, risk_local

K, N_OTHER, TAU2, R = 12, 200, 0.02, 40000
NK = np.unique(np.round(np.logspace(1, 4, 13)))
EPS = [None, 1.0, 0.1]


def simulate(rng):
    rows = []
    for nk in NK:
        n = np.array([nk] + [N_OTHER] * (K - 1), dtype=float)
        theta, th = ghb.draw(rng, n, TAU2, R)
        loc, loc_se = ghb.risk(th, theta)
        pool, pool_se = ghb.risk(ghb.full_pooling(th, n), theta)
        rows.append(dict(nk=nk, eps=None, method="local", sim=loc[0], se=loc_se[0], theory=float(risk_local(nk))))
        rows.append(dict(nk=nk, eps=None, method="pooling", sim=pool[0], se=pool_se[0],
                         theory=float(risk_full_pooling(nk, n[1:], TAU2))))
        for eps in EPS:
            x, s2 = ghb.release(rng, th, n, eps)
            M_th = loo_mse(n[1:], TAU2, eps=eps)
            mu_t = x[:, 1:].mean(axis=1)
            M_sim = np.mean((mu_t - 0.0) ** 2)
            est = ghb.oracle_hb(th, x, n, s2, TAU2)
            hb, hb_se = ghb.risk(est, theta)
            rows.append(dict(nk=nk, eps=eps, method="hb", sim=hb[0], se=hb_se[0],
                             theory=float(risk_hb(nk, TAU2, M_th)), M_sim=M_sim, M_theory=M_th))
    return rows


def m_grid(rng):
    """Equation (6) over a grid of site sizes and budgets."""
    out = []
    for nj in [20, 50, 100, 200, 500, 2000]:
        for eps in [0.05, 0.1, 0.3, 1.0, 3.0, None]:
            n = np.full(K - 1, float(nj))
            _, th = ghb.draw(rng, n, TAU2, R)
            x, _ = ghb.release(rng, th, n, eps)
            sim = x.mean(axis=1)
            out.append(dict(nj=nj, eps=eps, sim=float(np.mean(sim ** 2)),
                            se=float(np.std(sim ** 2) / np.sqrt(R)),
                            theory=float(loo_mse(n, TAU2, eps=eps))))
    return out


def main():
    rng = np.random.default_rng(1)
    rows, grid = simulate(rng), m_grid(rng)
    style.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(wspace=0.35))
    spec = [("local", None, style.GRAY400, "Local only", "o"),
            ("pooling", None, style.RED700, "Full pooling", "s"),
            ("hb", None, style.TEAL700, r"Hierarchical, no DP", "o"),
            ("hb", 1.0, style.BLUE600, r"Hierarchical, $\epsilon$=1", "^"),
            ("hb", 0.1, style.AMBER600, r"Hierarchical, $\epsilon$=0.1", "v")]
    for meth, eps, col, lab, mk in spec:
        rr = [r for r in rows if r["method"] == meth and r["eps"] == eps]
        nk = [r["nk"] for r in rr]
        a.plot(nk, [r["theory"] for r in rr], color=col, lw=1.2, label=lab)
        a.errorbar(nk, [r["sim"] for r in rr], yerr=[2 * r["se"] for r in rr], fmt=mk, ms=2.8,
                   color=col, mfc="white", mew=0.8, elinewidth=0.6, capsize=0)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("Hospital size $n_k$"); a.set_ylabel(r"Risk $\mathbb{E}(\tilde\theta_k-\theta_k)^2$")
    a.legend(frameon=False, fontsize=6.3, loc="lower left")
    a.set_title("Lines theory, markers simulation", fontsize=7.5)

    cols = {0.05: style.RED700, 0.1: style.AMBER600, 0.3: style.VIOLET600, 1.0: style.BLUE600,
            3.0: style.TEAL700, None: style.GRAY400}
    for eps, col in cols.items():
        g = [r for r in grid if r["eps"] == eps]
        b.errorbar([r["theory"] for r in g], [r["sim"] for r in g], yerr=[2 * r["se"] for r in g],
                   fmt="o", ms=3, color=col, mfc="white", mew=0.8, elinewidth=0.6,
                   label="no DP" if eps is None else rf"$\epsilon$={eps:g}")
    lo, hi = min(r["theory"] for r in grid) / 1.5, max(r["theory"] for r in grid) * 1.5
    b.plot([lo, hi], [lo, hi], color=style.SLATE800, lw=0.6, ls="--")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("$M$, theory"); b.set_ylabel("$M$ simulated")
    b.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=2)
    b.set_title(r"$n_j\in\{20,\dots,2000\}$", fontsize=7.5)
    ghb.FIGDIR.mkdir(parents=True, exist_ok=True); ghb.RESDIR.mkdir(parents=True, exist_ok=True)
    ghb.tidy(a, b)
    ghb.tidy(a, b)
    style.save(fig, ghb.FIGDIR / "s1_validate.pdf")

    def z(r):
        return abs(r["sim"] - r["theory"]) / r["se"]
    zs = [z(r) for r in rows] + [z(r) for r in grid]
    rel = [abs(r["sim"] / r["theory"] - 1) for r in rows + grid]
    summary = dict(n_checks=len(zs), max_abs_z=max(zs), share_within_2se=float(np.mean(np.array(zs) < 2)),
                   max_rel_err=max(rel), reps=R)
    print(summary)
    json.dump(dict(summary=summary, rows=rows, grid=grid), open(ghb.RESDIR / "s1_validate.json", "w"),
              indent=1, default=float)
    return summary


if __name__ == "__main__":
    main()
