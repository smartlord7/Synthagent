"""Simulation 9: atypical hospitals.

Proposition 8.1 bounds the risk on average over hospitals; a hospital whose effect u_k is far from
the others can lose by borrowing. Here the 24 hospitals of Simulation 2 (tau^2 = 0.02, epsilon = 1)
have site effects that are Gaussian, Student-t with 3 degrees of freedom (same variance), or
Gaussian with one outlying hospital at u = 6 tau. We report the risk relative to local estimation as
a function of the standardized effect |u_k|/tau, for Gaussian GHB and for a robust variant that
replaces the Gaussian prior N(mu_tilde, tau^2 + M) by a Student-t prior with 3 degrees of freedom and
the same variance, and for an opt-out rule: hospital k keeps its local estimate when
|theta_hat_k - mu_tilde_k| / sqrt(v_k + tau^2 + M_k) > 3. The robust variant changes only the local, noise-free computation of each
hospital, so it has no privacy cost.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

import ghb
import style

R, TAU2, EPS, NU, CHUNK, Z_OPT = 6000, 0.02, 1.0, 3.0, 1000, 3.0
BINS = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7])


def draw_u(rng, kind, R, K):
    tau = np.sqrt(TAU2)
    if kind == "gauss":
        return tau * rng.standard_normal((R, K))
    if kind == "t3":
        return tau / np.sqrt(NU / (NU - 2)) * rng.standard_t(NU, (R, K))
    u = tau * rng.standard_normal((R, K))
    u[:, 0 if kind == "outlier_small" else K - 1] = 6 * tau  # one outlying hospital: the smallest or the largest
    return u


def robust_posterior_mean(th, v, mu_t, var_prior):
    """Posterior mean of theta under likelihood N(th; theta, v) and a t_NU prior centred at mu_t."""
    z = np.linspace(-8, 8, 321)
    grid = th[..., None] + np.sqrt(v)[None, :, None] * z  # (R, K, G)
    scale = np.sqrt(var_prior * (NU - 2) / NU)[..., None]
    logw = -0.5 * z ** 2 + student_t.logpdf((grid - mu_t[..., None]) / scale, NU)
    w = np.exp(logw - logw.max(axis=-1, keepdims=True))
    return (w * grid).sum(-1) / w.sum(-1)


def simulate(rng, n, avg, kind):
    K = len(n)
    v = 1.0 / n
    out = {"u": [], "GHB": [], "Robust GHB": [], "GHB + opt-out": [], "Local": []}
    for _ in range(R // CHUNK):
        u = draw_u(rng, kind, CHUNK, K)
        theta = u
        th = theta + np.sqrt(v) * rng.standard_normal((CHUNK, K))
        x, s2 = ghb.release(rng, th, n, EPS)
        t2, mu_t, M = ghb.borrow(x, v + s2, avg)
        alpha = (t2 + M) / (t2 + M + v)
        g = alpha * th + (1 - alpha) * mu_t
        rob = robust_posterior_mean(th, v, mu_t, t2 + M)
        out["u"].append(u); out["Local"].append((th - theta) ** 2)
        z = np.abs(th - mu_t) / np.sqrt(v + t2 + M)  # local test, uses only quantities hospital k holds
        opt = np.where(z > Z_OPT, th, g)
        out["GHB"].append((g - theta) ** 2); out["Robust GHB"].append((rob - theta) ** 2)
        out["GHB + opt-out"].append((opt - theta) ** 2)
    return {k: np.concatenate(vv) for k, vv in out.items()}


def binned(res, v):
    z = np.abs(res["u"]) / np.sqrt(TAU2)
    idx = np.digitize(z, BINS) - 1
    rows = {}
    for m in ("GHB", "Robust GHB", "GHB + opt-out"):
        rel = res[m] / v[None, :]
        y, e, c = [], [], []
        for b in range(len(BINS) - 1):
            sel = rel[idx == b]
            c.append(int(sel.size))
            y.append(float(sel.mean()) if sel.size > 30 else np.nan)
            e.append(float(2 * sel.std() / np.sqrt(sel.size)) if sel.size > 30 else np.nan)
        rows[m] = dict(y=y, e=e, n=c)
    return rows


def main():
    rng = np.random.default_rng(9)
    n = ghb.site_sizes()
    v = 1.0 / n
    avg = ghb.Averager(ghb.metropolis_hastings(ghb.topologies()["Ring + 2-hop (degree 4)"]), 60)
    out = {}
    for kind in ("gauss", "t3", "outlier_small", "outlier_large"):
        res = simulate(rng, n, avg, kind)
        summ = {m: float((res[m] / v[None, :]).mean()) for m in ("GHB", "Robust GHB", "GHB + opt-out")}
        if kind.startswith("outlier"):
            j = 0 if kind == "outlier_small" else len(n) - 1
            others = np.arange(len(n)) != j
            summ.update({f"{m} outlier": float((res[m][:, j] / v[j]).mean()) for m in ("GHB", "Robust GHB", "GHB + opt-out")})
            summ.update({f"{m} others": float((res[m][:, others] / v[None, others]).mean()) for m in ("GHB", "Robust GHB", "GHB + opt-out")})
        out[kind] = dict(summary=summ, binned=binned(res, v))
        print(kind, {k: round(x, 3) for k, x in summ.items()}, flush=True)

    style.setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1, 0.8]))
    mid = (BINS[:-1] + BINS[1:]) / 2
    for ax, kind, title in ((axes[0], "gauss", "Gaussian site effects"), (axes[1], "t3", "Heavy-tailed site effects ($t_3$)")):
        for m, c, mk in (("GHB", style.TEAL700, "o"), ("Robust GHB", style.VIOLET600, "s"), ("GHB + opt-out", style.AMBER600, "^")):
            r = out[kind]["binned"][m]
            ax.errorbar(mid, r["y"], yerr=r["e"], color=c, marker=mk, ms=3, lw=1.2, elinewidth=0.6,
                        label=m if kind == "gauss" else None)
        means = ", ".join(f"{k.replace(' GHB', '').replace('GHB + ', '')} {out[kind]['summary'][k]:.2f}" for k in ("GHB", "Robust GHB", "GHB + opt-out"))
        ax.text(0.02, 0.97, "means: " + means, transform=ax.transAxes, fontsize=5.6, va="top")
        ax.axhline(1, color=style.SLATE800, lw=0.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"Standardized site effect $|u_k|/\tau$")
        ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel(r"Risk relative to local, $R_k/v_k$")
    ax = axes[2]
    names = ["GHB", "Robust GHB", "GHB + opt-out"]
    cols = [style.TEAL700, style.VIOLET600, style.AMBER600]
    ss, sl = out["outlier_small"]["summary"], out["outlier_large"]["summary"]
    xs = np.arange(3)
    ax.bar(xs - 0.27, [ss[f"{m} outlier"] for m in names], 0.26, color=cols, label=f"outlier is smallest ($n$={int(n[0])})")
    ax.bar(xs, [sl[f"{m} outlier"] for m in names], 0.26, color=cols, alpha=0.6, hatch="////", edgecolor=style.SLATE800, lw=0, label=f"outlier is largest ($n$={int(n[-1])})")
    ax.bar(xs + 0.27, [ss[f"{m} others"] for m in names], 0.26, color=cols, alpha=0.3, label="other hospitals")
    ax.axhline(1, color=style.SLATE800, lw=0.6)
    ax.set_xticks(xs, ["GHB", "Robust", "Opt-out"])
    ax.set_yscale("log")
    ax.set_title(r"One hospital at $u=6\tau$", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.3, loc="lower center",
               bbox_to_anchor=(0.36, -0.2), ncol=3)
    ghb.tidy(*axes)
    style.save(fig, ghb.FIGDIR / "sim9_atypical.pdf")
    json.dump(dict(n=n.tolist(), bins=BINS.tolist(), out=out), open(ghb.RESDIR / "sim9_atypical.json", "w"), indent=1)


if __name__ == "__main__":
    main()
