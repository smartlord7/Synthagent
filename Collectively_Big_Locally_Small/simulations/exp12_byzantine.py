"""Simulation 12: dishonest hospitals.

The 24 hospitals of Simulation 2 (tau^2 = 0.02, epsilon = 1). The f largest hospitals, which carry
the most weight, are dishonest: instead of their privatized estimate they release mu + b*tau. Sizes
are public, so they cannot change their weights. We report the mean risk of the honest hospitals,
relative to local estimation, for
  GHB          DerSimonian-Laird and the inverse-variance leave-one-out mean of all releases;
  robust GHB   each hospital first screens the releases (which it can obtain by flooding them over
               the graph): with m the inverse-variance weighted median and tau_0^2 a MAD-based
               estimate of the spread, releases with |x_j - m| / sqrt(r_j + tau_0^2) > 3 are dropped,
               and GHB is run on the rest. This is post-processing and costs no privacy.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import ghb
import style

R, TAU2, EPS = 4000, 0.02, 1.0
BS = [0, 1, 2, 3, 5, 8, 12, 20, 30]
FS = [1, 3]


def weighted_median(x, w):
    o = np.argsort(x, axis=1)
    xs = np.take_along_axis(x, o, 1)
    ws = np.take_along_axis(np.broadcast_to(w, x.shape), o, 1)
    c = np.cumsum(ws, 1) / ws.sum(1, keepdims=True)
    i = (c < 0.5).sum(1)
    return xs[np.arange(len(x)), i]


def robust(th, x, r, v):
    K = x.shape[1]
    m = weighted_median(x, 1.0 / r)
    mad = np.median(np.abs(x - m[:, None]), axis=1)
    t0 = np.maximum((1.4826 * mad) ** 2 - np.median(r), 0.0)
    keep = np.abs(x - m[:, None]) / np.sqrt(r + t0[:, None]) <= 3.0
    est = np.empty_like(th)
    for i in range(len(x)):  # replicate by replicate: the kept set differs
        kk = keep[i]
        xk, rk = x[i, kk][None, :], r[kk][None, :] if np.ndim(r) == 1 else r[i, kk][None, :]
        t2 = ghb.dersimonian_laird(xk, rk, ghb.Averager())[0, 0]
        w_all = 1.0 / (t2 + r)
        w = np.where(kk, w_all, 0.0)
        Sw = w.sum() - w  # leave-one-out among kept releases
        mu = ((w * x[i]).sum() - w * x[i]) / np.maximum(Sw, 1e-12)
        M = 1.0 / np.maximum(Sw, 1e-12)
        a = (t2 + M) / (t2 + M + v)
        est[i] = a * th[i] + (1 - a) * mu
    return est, keep.mean(0)


def main():
    rng = np.random.default_rng(12)
    n = ghb.site_sizes()
    v = 1.0 / n
    K = len(n)
    out = {}
    for f in FS:
        bad = np.arange(K - f, K)  # the f largest hospitals
        honest = np.setdiff1d(np.arange(K), bad)
        rows = {"GHB": [], "Robust GHB": [], "kept_bad": []}
        for b in BS:
            theta, th = ghb.draw(rng, n, TAU2, R)
            x, s2 = ghb.release(rng, th, n, EPS)
            x[:, bad] = b * np.sqrt(TAU2)  # mu = 0
            r = v + s2
            plain = ghb.ghb(th, x, n, s2)
            rob, kept = robust(th, x, r, v)
            for name, est in (("GHB", plain), ("Robust GHB", rob)):
                rk, _ = ghb.risk(est[:, honest], theta[:, honest])
                rows[name].append(float((rk / v[honest]).mean()))
            rows["kept_bad"].append(float(kept[bad].mean()))
            print(f"f={f} b={b}: GHB {rows['GHB'][-1]:.3f} robust {rows['Robust GHB'][-1]:.3f} "
                  f"attackers kept {rows['kept_bad'][-1]:.2f}", flush=True)
        out[f] = rows
    style.setup()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    for f, ls in zip(FS, ["-", "--"]):
        ax.plot(BS, out[f]["GHB"], color=style.TEAL700, ls=ls, marker="o", ms=3, lw=1.2, label=f"GHB, {f} dishonest")
        ax.plot(BS, out[f]["Robust GHB"], color=style.VIOLET600, ls=ls, marker="s", ms=3, lw=1.2,
                label=f"Robust GHB, {f} dishonest")
    ax.axhline(1, color=style.SLATE800, lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel(r"Size of the lie, $b$ (in units of $\tau$)")
    ax.set_ylabel(r"Honest hospitals: mean $R_k/v_k$")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ghb.tidy(ax)
    style.save(fig, ghb.FIGDIR / "sim12_byzantine.pdf")
    json.dump(dict(b=BS, out={str(k): v for k, v in out.items()}), open(ghb.RESDIR / "sim12_byzantine.json", "w"), indent=1)


if __name__ == "__main__":
    main()
