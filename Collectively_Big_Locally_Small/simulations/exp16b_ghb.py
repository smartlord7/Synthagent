"""Simulation 16B: ablation and sensitivity analysis of GHB as released (joint releases; real_v3.GHB_V5 without the
pan-network step, since a simulated network is alone), in the logistic world and the eight settings of Simulation 16.

In every replicate all variants use the same data and the same privacy noise (one seed per replicate), so
differences are paired. Ablation: one component removed or replaced at a time. Sensitivity: one fixed setting varied
at a time (half-Cauchy scale of tau, non-borrowing weight, budget share of the information release, number of
quantiles Q, default prior scales). Local estimation is a reference.
    python3 exp16b_ghb.py            (results/sim16_ghb.json, figures/sim16_ghb.pdf)
    python3 exp16b_ghb.py --redraw
"""
import hashlib
import json
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import ghb
import ghb_logit as GL
import ghb_v3 as V3
import ghb_v4 as V4
import logit_world as L
import style
from exp7_logistic import P, R_X, fit
from exp16_ablation import LABELS, SETTINGS, T

REPS = 40
BASE = dict(L.V4_CFG)  # GHB as released, one network
ABLATION = [
    ("GHB (as released)", {}),
    ("Without the non-borrowing components", dict(safety=False)),
    ("Without propagating the privacy covariance", dict(vdp=False)),
    ("Without removing the hospital's own data", dict(loo=False)),
    ("Intercept borrowed like the slopes", dict(int_sd=None, free_int=None)),
    ("Loose sensitivity of the information release", dict(tight=False)),
    ("Full information matrix always", dict(hdiag=False)),
    ("Diagonal information always", dict(hdiag=True)),
    ("Without the anchor", dict(_anchor=False)),
]
SENSITIVITY = [
    ("Scale of tau 0.1", dict(scale=0.1)), ("Scale of tau 0.25", dict(scale=0.25)),
    ("Scale of tau 1", dict(scale=1.0)), ("Scale of tau 2", dict(scale=2.0)),
    ("Non-borrowing weight 0.1", dict(safety_w=0.1)), ("Non-borrowing weight 1", dict(safety_w=1.0)),
    ("Non-borrowing weight 3", dict(safety_w=3.0)),
    ("Information share 0.25", dict(hshare=0.25)), ("Information share 0.75", dict(hshare=0.75)),
    ("Q = 1", dict(_Q=1)), ("Q = 3", dict(_Q=3)), ("Q = 6", dict(_Q=6)), ("Q = 24", dict(_Q=24)),
    ("Default scales {2.5}", dict(_SDS=(2.5,))), ("Default scales {1}", dict(_SDS=(1.0,))),
    ("Default scales {0.5, 1, 2.5, 5}", dict(_SDS=(0.5, 1.0, 2.5, 5.0))),
]
VARIANTS = ABLATION + SENSITIVITY


def ghb_variant(data, mu0, eps, seed, change):
    cfg = dict(BASE, **{k: v for k, v in change.items() if not k.startswith("_")})
    cfg["anchor"] = (mu0, 0.3) if change.get("_anchor", True) else None  # the published model, as in Simulation 15
    q, sds = change.get("_Q", V3.Q), change.get("_SDS", V3.SDS)
    saved = V4.Q, V4.SDS, V3.quantiles.__defaults__
    V4.Q, V4.SDS, V3.quantiles.__defaults__ = q, tuple(sds), (GL.GRID, q)
    try:
        return V4.estimate({"net": data}, np.random.default_rng(seed), eps, 1e-5, R_X, cfg=cfg)["net"]
    finally:
        V4.Q, V4.SDS, V3.quantiles.__defaults__ = saved


def one_rep(rng, n, tau2, eps, anchor_err=0.3, shift=0.0):
    theta, means, mu0, data = L.world(rng, n, tau2, shift, anchor_err)
    seed = int.from_bytes(hashlib.sha256(data[0][0].tobytes() + data[-1][1].tobytes()).digest()[:8], "little")
    est = {name: ghb_variant(data, mu0, eps, seed, ch) for name, ch in VARIANTS}
    est["Local"] = np.array([fit(X, y, mu0, np.ones(P)) for X, y in data])
    res = [L.evaluate(rng, theta[k], means[k], {m: e[k] for m, e in est.items()}) for k in range(len(n))]
    return {m: np.array([res[k][m][0] for k in range(len(n))]) for m in est}


def setting(item):
    i, (label, kw) = item
    rng = np.random.default_rng([162, i])
    n = ghb.site_sizes()
    KL = {}
    for _ in range(REPS):
        for m, v in one_rep(rng, n, **kw).items():
            KL.setdefault(m, []).append(v)
    KL = {m: np.array(v) for m, v in KL.items()}
    full = KL["GHB (as released)"].mean(axis=1)
    loc = KL["Local"].mean(axis=0)
    out = {}
    for m, v in KL.items():
        per = v.mean(axis=1)
        d = per - full
        out[m] = dict(kl=float(per.mean()), se=float(per.std(ddof=1) / np.sqrt(REPS)), diff=float(d.mean()),
                      diff_se=float(d.std(ddof=1) / np.sqrt(REPS)), ratio=float(per.mean() / full.mean()),
                      worst=float((v.mean(axis=0) / loc).max()))
    print(label, {m: round(o["ratio"], 3) for m, o in out.items()}, flush=True)
    return label, out


def figure(res):
    style.setup()
    rows = [m for m, _ in VARIANTS] + ["Local"]
    cols = [lab for lab, _ in SETTINGS]
    Z = np.array([[res[c][m]["ratio"] for c in cols] for m in rows])
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lz = np.log2(Z)
    im = ax.imshow(lz, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            r = res[cols[j]][rows[i]]
            sig = abs(r["diff"]) > 2 * r["diff_se"]
            ax.text(j, i, f"{Z[i, j]:.2f}" + ("" if sig or i == 0 else " ·"), ha="center", va="center",
                    fontsize=5.4, color="white" if abs(lz[i, j]) > 0.6 else "black", fontweight="bold" if sig else "normal")
    ax.axhline(len(ABLATION) - 0.5, color="black", lw=0.8)
    ax.axhline(len(VARIANTS) - 0.5, color="black", lw=0.8)
    ax.set_xticks(range(len(cols)), [LABELS[c] for c in cols], fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(["0.5", "0.71", "1", "1.41", "2"], fontsize=6)
    cb.set_label("Error relative to GHB as released", fontsize=6.5)
    for s in ax.spines.values():
        s.set_visible(False)
    style.save(fig, ghb.FIGDIR / "sim16_ghb.pdf")


def main():
    with Pool(4) as pool:
        res = dict(pool.map(setting, list(enumerate(SETTINGS))))
    json.dump(dict(reps=REPS, res=res, ablation=[m for m, _ in ABLATION], sensitivity=[m for m, _ in SENSITIVITY]),
              open(ghb.RESDIR / "sim16_ghb.json", "w"), indent=1)
    figure(res)


if __name__ == "__main__":
    if "--redraw" in sys.argv:
        figure(json.load(open(ghb.RESDIR / "sim16_ghb.json"))["res"])
    else:
        main()
