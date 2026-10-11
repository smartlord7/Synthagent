"""Simulation 14: privacy accounting when the model is refitted on the same patients.

A hospital that refits its head r times on the same patients makes r releases, and their privacy
costs compose. For a fixed total budget (epsilon = 1, delta = 1e-5 over all refits) we compare:
  basic composition   each release gets (epsilon/r, delta/r) and the classical Gaussian mechanism;
  zCDP                the Gaussian mechanism with noise s is rho-zCDP with rho = Delta^2/(2 s^2), rho adds
                      over releases, and rho-zCDP implies (rho + 2 sqrt(rho ln(1/delta)), delta)-DP
                      (Bun and Steinke, 2016); the total rho is chosen so that this equals epsilon.
For each scheme, the borrowed patients B of Equation (5) after the last refit (closed form,
inverse-variance leave-one-out mean, the 24 hospitals of Simulation 2, tau^2 = 0.02, C = 1).
Refits on disjoint new patients do not compose; B is then that of a single release.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import ghb
import style

EPS, DELTA, TAU2, C = 1.0, 1e-5, 0.02, 1.0
RS = [1, 2, 4, 8, 16, 32, 64]


def B_for_noise(n, s2):
    w = 1.0 / (TAU2 + 1.0 / n + s2)
    M = 1.0 / (w.sum() - w)  # leave-one-out, per hospital
    return 1.0 / (TAU2 + M)


def main():
    n = ghb.site_sizes()
    sens = 2 * C / n
    rho_tot = brentq(lambda r: r + 2 * np.sqrt(r * np.log(1 / DELTA)) - EPS, 1e-9, 10)
    rows = []
    for r in RS:
        e, d = EPS / r, DELTA / r
        s2_basic = (sens * np.sqrt(2 * np.log(1.25 / d)) / e) ** 2
        s2_zcdp = sens ** 2 / (2 * rho_tot / r)
        rows.append(dict(r=r, B_basic=float(B_for_noise(n, s2_basic).mean()), B_zcdp=float(B_for_noise(n, s2_zcdp).mean()),
                         noise_ratio=float(np.sqrt(s2_basic[0] / s2_zcdp[0]))))
        print(rows[-1])
    B_none = float(B_for_noise(n, np.zeros_like(n)).mean())
    B_one = rows[0]["B_basic"]
    style.setup()
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    ax.plot(RS, [x["B_basic"] for x in rows], "-o", ms=3, color=style.RED700, lw=1.2, label="basic composition")
    ax.plot(RS, [x["B_zcdp"] for x in rows], "-s", ms=3, color=style.TEAL700, lw=1.2, label="zero-concentrated DP")
    ax.axhline(B_one, color=style.SLATE800, lw=0.7, ls="--", label="refits on new patients (no composition)")
    ax.axhline(B_none, color=style.GRAY400, lw=0.7, ls=":", label="no privacy")
    ax.set_xscale("log", base=2); ax.set_xticks(RS, [str(r) for r in RS]); ax.minorticks_off()
    ax.set_xlabel("Refits on the same patients, $r$"); ax.set_ylabel("Borrowed patients $B$ (mean)")
    ax.set_title(r"Total budget $\epsilon$=1, $\delta=10^{-5}$", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    style.save(fig, ghb.FIGDIR / "sim14_accounting.pdf")
    json.dump(dict(rows=rows, B_none=B_none, rho_total=rho_tot), open(ghb.RESDIR / "sim14_accounting.json", "w"), indent=1)


if __name__ == "__main__":
    main()
