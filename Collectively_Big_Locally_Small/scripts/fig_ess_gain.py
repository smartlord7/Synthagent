"""Figure 8 (right): relative gain in effective sample size n_k^eff / n_k (Equations 5 and 6)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from blueprint import borrowed_patients, loo_mse  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_ess_gain.pdf"
K, N_OTHER, N_K, SIGMA2, DELTA = 12, 200, 50, 1.0, 1e-5
CMAP = LinearSegmentedColormap.from_list("ess", ["#f8fafc", "#99f6e4", "#0f766e", "#042f2e"])
GRID = 120


def gain(eps, tau2):
    others = np.full(K - 1, N_OTHER)
    M = loo_mse(others, tau2, SIGMA2, eps=eps, delta=DELTA)
    return 1 + borrowed_patients(tau2, M, SIGMA2) / N_K


def main():
    style.setup()
    eps = np.logspace(-1.5, 1, GRID)
    tau2 = np.logspace(-3, 0, GRID)
    G = np.array([[gain(e, t) for e in eps] for t in tau2])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    mesh = ax.pcolormesh(eps, tau2, np.log10(G), cmap=CMAP, shading="gouraud", rasterized=True)
    cs = ax.contour(eps, tau2, G, levels=[1.5, 2, 4, 8, 16], colors=style.SLATE800, linewidths=0.5)
    ax.clabel(cs, fmt=lambda v: f"{v:g}x", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Privacy budget $\\epsilon$ per release")
    ax.set_ylabel("Heterogeneity $\\tau^2$")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.outline.set_linewidth(0.3)
    cb.set_label("$\\log_{10}(n_k^{\\mathrm{eff}}/n_k)$")
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
