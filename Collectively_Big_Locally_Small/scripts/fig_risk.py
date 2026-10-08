"""Figure 8 (left): risk of hospital k as a function of its size (Propositions 8.1 and 8.2)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from blueprint import loo_mse, risk_full_pooling, risk_hb, risk_local  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_risk.pdf"
K, N_OTHER, SIGMA2, TAU2, DELTA = 12, 200, 1.0, 0.02, 1e-5


def main():
    style.setup()
    n = np.logspace(1, np.log10(5000), 200)
    others = np.full(K - 1, N_OTHER)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(n, risk_local(n, SIGMA2), color=style.GRAY400, linewidth=1.4, label="Local only")
    ax.plot(n, risk_full_pooling(n, others, TAU2, SIGMA2), color=style.AMBER600, linewidth=1.4,
            label="Full pooling")
    ax.plot(n, risk_hb(n, TAU2, loo_mse(others, TAU2, SIGMA2), SIGMA2), color=style.TEAL700,
            linewidth=1.8, label="Hierarchical borrowing")
    for eps, ls, col in [(1.0, "--", style.BLUE600), (0.1, ":", style.VIOLET600)]:
        M = loo_mse(others, TAU2, SIGMA2, eps=eps, delta=DELTA)
        ax.plot(n, risk_hb(n, TAU2, M, SIGMA2), color=col, linewidth=1.2, linestyle=ls,
                label=f"  + DP, $\\epsilon$={eps:g}")
    ax.axhline(TAU2, color=style.SLATE800, linewidth=0.5, linestyle="-.")
    ax.text(3600, TAU2 * 1.05, "$\\tau^2$", fontsize=7, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Local sample size $n_k$")
    ax.set_ylabel("Site risk $R_k$ (MSE)")
    ax.legend(loc="lower left", frameon=False)
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
