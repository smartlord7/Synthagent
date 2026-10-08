"""Figure 2: deflation of 2,000 ICU stays into an effective sample size (Example 2.4)."""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_deflation.pdf"


def riley_n_star(p, prevalence, r2_fraction=0.15, shrinkage=0.9):
    """First criterion of Riley et al. (2019b), Equation (1).

    The anticipated Cox-Snell R^2 is a fraction of its maximum, which for a
    binary outcome with the given prevalence is 1 - exp(2 ln L0 / n), with
    ln L0 / n = pi ln pi + (1 - pi) ln(1 - pi).
    """
    pi = prevalence
    max_r2 = 1 - math.exp(2 * (pi * math.log(pi) + (1 - pi) * math.log(1 - pi)))
    r2 = r2_fraction * max_r2
    return p / ((shrinkage - 1) * math.log(1 - r2 / shrinkage))


def main():
    style.setup()
    n = 2000
    deff = 1 + (1.6 - 1) * 0.5          # 1.3
    rho_l, eta, rho_s, prevalence = 0.7, 0.1, 0.6, 0.08
    stages = [n, n / deff]
    stages.append(stages[-1] * rho_l)
    stages.append(stages[-1] * (1 - 2 * eta) ** 2)
    stages.append(stages[-1] * rho_s)
    n_star = riley_n_star(20, prevalence)
    events = stages[-1] * prevalence

    labels = ["Nominal\nrecords", "Clustering\n(DEFF=1.3)", "Labeled\n(70%)",
              "Label noise\n($\\eta$=0.1)", "Shift\n(ESS 60%)"]
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    width = 0.62
    for i, v in enumerate(stages):
        color = style.GRAY400 if i == 0 else style.TEAL700
        ax.bar(i, v, width=width, color=color, linewidth=0)
        if i > 0:
            ax.bar(i, stages[i - 1] - v, bottom=v, width=width, color=style.RED200,
                   edgecolor=style.RED700, linewidth=0.4, hatch="////")
        ax.text(i, v - 60, f"{v:,.0f}", ha="center", va="top", color="white",
                fontsize=7, fontweight="bold")
    ax.axhline(n_star, color=style.RED700, linestyle="--", linewidth=1.0)
    ax.text(4.5, n_star + 50, f"$n^\\star\\approx${n_star:,.0f} (Riley et al., p=20)",
            ha="right", va="bottom", color=style.RED700, fontsize=7)
    ax.annotate(f"only {events:.0f} outcome events\nat {prevalence:.0%} prevalence",
                xy=(4.32, stages[-1] / 2), xytext=(4.55, 1646), fontsize=7,
                color=style.RED700, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=style.RED700, linewidth=0.6))
    ax.set_xticks(range(len(stages)), labels)
    ax.set_xlim(-0.5, 5.9)
    ax.set_ylim(0, 3100)
    ax.set_ylabel("Patients (effective)")
    style.save(fig, OUT)
    print(f"n_eff={stages[-1]:.0f}, events={events:.1f}, n*={n_star:.0f}, sigma={n_star / stages[-1]:.2f}")


if __name__ == "__main__":
    main()
