"""Figure 5: the Scarcity-Provenance matrix of judged potential."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from taxonomy import CHANNELS, CLASSES, FAMILY_BREAKS, POTENTIAL  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_matrix.pdf"
GRADE_COLORS = ["#ffffff", "#ccfbf1", "#13b8a6", "#0f4c45"]


def main():
    style.setup()
    grades = np.array(POTENTIAL)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.imshow(grades, cmap=ListedColormap(GRADE_COLORS), vmin=-0.5, vmax=3.5, aspect="auto")
    nrow, ncol = grades.shape
    for x in np.arange(-0.5, ncol):
        ax.axvline(x, color="#e5e7eb", linewidth=0.6)
    for y in np.arange(-0.5, nrow):
        ax.axhline(y, color="#e5e7eb", linewidth=0.6)
    for b in FAMILY_BREAKS:
        ax.axhline(b - 0.5, color=style.SLATE800, linewidth=0.8)
    for i in range(nrow):
        for j in range(ncol):
            g = grades[i, j]
            if g:
                ax.text(j, i, "+" * g, ha="center", va="center", fontsize=7,
                        color="white" if g == 3 else style.SLATE800)
    ax.set_xticks(range(ncol), [name for _, name in CHANNELS], rotation=35, ha="right")
    ax.set_yticks(range(nrow), [f"{code} {name}" for code, name in CLASSES])
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
