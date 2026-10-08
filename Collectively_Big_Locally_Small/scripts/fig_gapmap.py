"""Figure 7: gap map of corpus coverage against the judged potential of Figure 5."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from corpus import coverage, load  # noqa: E402
from taxonomy import CHANNELS, CLASSES, FAMILY_BREAKS, POTENTIAL  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_gapmap.pdf"
SHORT = {"O1": "Rarity", "O2": "Event sparsity", "O3": "Stratum sparsity", "E1": "Label poverty",
         "E2": "Measurement poverty", "E3": "Dimensional excess", "T1": "Obsolescence",
         "T2": "Context mismatch", "C1": "Fragmentation", "C2": "Consent, access"}
CMAP = LinearSegmentedColormap.from_list("gap", ["#ffffff", "#ccfbf1", "#13b8a6", "#0f4c45"])


def gap_cells(counts, potential, min_potential=2, max_count=2):
    """Cells judged at least moderately promising that contain few works."""
    return [(i, j) for i in range(counts.shape[0]) for j in range(counts.shape[1])
            if potential[i][j] >= min_potential and counts[i, j] <= max_count]


def main():
    style.setup()
    cov = coverage(load())
    counts = np.array([[cov[(c, ch)] for ch, _ in CHANNELS] for c, _ in CLASSES])
    shade = np.log1p(counts) / np.log1p(counts.max())
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.imshow(shade, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    gaps = gap_cells(counts, POTENTIAL)
    for i, j in gaps:
        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                               edgecolor=style.RED700, linewidth=1.3))
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            dark = shade[i, j] > 0.6
            ax.text(j, i - 0.12, f"{counts[i, j]}", ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color="white" if dark else style.SLATE800)
            g = POTENTIAL[i][j]
            ax.text(j, i + 0.27, "+" * g if g else "·", ha="center", va="center",
                    fontsize=5.5, color="white" if dark else "#6b7280")
    nrow, ncol = counts.shape
    for x in np.arange(-0.5, ncol):
        ax.axvline(x, color="#e5e7eb", linewidth=0.6)
    for y in np.arange(-0.5, nrow):
        ax.axhline(y, color="#e5e7eb", linewidth=0.6)
    for b in FAMILY_BREAKS:
        ax.axhline(b - 0.5, color=style.SLATE800, linewidth=0.8)
    ax.set_xticks(range(ncol), [f"{code} {name}" for code, name in CHANNELS], rotation=35, ha="right")
    ax.set_yticks(range(nrow), [f"{code} {SHORT[code]}" for code, _ in CLASSES])
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    style.save(fig, OUT)
    print(f"{len(gaps)} gap cells:", [(CLASSES[i][0], CHANNELS[j][0]) for i, j in gaps])


if __name__ == "__main__":
    main()
