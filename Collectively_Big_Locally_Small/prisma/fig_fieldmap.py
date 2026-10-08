"""Supplementary figure: coverage of the hand-coded corpus (Figure 7 of the paper) next to the
dictionary-coded field-level map of the systematic search, with the gap cells of the corpus outlined.
Field cells are shaded by percentile rank among the 60 cells; corpus gap cells that fall in the
bottom third of the field map are confirmed."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import style  # noqa: E402
from corpus import coverage, load  # noqa: E402
from fig_gapmap import SHORT, gap_cells  # noqa: E402
from taxonomy import CHANNELS, CLASSES, FAMILY_BREAKS, POTENTIAL  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("gap", ["#ffffff", "#ccfbf1", "#13b8a6", "#0f4c45"])
OUT = ROOT / "supplement" / "figures" / "s8_fieldmap.pdf"


def panel(ax, counts, shade, gaps, title, confirmed=None, ylabels=True):
    ax.imshow(shade, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i, j in gaps:
        ok = confirmed is None or confirmed[(i, j)]
        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=style.RED700,
                               linewidth=1.3, linestyle="-" if ok else "--"))
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color="white" if shade[i, j] > 0.6 else style.SLATE800)
    for x in np.arange(-0.5, counts.shape[1]):
        ax.axvline(x, color="#e5e7eb", linewidth=0.6)
    for y in np.arange(-0.5, counts.shape[0]):
        ax.axhline(y, color="#e5e7eb", linewidth=0.6)
    for b in FAMILY_BREAKS:
        ax.axhline(b - 0.5, color=style.SLATE800, linewidth=0.8)
    ax.set_xticks(range(len(CHANNELS)), [c for c, _ in CHANNELS])
    ax.set_yticks(range(len(CLASSES)), [f"{c} {SHORT[c]}" for c, _ in CLASSES] if ylabels else [])
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=7.5)


def main():
    style.setup()
    cov = coverage(load())
    corpus = np.array([[cov[(c, ch)] for ch, _ in CHANNELS] for c, _ in CLASSES])
    summary = json.load(open(ROOT / "prisma" / "data" / "classify_summary.json"))
    fc = summary["field_cells"]
    field = np.array([[fc[f"{c}x{ch}"] for ch, _ in CHANNELS] for c, _ in CLASSES])
    flat = field.ravel()
    pct = np.array([(flat < v).mean() for v in flat]).reshape(field.shape)
    gaps = gap_cells(corpus, POTENTIAL)
    confirmed = {g: pct[g] < 1 / 3 for g in gaps}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw=dict(wspace=0.05))
    panel(a, corpus, np.log1p(corpus) / np.log1p(corpus.max()), gaps, "Coded corpus (290 works)")
    panel(b, field, pct, gaps, f"Systematic search ({summary['eligible']:,} records)",
          confirmed, ylabels=False)
    style.save(fig, OUT)
    print(sum(confirmed.values()), "of", len(gaps), "gaps confirmed")


if __name__ == "__main__":
    main()
