"""Figure 6: profile of the coded corpus by year, modality and contribution type."""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from corpus import CHANNELS, load  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_corpus.pdf"
MODALITIES = [("Imaging", "Imaging"), ("EHR", "Structured EHR"), ("Tabular", "Tabular, registry"),
              ("Text", "Clinical text"), ("Omics", "Omics"), ("Signals", "Signals"),
              ("Multimodal", "Multimodal")]
TYPES = [("Method", "Method"), ("Theory", "Theory"), ("Study", "Single-data study"),
         ("Multi-site", "Multi-site study"), ("Resource", "Data, infrastructure"),
         ("Review", "Review"), ("Evidence", "Failure, risk evidence")]
YEARS = list(range(2012, 2026))
LEGEND_X = 253.61
LEGEND_Y = 3.7


def stacked(ax, cats, key, rows, horizontal):
    counts = {ch: Counter(key(r) for r in rows if r["primary_channel"] == ch) for ch in CHANNELS}
    pos = np.arange(len(cats))
    base = np.zeros(len(cats))
    for ch in CHANNELS:
        vals = np.array([counts[ch][c] for c in cats], dtype=float)
        if horizontal:
            ax.barh(pos, vals, left=base, color=style.CHANNEL_COLORS[ch], height=0.75,
                    label=style.CHANNEL_NAMES[ch])
        else:
            ax.bar(pos, vals, bottom=base, color=style.CHANNEL_COLORS[ch], width=0.8,
                   label=style.CHANNEL_NAMES[ch])
        base += vals
    return pos


def main():
    style.setup()
    rows = load()
    # Figure and axes geometry in points (the layout of the published figure).
    width, height = 518.06, 187.84
    fig = plt.figure(figsize=(width / 72, height / 72))
    rects = [(36.231, 62.169, 106.816, 105.631), (239.525, 62.169, 79.123, 105.631),
             (415.127, 62.169, 79.123, 105.631)]
    axes = [fig.add_axes((x / width, y / height, w / width, h / height)) for x, y, w, h in rects]

    ax = axes[0]
    year_key = lambda r: "<2012" if r["year"] < 2012 else str(r["year"])  # noqa: E731
    cats = ["<2012"] + [str(y) for y in YEARS]
    pos = stacked(ax, cats, year_key, rows, horizontal=False)
    ax.set_xticks(pos, ["<2012"] + [f"'{y % 100:02d}" for y in YEARS], rotation=90)
    ax.set_ylabel("papers")
    ax.set_title("(a) Year of publication", loc="left", fontsize=9)

    ax = axes[1]
    health = [r for r in rows if r["modality"] != "General"]
    pos = stacked(ax, [m for m, _ in MODALITIES], lambda r: r["modality"], health, horizontal=True)
    ax.set_yticks(pos, [name for _, name in MODALITIES])
    ax.invert_yaxis()
    ax.set_xlabel("papers")
    ax.set_title("(b) Healthcare modality", loc="left", fontsize=9)

    ax = axes[2]
    pos = stacked(ax, [t for t, _ in TYPES], lambda r: r["contribution_type"], rows, horizontal=True)
    ax.set_yticks(pos, [name for _, name in TYPES])
    ax.invert_yaxis()
    ax.set_xlabel("papers")
    ax.set_title("(c) Contribution type", loc="left", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(LEGEND_X / width, LEGEND_Y / height), ncol=7,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=1.0)
    style.save(fig, OUT)


if __name__ == "__main__":
    main()
