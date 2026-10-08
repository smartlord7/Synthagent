"""Shared Matplotlib style for the figures of the paper (DejaVu Serif, Tailwind palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette (Tailwind CSS colours used throughout the paper)
SLATE800 = "#1f2933"
GRAY400 = "#9aa5b1"
TEAL700 = "#0f766e"
TEAL100 = "#ccfbf1"
RED700 = "#b91c1c"
RED200 = "#fecaca"
AMBER600 = "#d97706"
BLUE600 = "#2563eb"
VIOLET600 = "#7c3aed"

# Colours of the six provenance channels and of "no remedy"
CHANNEL_COLORS = {
    "P1": TEAL700,
    "P2": AMBER600,
    "P3": RED700,
    "P4": BLUE600,
    "P5": VIOLET600,
    "P6": "#6b7280",
    "none": "#d1d5db",
}
CHANNEL_NAMES = {
    "P1": "P1 Pooling",
    "P2": "P2 Proxy",
    "P3": "P3 Prior",
    "P4": "P4 Unlabeled",
    "P5": "P5 Procurement",
    "P6": "P6 Re-expression",
    "none": "no remedy",
}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": SLATE800,
        "axes.labelcolor": SLATE800,
        "xtick.color": SLATE800,
        "ytick.color": SLATE800,
        "text.color": SLATE800,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 1.0,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
