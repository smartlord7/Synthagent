"""Load the coded corpus (data/corpus.csv) and compute the counts reported in Sections 5 and 7."""
import csv
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "corpus.csv"
CHANNELS = ["P1", "P2", "P3", "P4", "P5", "P6", "none"]


def load(path=DATA):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["year"] = int(r["year"])
        r["scarcity"] = [c for c in r["scarcity_classes"].split(";") if c]
    return rows


def summary(rows):
    """Counts quoted in the text of Sections 5 and 7."""
    remedy = [r for r in rows if r["primary_channel"] != "none"]
    health = [r for r in rows if r["modality"] != "General"]
    out = {
        "works": len(rows),
        "since_2019": sum(r["year"] >= 2019 for r in rows),
        "year_range": (min(r["year"] for r in rows), max(r["year"] for r in rows)),
        "by_primary_channel": Counter(r["primary_channel"] for r in rows),
        "remedy_works": len(remedy),
        "with_secondary": sum(bool(r["secondary_channel"]) for r in remedy),
        "health_works": len(health),
        "by_modality": Counter(r["modality"] for r in rows),
        "by_type": Counter(r["contribution_type"] for r in rows),
        "multisite_by_channel": Counter(r["primary_channel"] for r in rows
                                        if r["contribution_type"] == "Multi-site"),
        "by_class": Counter(c for r in rows for c in r["scarcity"]),
    }
    return out


def coverage(rows):
    """Works targeting each scarcity class through each channel, as primary or secondary."""
    cells = Counter()
    for r in rows:
        chans = {r["primary_channel"]} | ({r["secondary_channel"]} if r["secondary_channel"] else set())
        for c in r["scarcity"]:
            for ch in chans:
                if ch != "none":
                    cells[(c, ch)] += 1
    return cells


if __name__ == "__main__":
    rows = load()
    for k, v in summary(rows).items():
        print(f"{k}: {v}")
