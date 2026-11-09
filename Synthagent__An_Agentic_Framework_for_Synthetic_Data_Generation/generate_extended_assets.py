"""
Assets for the ten-dataset generator comparison and the sensitivity study.

generate_paper_assets.py covers the original six datasets (control-layer variants, LLM
variants, real-only baselines). This script adds, from the same results folder:

- generated/datasets10.tex            the ten datasets with the Riley et al. sample size
- generated/generators10.csv / .tex   every generator over the ten datasets
- generated/generators10_f1.csv       F1 change per generator and dataset
- generated/friedman10.csv            dataset-level Friedman and Holm-Wilcoxon tests
- generated/sensitivity.csv / .tex    the reference configuration under changed parameters
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import generate_paper_assets as gpa

ALL10 = {
    **gpa.DATASET_LABELS,
    "breast_cancer_coimbra.csv": "Breast cancer (Coimbra)",
    "hepatitis.csv": "Hepatitis",
    "chronic_kidney_disease.csv": "Kidney disease",
    "thoracic_surgery.csv": "Thoracic surgery",
}
TARGETS10 = {
    **gpa.TARGETS,
    "breast_cancer_coimbra.csv": "Classification",
    "hepatitis.csv": "Class",
    "chronic_kidney_disease.csv": "class",
    "thoracic_surgery.csv": "Risk1Yr",
}
GENERATORS10 = [
    ("full", "Synthagent, GPT-4o mini"),
    ("backend_gpt_4_1", "Synthagent, GPT-4.1"),
    ("backend_claude_sonnet_4_6", "Synthagent, Sonnet 4.6"),
    ("raw_prompting", "Direct prompting"),
    ("structured_output", "Structured outputs"),
    ("cllm", "CLLM"),
    ("smote", "SMOTE"),
    ("adasyn", "ADASYN"),
    ("borderline_smote", "Borderline-SMOTE"),
    ("gaussian_copula", "Gaussian copula"),
    ("cart", "CART"),
    ("arf", "ARF"),
    ("ctgan", "CTGAN"),
    ("tvae", "TVAE"),
    ("tabddpm", "TabDDPM"),
    ("great", "GReaT"),
    ("bayesian_network", "Bayesian network"),
]
LLM_ARMS = {"full", "backend_gpt_4_1", "backend_claude_sonnet_4_6", "raw_prompting", "structured_output", "cllm"}
SENSITIVITY = [
    ("full", r"Reference ($\rho=20\%$, $b=30$, $T=0.3$)"),
    ("ctx_10", r"$\rho=10\%$"),
    ("ctx_40", r"$\rho=40\%$"),
    ("batch_15", r"$b=15$"),
    ("batch_60", r"$b=60$"),
    ("temp_0", r"$T=0$"),
    ("temp_10", r"$T=1.0$"),
]


def riley_n(n, d, k, n_min, shrinkage=0.9, share=0.15):
    """First criterion of Riley et al. (binary; smallest class versus the rest)."""
    phi = n_min / n
    max_r2 = 1 - math.exp(2 * (phi * math.log(phi) + (1 - phi) * math.log(1 - phi)))
    r2 = share * max_r2
    params = d * max(1, k - 1)
    return params / ((shrinkage - 1) * math.log(1 - r2 / shrinkage))


def dataset_table():
    rows = []
    for filename, label in ALL10.items():
        raw = pd.read_csv(gpa.DATA / filename)
        target = TARGETS10[filename]
        counts = raw[target].value_counts()
        n, d, k, n_min = len(raw), len(raw.columns) - 1, raw[target].nunique(), int(counts.min())
        n_star = riley_n(n, d, k, n_min)
        rows.append({"Dataset": label, "n": n, "d": d, "k": k, "n_min": n_min,
                     "minority_pct": 100 * n_min / n, "missing_pct": 100 * raw.isna().sum().sum() / raw.size,
                     "n_star": n_star, "ratio": n_star / (0.8 * n)})
    table = pd.DataFrame(rows)
    table.to_csv(gpa.OUT / "datasets10.csv", index=False)
    lines = [f"{r.Dataset} & {r.n} & {r.d} & {r.k} & {r.n_min} ({r.minority_pct:.1f}) & {r.missing_pct:.1f} & {r.n_star:,.0f} & {r.ratio:.1f} \\\\"
             for r in table.itertuples()]
    (gpa.OUT / "datasets10.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def load(arms, datasets):
    summary = pd.read_csv(gpa.RUN / "ablation_summary.csv")
    summary = summary[summary.arm.isin(arms) & summary.dataset.isin(datasets)].copy()
    summary = gpa.refresh_utility_from_per_fold(summary)
    summary = gpa.mask_failed_generated_metrics(summary)
    summary = gpa.mask_unrun_cells(summary)
    return gpa.add_utility_columns(summary)


def complete(df, arm, datasets):
    g = df[df.arm == arm]
    return set(g.dataset) == set(datasets) and (g.folds == 10).all()


def generator_comparison():
    arms = [a for a, _ in GENERATORS10]
    df = load(arms, ALL10)
    present = [a for a in arms if complete(df, a, ALL10)]
    labels = dict(GENERATORS10)
    rows, per_dataset = [], {}
    for arm in present:
        g = df[df.arm == arm].set_index("dataset")
        f1 = 100 * g["mean_f1_delta"]
        acc = 100 * g["mean_accuracy_delta"]
        per_dataset[labels[arm]] = f1.reindex(list(ALL10)).rename(index=ALL10)
        deliv = (g.get("gen_stats_delivered_rows_mean") / g.get("gen_stats_requested_rows_mean") * 100) if arm in LLM_ARMS else None
        rows.append({
            "arm": arm, "Generator": labels[arm],
            "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "C2ST": g["Gen C2ST Accuracy_mean"].mean(),
            "AUC": g["Gen MIA Attack AUC_mean"].mean(),
            "dAcc": acc.mean(), "dF1": f1.mean(), "dF1_median": f1.median(),
            "worst": f1.min(), "losses": int((f1 < -1).sum()),
            "delivered_new4": float(deliv.reindex(list(ALL10)[6:]).mean()) if deliv is not None else np.nan,
        })
    table = pd.DataFrame(rows)
    table.to_csv(gpa.OUT / "generators10.csv", index=False)
    pd.DataFrame(per_dataset).to_csv(gpa.OUT / "generators10_f1.csv")

    # Dataset-level tests over the ten datasets
    gpa.DATASET_LABELS = ALL10            # dataset_block reindexes on this mapping
    tests, comps = [], []
    for metric in ("accuracy", "f1"):
        _, omnibus, comparisons = gpa.rank_and_test(df, present, metric, "generators10")
        tests.append(omnibus)
        comps.extend(comparisons)
        for ref in ("backend_claude_sonnet_4_6", "full"):
            block = gpa.dataset_block(df, present, metric).dropna()
            raw = []
            for arm in present:
                if arm == ref:
                    continue
                diff = block[ref] - block[arm]
                try:
                    _, p = gpa.wilcoxon(diff, alternative="two-sided")
                except ValueError:
                    p = 1.0
                raw.append((arm, float(diff.mean()), float(p)))
            for (arm, md, p), ph in zip(raw, gpa.holm_adjust([r[2] for r in raw])):
                comps.append({"family": f"vs_{ref}", "metric": metric, "best": ref, "comparison": arm,
                              "mean_difference": md, "p_raw": p, "p_holm": ph})
    gpa.DATASET_LABELS = {k: v for k, v in ALL10.items() if k in gpa.TARGETS}
    pd.DataFrame(tests).to_csv(gpa.OUT / "friedman10.csv", index=False)
    pd.DataFrame(comps).to_csv(gpa.OUT / "posthoc10.csv", index=False)

    # Table rows: fidelity and privacy means; utility as median +- SD over datasets and the
    # share of datasets on which the F1 change is positive.
    table["dAcc_sd"] = [100 * df[df.arm == a]["mean_accuracy_delta"].std() for a in table.arm]
    table["dAcc_median"] = [100 * df[df.arm == a]["mean_accuracy_delta"].median() for a in table.arm]
    table["dF1_sd"] = [100 * df[df.arm == a]["mean_f1_delta"].std() for a in table.arm]
    table["improved_pct"] = [100 * (df[df.arm == a]["mean_f1_delta"] > 0).mean() for a in table.arm]
    table.to_csv(gpa.OUT / "generators10.csv", index=False)
    best = {"JSD": table.JSD.max(), "C2ST": table.C2ST.sub(0.5).abs().min(), "AUC": table.AUC.sub(0.5).abs().min(),
            "acc": table.dAcc_median.max(), "f1": table.dF1_median.max(), "imp": table.improved_pct.max()}

    def b(text, hit):
        return f"\\best{{{text}}}" if hit else text

    def ms(m, sd):
        return f"${m:+.2f}\\pm{sd:.2f}$"

    lines = []
    for r in table.itertuples():
        cells = [r.Generator,
                 b(f"{r.JSD:.1f}", r.JSD == best["JSD"]),
                 b(f"{r.C2ST:.3f}", abs(r.C2ST - 0.5) == best["C2ST"]),
                 b(f"{r.AUC:.3f}", abs(r.AUC - 0.5) == best["AUC"]),
                 b(ms(r.dAcc_median, r.dAcc_sd), r.dAcc_median == best["acc"]),
                 b(ms(r.dF1_median, r.dF1_sd), r.dF1_median == best["f1"]),
                 b(f"{r.improved_pct:.0f}", r.improved_pct == best["imp"])]
        lines.append(" & ".join(cells) + " \\\\")
    (gpa.OUT / "generators10.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table, pd.DataFrame(tests), pd.DataFrame(comps)


def radar(per_dataset):
    """F1 change per dataset for a few representative generators, as a radar chart (symmetric log radius)."""
    import matplotlib.pyplot as plt
    gpa.print_style()
    show = [("Synthagent, Sonnet 4.6", gpa.C_REF), ("Synthagent, GPT-4o mini", gpa.C_LLM),
            ("CLLM", "#8e44ad"), ("SMOTE", gpa.C_BASE), ("CART", "#2e8b57")]
    data = pd.DataFrame(per_dataset)
    labels = list(data.index)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    # Symmetric log radius: linear within +-1 pp and logarithmic beyond, so the small
    # differences near zero, where most generators lie, are spread out and outliers compressed.
    def r(x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.log10(1 + np.abs(x)) - np.sign(-12) * np.log10(13)
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(gpa.COL_W, 2.9), subplot_kw={"polar": True})
    for (name, color), m in zip(show, markers):
        values = data[name].tolist()
        ax.plot(angles, r(values + values[:1]), color=color, lw=1.2, marker=m, ms=2.4, label=name)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, r(0)), color="0.3", lw=0.8, ls="--")
    ax.set_ylim(0, r(15))
    # dataset names just outside the outer ring, aligned away from the centre so no spoke crosses them
    ax.set_xticks(angles[:-1], [""] * len(labels))
    for ang, lab in zip(angles[:-1], labels):
        c, s_ = np.cos(ang), np.sin(ang)
        ha = "left" if c > 0.1 else "right" if c < -0.1 else "center"
        va = "bottom" if s_ > 0.1 else "top" if s_ < -0.1 else "center"
        ax.text(ang, r(15) * 1.04, lab.replace(" (Coimbra)", " (C.)"), fontsize=6, ha=ha, va=va)
    ticks = [-10, -3, -1, 0, 1, 3, 10]
    ax.set_yticks(r(ticks), [f"{t:+d}" if t else "0" for t in ticks], fontsize=5.5)
    ax.set_rlabel_position(90)
    for t in ax.get_yticklabels():
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.8))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=5.8,
              handlelength=1.4, columnspacing=0.8)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.2)
    fig.savefig(gpa.OUT / "radar_f1.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(gpa.OUT / "radar_f1.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def sensitivity():
    arms = [a for a, _ in SENSITIVITY]
    six = gpa.DATASET_LABELS
    df = load(arms, six)
    rows = []
    for arm, label in SENSITIVITY:
        if not complete(df, arm, six):
            continue
        g = df[df.arm == arm]
        rows.append({"arm": arm, "Setting": label,
                     "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
                     "Corr": g["Gen Correlation Stability_mean"].mean(),
                     "C2ST": g["Gen C2ST Accuracy_mean"].mean(),
                     "AUC": g["Gen MIA Attack AUC_mean"].mean(),
                     "dAcc": 100 * g["mean_accuracy_delta"].mean(),
                     "dF1": 100 * g["mean_f1_delta"].mean(),
                     "tokens": g["gen_stats_total_tokens_mean"].mean()})
    table = pd.DataFrame(rows)
    table.to_csv(gpa.OUT / "sensitivity.csv", index=False)
    tests = []
    present = list(table.arm)
    for metric in ("accuracy", "f1"):
        _, omnibus, _ = gpa.rank_and_test(df, present, metric, "sensitivity")
        tests.append(omnibus)
    pd.DataFrame(tests).to_csv(gpa.OUT / "friedman_sensitivity.csv", index=False)
    lines = [f"{r.Setting} & {r.JSD:.2f} & {r.Corr:.2f} & {r.C2ST:.3f} & {r.AUC:.3f} & ${r.dAcc:+.2f}$ & ${r.dF1:+.2f}$ & {r.tokens:,.0f} \\\\"
             for r in table.itertuples()]
    (gpa.OUT / "sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table, pd.DataFrame(tests)


def main():
    print(dataset_table().round(1).to_string(index=False))
    t, f, c = generator_comparison()
    radar(pd.read_csv(gpa.OUT / "generators10_f1.csv", index_col=0))
    print(t.round(2).to_string(index=False)); print(f.to_string(index=False))
    print(c[c.p_holm < 0.05].to_string(index=False))
    s, sf = sensitivity()
    print(s.round(3).to_string(index=False)); print(sf.to_string(index=False))


if __name__ == "__main__":
    main()
