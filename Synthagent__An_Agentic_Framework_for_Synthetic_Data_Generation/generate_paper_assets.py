"""Reproduce the manuscript tables, figures, and statistical comparisons.

The script deliberately excludes model-sweep arms that do not contain all six
datasets with ten completed repetitions.  It reads only the September 2026
ablation run and the source CSV datasets; no values are entered by hand.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, t, wilcoxon


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN = REPO / "src" / "TabuLLM" / "results" / "ablation_20260918_114707"
DATA = REPO / "src" / "TabuLLM" / "data" / "medical"
OUT = HERE / "generated"
OUT.mkdir(exist_ok=True)

DATASET_LABELS = {
    "heart_disease.csv": "Heart disease",
    "breast_cancer_wisconsin_diagnostic.csv": "Breast cancer",
    "heart_failure_clinical_records.csv": "Heart failure",
    "indian_diabetes.csv": "Diabetes",
    "indian_liver_patient_dataset.csv": "Liver disease",
    "parkinsons.csv": "Parkinson's",
}
TARGETS = {
    "heart_disease.csv": "num",
    "breast_cancer_wisconsin_diagnostic.csv": "Diagnosis",
    "heart_failure_clinical_records.csv": "death_event",
    "indian_diabetes.csv": "Outcome",
    "indian_liver_patient_dataset.csv": "Selector",
    "parkinsons.csv": "status",
}
MODELS = ["rf", "dt", "mlp", "svm", "lr", "knn", "nb", "gb", "ab", "xgb"]
MODEL_LABELS = {
    "rf": "RF", "dt": "DT", "mlp": "MLP", "svm": "SVM", "lr": "LR",
    "knn": "kNN", "nb": "NB", "gb": "GB", "ab": "AdaBoost", "xgb": "XGBoost",
}
ARM_LABELS = {
    "full": "Synthagent",
    "no_rule_injection": "No rule injection",
    "no_post_processing": "No post-processing",
    "no_context_join": "Synthetic-only context arm",
    "strategy_dense": "Dense context",
    "strategy_diverse": "Diverse context",
    "ponder_strategy": "LLM-selected strategy",
    "smote": "SMOTE",
    "backend_gpt_4_1_mini": "GPT-4.1 mini",
    "backend_gpt_4o": "GPT-4o",
    "backend_gpt_4_1": "GPT-4.1",
}
ARCH_ARMS = [
    "full", "no_rule_injection", "no_post_processing", "no_context_join",
    "strategy_dense", "strategy_diverse", "ponder_strategy", "smote",
]
BACKEND_ARMS = ["full", "backend_gpt_4_1_mini", "backend_gpt_4o", "backend_gpt_4_1"]


def completed_arms(summary: pd.DataFrame) -> list[str]:
    checkpoint = json.loads((RUN / "checkpoint.json").read_text(encoding="utf-8"))
    done = set(checkpoint["completed"])
    expected_ds = set(DATASET_LABELS)
    valid = []
    for arm, group in summary.groupby("arm"):
        datasets = set(group["dataset"])
        keys = {f"{arm}::{d}" for d in expected_ds}
        if datasets == expected_ds and keys.issubset(done) and (group["folds"] == 10).all():
            valid.append(arm)
    return sorted(valid)


def add_utility_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in ("accuracy", "f1"):
        for regime in ("real", "gen", "aug", "delta", "delta_gen"):
            cols = [f"{m}_{metric}_{regime}" for m in MODELS]
            df[f"mean_{metric}_{regime}"] = df[cols].mean(axis=1, skipna=True)
    return df


def refresh_utility_from_per_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute ML aggregates from the canonical per-repetition files.

    One early summary writer left the no-context arm's ML aggregate cells empty,
    although its ten per-repetition files contain the evaluations.  Re-aggregating
    every complete arm here both repairs that omission and gives all arms the same
    numerical path into the paper.
    """
    df = df.copy()
    regime_map = {"real": "original", "gen": "generated", "aug": "augmented"}
    for idx, row in df.iterrows():
        stem = Path(row.dataset).stem
        fold_path = RUN / row.arm / f"{stem}_per_fold.csv"
        if not fold_path.exists():
            continue
        fold = pd.read_csv(fold_path)
        # Fidelity, attacks, and generation telemetry use the same names in the
        # per-repetition files, with ``_mean`` appended in the summary.
        direct_metrics = [
            "Gen Synthetic Fidelity (JSD)", "Gen Correlation Stability",
            "Gen Distribution Consistency", "Gen Entropy Preservation",
            "Gen Mean Z Score Drift", "Gen Categorical Fidelity",
            "Gen Privacy Risk", "Gen Data Density", "Gen Coverage Score",
            "Gen Feature Accuracy", "Gen Novelty Score", "Gen C2ST Accuracy",
            "Gen MIA Attack AUC",
            "Gen Synthetic Fidelity (JSD) (minority)",
            "Gen Synthetic Fidelity (JSD) (majority)",
        ]
        for metric in direct_metrics:
            if metric in fold:
                df.at[idx, f"{metric}_mean"] = fold[metric].mean()
                df.at[idx, f"{metric}_std"] = fold[metric].std(ddof=0)
        for metric in ("gen_stats_prompt_tokens", "gen_stats_candidates_tokens",
                       "gen_stats_total_tokens", "gen_stats_rule_violations",
                       "gen_stats_repair_attempts", "gen_stats_generation_seconds",
                       "fold_seconds"):
            if metric in fold:
                df.at[idx, f"{metric}_mean"] = fold[metric].mean()
                df.at[idx, f"{metric}_std"] = fold[metric].std(ddof=0)
        for model in MODELS:
            for metric in ("accuracy", "f1"):
                means = {}
                sds = {}
                for short, long_name in regime_map.items():
                    source = f"{model}_{metric}_{long_name}"
                    if source in fold:
                        means[short] = fold[source].mean()
                        sds[short] = fold[source].std(ddof=0)
                        df.at[idx, f"{model}_{metric}_{short}"] = means[short]
                        df.at[idx, f"{model}_{metric}_{short}_sd"] = sds[short]
                if {"real", "aug"}.issubset(means):
                    delta = fold[f"{model}_{metric}_augmented"] - fold[f"{model}_{metric}_original"]
                    df.at[idx, f"{model}_{metric}_delta"] = delta.mean()
                    df.at[idx, f"{model}_{metric}_delta_sd"] = delta.std(ddof=0)
                if {"real", "gen"}.issubset(means):
                    delta = fold[f"{model}_{metric}_generated"] - fold[f"{model}_{metric}_original"]
                    df.at[idx, f"{model}_{metric}_delta_gen"] = delta.mean()
                    df.at[idx, f"{model}_{metric}_delta_gen_sd"] = delta.std(ddof=0)
    return df


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for j, idx in enumerate(order):
        running = max(running, (n - j) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def dataset_block(df: pd.DataFrame, arms: list[str], metric: str) -> pd.DataFrame:
    col = f"mean_{metric}_aug"
    return (df[df.arm.isin(arms)]
            .pivot(index="dataset", columns="arm", values=col)
            .reindex(index=DATASET_LABELS, columns=arms))


def rank_and_test(df: pd.DataFrame, arms: list[str], metric: str, family: str):
    block = dataset_block(df, arms, metric).dropna()
    ranks = np.vstack([rankdata(-row, method="average") for row in block.to_numpy()])
    avg_ranks = pd.Series(ranks.mean(axis=0), index=block.columns).sort_values()
    stat, p_omnibus = friedmanchisquare(*(block[a].to_numpy() for a in arms))
    best = avg_ranks.index[0]
    comparisons = []
    raw_ps = []
    for arm in arms:
        if arm == best:
            continue
        diff = block[best] - block[arm]
        try:
            w, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
        except ValueError:
            w, p = 0.0, 1.0
        raw_ps.append(float(p))
        comparisons.append({
            "family": family,
            "metric": metric,
            "best": best,
            "comparison": arm,
            "mean_difference": float(diff.mean()),
            "median_difference": float(diff.median()),
            "wilcoxon_w": float(w),
            "p_raw": float(p),
        })
    adjusted = holm_adjust(raw_ps)
    for row, p_holm in zip(comparisons, adjusted):
        row["p_holm"] = p_holm
    omnibus = {
        "family": family, "metric": metric, "n_blocks": len(block),
        "n_arms": len(arms), "friedman_chi2": float(stat),
        "friedman_p": float(p_omnibus), "best": best,
    }
    return avg_ranks, omnibus, comparisons


def save_dataset_table():
    rows = []
    for filename, label in DATASET_LABELS.items():
        raw = pd.read_csv(DATA / filename)
        target = TARGETS[filename]
        counts = raw[target].value_counts(dropna=False)
        minority = int(counts.min())
        rows.append({
            "Dataset": label,
            "Rows": len(raw),
            "Features": len(raw.columns) - 1,
            "Target": target.replace("_", r"\_"),
            "Classes": raw[target].nunique(dropna=True),
            "Minority (\\%)": 100 * minority / len(raw),
            "Missing (\\%)": 100 * raw.isna().sum().sum() / raw.size,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "datasets.csv", index=False)
    tex = table.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.1f}")
    (OUT / "datasets.tex").write_text(tex, encoding="utf-8")


def save_full_results(df: pd.DataFrame):
    full = df[df.arm == "full"].copy()
    full["Dataset"] = full.dataset.map(DATASET_LABELS)
    out = pd.DataFrame({
        "Dataset": full["Dataset"],
        "JSD": full["Gen Synthetic Fidelity (JSD)_mean"],
        "Corr.": full["Gen Correlation Stability_mean"],
        "C2ST": full["Gen C2ST Accuracy_mean"],
        "MIA": full["Gen MIA Attack AUC_mean"],
        "Acc. real": full["mean_accuracy_real"],
        "Acc. aug.": full["mean_accuracy_aug"],
        "$\\Delta$Acc. (pp)": 100 * full["mean_accuracy_delta"],
        "$\\Delta$F1 (pp)": 100 * full["mean_f1_delta"],
    }).sort_values("Dataset")
    mean = {c: out[c].mean() for c in out.columns if c != "Dataset"}
    mean["Dataset"] = "Macro mean"
    out = pd.concat([out, pd.DataFrame([mean])], ignore_index=True)
    out.to_csv(OUT / "full_results.csv", index=False)
    (OUT / "full_results.tex").write_text(
        out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"), encoding="utf-8")


def save_real_baselines(df: pd.DataFrame):
    """Export every real-only classifier baseline with repetition variability."""
    full = df[df.arm == "full"].set_index("dataset")
    dataset_order = list(DATASET_LABELS)
    short = {
        "heart_disease.csv": "Heart",
        "breast_cancer_wisconsin_diagnostic.csv": "Breast",
        "heart_failure_clinical_records.csv": "Failure",
        "indian_diabetes.csv": "Diabetes",
        "indian_liver_patient_dataset.csv": "Liver",
        "parkinsons.csv": "Parkinson",
    }
    records = []
    tex_rows = []
    for model in MODELS:
        for metric, metric_label in (("accuracy", "Acc."), ("f1", "F1")):
            cells = []
            for dataset in dataset_order:
                mean = full.loc[dataset, f"{model}_{metric}_real"]
                std = full.loc[dataset, f"{model}_{metric}_real_sd"]
                records.append({
                    "model": MODEL_LABELS[model], "metric": metric,
                    "dataset": DATASET_LABELS[dataset], "mean": mean, "std": std,
                })
                cells.append("n/a" if pd.isna(mean) else f"{100 * mean:.1f} $\\pm$ {100 * std:.1f}")
            tex_rows.append(
                f"{MODEL_LABELS[model]} & {metric_label} & " + " & ".join(cells) + r" \\"
            )
        if model != MODELS[-1]:
            tex_rows.append(r"\addlinespace[1pt]")
    pd.DataFrame(records).to_csv(OUT / "real_only_baselines.csv", index=False)
    header = "Model & Metric & " + " & ".join(short[d] for d in dataset_order) + r" \\"
    tex = "\n".join([
        r"\begin{tabular}{llrrrrrr}", r"\toprule", header, r"\midrule",
        *tex_rows, r"\bottomrule", r"\end{tabular}", "",
    ])
    (OUT / "real_only_baselines.tex").write_text(tex, encoding="utf-8")


def save_arm_tables(df: pd.DataFrame):
    rows = []
    for arm in ARCH_ARMS:
        g = df[df.arm == arm]
        rows.append({
            "Configuration": ARM_LABELS[arm],
            "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "Corr.": g["Gen Correlation Stability_mean"].mean(),
            "C2ST": g["Gen C2ST Accuracy_mean"].mean(),
            "MIA": g["Gen MIA Attack AUC_mean"].mean(),
            "$\\Delta$Acc. (pp)": 100 * g["mean_accuracy_delta"].mean(),
            "$\\Delta$F1 (pp)": 100 * g["mean_f1_delta"].mean(),
            "Seconds": g["gen_stats_generation_seconds_mean"].mean(),
            "Tokens": g["gen_stats_total_tokens_mean"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ablation_summary.csv", index=False)
    (OUT / "ablation_summary.tex").write_text(
        out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.2f}"), encoding="utf-8")

    rows = []
    for arm in BACKEND_ARMS:
        g = df[df.arm == arm]
        rows.append({
            "Backend": "GPT-4o mini" if arm == "full" else ARM_LABELS[arm],
            "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "Corr.": g["Gen Correlation Stability_mean"].mean(),
            "C2ST": g["Gen C2ST Accuracy_mean"].mean(),
            "MIA": g["Gen MIA Attack AUC_mean"].mean(),
            "$\\Delta$Acc. (pp)": 100 * g["mean_accuracy_delta"].mean(),
            "$\\Delta$F1 (pp)": 100 * g["mean_f1_delta"].mean(),
            "Seconds": g["gen_stats_generation_seconds_mean"].mean(),
            "Tokens": g["gen_stats_total_tokens_mean"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "backend_summary.csv", index=False)
    (OUT / "backend_summary.tex").write_text(
        out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.2f}"), encoding="utf-8")


def plot_utility_heatmap(df: pd.DataFrame):
    full = df[df.arm == "full"].set_index("dataset")
    acc = pd.DataFrame({MODEL_LABELS[m]: 100 * full[f"{m}_accuracy_delta"] for m in MODELS}).T
    f1 = pd.DataFrame({MODEL_LABELS[m]: 100 * full[f"{m}_f1_delta"] for m in MODELS}).T
    acc.columns = [DATASET_LABELS[c] for c in acc.columns]
    f1.columns = [DATASET_LABELS[c] for c in f1.columns]
    vmax = np.nanmax(np.abs(np.r_[acc.to_numpy().ravel(), f1.to_numpy().ravel()]))
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.3), constrained_layout=True)
    for ax, mat, title in zip(axes, (acc, f1), ("Accuracy change", "F1 change")):
        sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    annot=True, fmt=".1f", linewidths=.5, cbar_kws={"label": "percentage points"}, ax=ax)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.savefig(OUT / "utility_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "utility_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_landscape(df: pd.DataFrame):
    """Show both baseline level and split sensitivity for every model-task pair."""
    full = df[df.arm == "full"].set_index("dataset")
    means = pd.DataFrame({
        DATASET_LABELS[d]: [100 * full.loc[d, f"{m}_accuracy_real"] for m in MODELS]
        for d in DATASET_LABELS
    }, index=[MODEL_LABELS[m] for m in MODELS])
    stds = pd.DataFrame({
        DATASET_LABELS[d]: [100 * full.loc[d, f"{m}_accuracy_real_sd"] for m in MODELS]
        for d in DATASET_LABELS
    }, index=[MODEL_LABELS[m] for m in MODELS])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    sns.heatmap(means, ax=axes[0], cmap="YlGnBu", vmin=45, vmax=100,
                annot=True, fmt=".1f", linewidths=.5,
                cbar_kws={"label": "accuracy (%)", "shrink": .82})
    sns.heatmap(stds, ax=axes[1], cmap="magma_r", vmin=0,
                annot=True, fmt=".1f", linewidths=.5,
                cbar_kws={"label": "standard deviation (pp)", "shrink": .82})
    axes[0].set_title("Real-only accuracy", loc="left", fontweight="bold")
    axes[1].set_title("Sensitivity to the data split", loc="left", fontweight="bold")
    for ax in axes:
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=32)
    fig.savefig(OUT / "baseline_landscape.pdf", bbox_inches="tight")
    fig.savefig(OUT / "baseline_landscape.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_augmentation_uncertainty():
    """Plot holdout-level utility changes after averaging over classifiers."""
    records = []
    for dataset, label in DATASET_LABELS.items():
        fold = pd.read_csv(RUN / "full" / f"{Path(dataset).stem}_per_fold.csv")
        for metric in ("accuracy", "f1"):
            for _, row in fold.iterrows():
                deltas = [
                    row[f"{m}_{metric}_augmented"] - row[f"{m}_{metric}_original"]
                    for m in MODELS
                    if pd.notna(row.get(f"{m}_{metric}_augmented"))
                    and pd.notna(row.get(f"{m}_{metric}_original"))
                ]
                records.append({
                    "dataset": label, "metric": metric,
                    "fold": int(row["fold"]), "delta": 100 * np.mean(deltas),
                })
    values = pd.DataFrame(records)
    order = list(DATASET_LABELS.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True, sharey=True)
    palette = sns.color_palette("colorblind", len(order))
    for ax, metric, title in zip(axes, ("accuracy", "f1"), ("Accuracy", "F1")):
        subset = values[values.metric == metric]
        for y, (dataset, color) in enumerate(zip(order, palette)):
            x = subset[subset.dataset == dataset].delta.to_numpy()
            mean = x.mean()
            ci = t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            jitter = np.linspace(-.12, .12, len(x))
            ax.scatter(x, y + jitter, s=18, color=color, alpha=.38, linewidth=0)
            ax.errorbar(mean, y, xerr=ci, fmt="D", markersize=5.5, color=color,
                        markeredgecolor="white", markeredgewidth=.7, capsize=3, lw=1.7)
        ax.axvline(0, color="0.35", lw=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("real + generated minus real only (pp)")
        ax.grid(axis="x", alpha=.25)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    fig.savefig(OUT / "augmentation_uncertainty.pdf", bbox_inches="tight")
    fig.savefig(OUT / "augmentation_uncertainty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_configuration_scorecard(df: pd.DataFrame):
    """Rank each completed architecture arm on every reported objective."""
    rows = []
    for arm in ARCH_ARMS:
        g = df[df.arm == arm]
        rows.append({
            "arm": ARM_LABELS[arm],
            "JSD fidelity": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "Correlation": g["Gen Correlation Stability_mean"].mean(),
            "C2ST closeness": -abs(g["Gen C2ST Accuracy_mean"].mean() - .5),
            "MIA closeness": -abs(g["Gen MIA Attack AUC_mean"].mean() - .5),
            "Accuracy change": g["mean_accuracy_delta"].mean(),
            "F1 change": g["mean_f1_delta"].mean(),
            "Speed": -g["gen_stats_generation_seconds_mean"].mean(),
            "Token economy": -g["gen_stats_total_tokens_mean"].mean(),
        })
    raw = pd.DataFrame(rows).set_index("arm")
    # The SMOTE arm contains only a synthetic minority tail, so its raw C2ST
    # accuracy has unequal class priors and must not be ranked against full-table
    # generators.
    raw.loc[ARM_LABELS["smote"], "C2ST closeness"] = np.nan
    ranks = raw.rank(axis=0, ascending=False, method="average")
    fig, ax = plt.subplots(figsize=(9.8, 4.3), constrained_layout=True)
    sns.heatmap(ranks, ax=ax, cmap="Blues_r", vmin=1, vmax=len(ARCH_ARMS),
                annot=True, fmt=".1f", linewidths=.6,
                cbar_kws={"label": "rank (1 = best)", "shrink": .8})
    ax.set(xlabel="Evaluation objective", ylabel="")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("No single configuration wins every objective", loc="left", fontweight="bold")
    fig.savefig(OUT / "configuration_scorecard.pdf", bbox_inches="tight")
    fig.savefig(OUT / "configuration_scorecard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ranks(rank_results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    for ax, (metric, ranks) in zip(axes, rank_results.items()):
        ranks = ranks.sort_values(ascending=False)
        labels = [ARM_LABELS[a] for a in ranks.index]
        colors = ["#1f77b4" if a == "full" else "#9ecae1" if a != "smote" else "#fdae6b" for a in ranks.index]
        ax.barh(labels, ranks.values, color=colors, edgecolor="white")
        ax.set_xlabel("Average rank (lower is better)")
        ax.set_title(metric.upper(), loc="left", fontweight="bold")
        ax.set_xlim(1, len(ranks))
        ax.grid(axis="x", alpha=.25)
    fig.savefig(OUT / "architecture_ranks.pdf", bbox_inches="tight")
    fig.savefig(OUT / "architecture_ranks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(df: pd.DataFrame):
    rows = []
    for arm in ARCH_ARMS:
        g = df[df.arm == arm]
        rows.append({
            "arm": arm,
            "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "F1": 100 * g["mean_f1_delta"].mean(),
            "C2ST": g["Gen C2ST Accuracy_mean"].mean(),
            "seconds": g["gen_stats_generation_seconds_mean"].mean(),
        })
    p = pd.DataFrame(rows)
    smote = p[p.arm == "smote"].iloc[0]
    p = p[p.arm != "smote"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    palette = sns.color_palette("colorblind", len(p))
    handles = []
    for i, row in p.iterrows():
        label = ARM_LABELS[row.arm]
        h = axes[0].scatter(row.JSD, row.F1, s=85, color=palette[i], edgecolor="white",
                            linewidth=.8, label=label)
        handles.append(h)
        axes[1].scatter(row.seconds, row.F1, s=85, color=palette[i], edgecolor="white", linewidth=.8)
    axes[0].set(xlabel="Generated-only JSD fidelity (0-100)", ylabel="Mean augmented F1 change (pp)")
    axes[1].set(xlabel="Mean generation time per repetition (s)", ylabel="Mean augmented F1 change (pp)")
    axes[0].set_xlim(p.JSD.min() - .12, p.JSD.max() + .12)
    axes[1].set_xlim(p.seconds.min() - 15, p.seconds.max() + 15)
    ymin, ymax = p.F1.min() - .18, p.F1.max() + .18
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    axes[0].text(.02, .04,
                 f"SMOTE reference: JSD {smote.JSD:.1f}, ΔF1 {smote.F1:+.2f} pp",
                 transform=axes[0].transAxes, fontsize=8, color="0.30",
                 bbox=dict(boxstyle="round,pad=.3", facecolor="white", edgecolor="0.75"))
    axes[1].text(.02, .04,
                 f"SMOTE reference: {smote.seconds:.3f} s, ΔF1 {smote.F1:+.2f} pp",
                 transform=axes[1].transAxes, fontsize=8, color="0.30",
                 bbox=dict(boxstyle="round,pad=.3", facecolor="white", edgecolor="0.75"))
    for ax, title in zip(axes, ("Quality and utility", "Cost and utility")):
        ax.axhline(0, color="0.5", lw=.8)
        ax.grid(alpha=.25)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.legend(handles=handles, labels=[ARM_LABELS[a] for a in p.arm], loc="outside lower center",
               ncol=4, frameon=False, fontsize=8)
    fig.set_constrained_layout_pads(h_pad=.13)
    fig.savefig(OUT / "quality_cost_frontier.pdf", bbox_inches="tight")
    fig.savefig(OUT / "quality_cost_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_backends(df: pd.DataFrame):
    records = []
    for arm in BACKEND_ARMS:
        g = df[df.arm == arm]
        for _, row in g.iterrows():
            records.append({"Backend": "GPT-4o mini" if arm == "full" else ARM_LABELS[arm],
                            "Dataset": DATASET_LABELS[row.dataset],
                            "Accuracy": 100 * row.mean_accuracy_delta,
                            "F1": 100 * row.mean_f1_delta,
                            "JSD": row["Gen Synthetic Fidelity (JSD)_mean"]})
    long = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ["Accuracy", "F1", "JSD"],
                                  ["Accuracy change (pp)", "F1 change (pp)", "JSD fidelity"]):
        sns.boxplot(data=long, x="Backend", y=metric, color="#9ecae1", width=.55, fliersize=0, ax=ax)
        sns.stripplot(data=long, x="Backend", y=metric, color="#17365d", size=4, jitter=.12, ax=ax)
        ax.tick_params(axis="x", rotation=28)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "backend_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "backend_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_minority(df: pd.DataFrame):
    sub = df[df.arm.isin(ARCH_ARMS)].copy()
    p = pd.DataFrame({
        "Configuration": sub.arm.map(ARM_LABELS),
        "Minority": sub["Gen Synthetic Fidelity (JSD) (minority)_mean"],
        "Majority": sub["Gen Synthetic Fidelity (JSD) (majority)_mean"],
    }).melt("Configuration", var_name="Class", value_name="JSD")
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    sns.pointplot(data=p, x="Configuration", y="JSD", hue="Class", dodge=.28,
                  errorbar=("ci", 95), markers=["o", "s"], linestyles="none", ax=ax)
    ax.tick_params(axis="x", rotation=28)
    ax.set_xlabel("")
    ax.set_ylabel("Generated-only JSD fidelity (0-100)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OUT / "minority_fidelity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "minority_fidelity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def full_cell_tests():
    """Per dataset/classifier tests of augmented minus real performance."""
    records = []
    for dataset in DATASET_LABELS:
        fold = pd.read_csv(RUN / "full" / f"{Path(dataset).stem}_per_fold.csv")
        for model in MODELS:
            for metric in ("accuracy", "f1"):
                aug_col = f"{model}_{metric}_augmented"
                real_col = f"{model}_{metric}_original"
                if aug_col not in fold or real_col not in fold:
                    continue
                delta = (fold[aug_col] - fold[real_col]).dropna()
                if delta.empty:
                    continue
                try:
                    if np.allclose(delta, 0):
                        w, p = 0.0, 1.0
                    else:
                        w, p = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
                except ValueError:
                    w, p = 0.0, 1.0
                records.append({
                    "dataset": dataset, "model": model, "metric": metric,
                    "mean_delta": delta.mean(), "median_delta": delta.median(),
                    "wilcoxon_w": float(w), "p_raw": float(p),
                })
    out = pd.DataFrame(records)
    adjusted = []
    for metric, group in out.groupby("metric", sort=False):
        vals = holm_adjust(group.p_raw.tolist())
        adjusted.extend(zip(group.index, vals))
    out["p_holm"] = np.nan
    for idx, val in adjusted:
        out.at[idx, "p_holm"] = val
    out["direction"] = np.where(out.mean_delta > 0, "positive",
                                 np.where(out.mean_delta < 0, "negative", "zero"))
    out.to_csv(OUT / "full_cell_wilcoxon_holm.csv", index=False)
    summary = (out.groupby(["metric", "direction"]).size().unstack(fill_value=0))
    summary["significant_positive"] = [
        ((out.metric == metric) & (out.mean_delta > 0) & (out.p_holm < .05)).sum()
        for metric in summary.index
    ]
    summary["significant_negative"] = [
        ((out.metric == metric) & (out.mean_delta < 0) & (out.p_holm < .05)).sum()
        for metric in summary.index
    ]
    summary.to_csv(OUT / "full_cell_test_summary.csv")


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    summary = pd.read_csv(RUN / "ablation_summary.csv")
    valid = completed_arms(summary)
    expected = set(ARCH_ARMS + BACKEND_ARMS)
    missing = expected - set(valid)
    if missing:
        raise RuntimeError(f"Required complete arms missing: {sorted(missing)}")
    summary = refresh_utility_from_per_fold(summary[summary.arm.isin(valid)].copy())
    summary = add_utility_columns(summary)
    summary.to_csv(OUT / "analysis_rows.csv", index=False)
    (OUT / "included_arms.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")

    save_dataset_table()
    save_full_results(summary)
    save_real_baselines(summary)
    save_arm_tables(summary)
    full_cell_tests()

    omnibus_rows = []
    comparison_rows = []
    architecture_ranks = {}
    rank_rows = []
    for family, arms in (("architecture", ARCH_ARMS), ("backend", BACKEND_ARMS)):
        for metric in ("accuracy", "f1"):
            ranks, omnibus, comparisons = rank_and_test(summary, arms, metric, family)
            omnibus_rows.append(omnibus)
            comparison_rows.extend(comparisons)
            for arm, rank in ranks.items():
                rank_rows.append({"family": family, "metric": metric, "arm": arm, "average_rank": rank})
            if family == "architecture":
                architecture_ranks[metric] = ranks
    pd.DataFrame(omnibus_rows).to_csv(OUT / "friedman_tests.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(OUT / "posthoc_wilcoxon_holm.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUT / "average_ranks.csv", index=False)

    plot_utility_heatmap(summary)
    plot_baseline_landscape(summary)
    plot_augmentation_uncertainty()
    plot_configuration_scorecard(summary)
    plot_ranks(architecture_ranks)
    plot_frontier(summary)
    plot_backends(summary)
    plot_minority(summary)


if __name__ == "__main__":
    main()
