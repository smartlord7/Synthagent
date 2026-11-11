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
RUN = REPO / "src" / "Synthagent" / "results" / "ablation_20260918_114707"
DATA = REPO / "src" / "Synthagent" / "data" / "medical"
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
    "no_rule_injection": "No inferred type rules",
    "no_post_processing": "No post-processing",
    "no_context_join": "No context join",
    "strategy_dense": "Dense context",
    "strategy_diverse": "Diverse context",
    "ponder_strategy": "LLM-selected hyperparameters",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "raw_prompting": "Raw prompting",
    "backend_gpt_4_1_mini": "GPT-4.1 mini",
    "backend_gpt_4o": "GPT-4o",
    "backend_gpt_4_1": "GPT-4.1",
    "full_logged": "Logged rerun",
    "boundaries": "Range enforcement",
    "structured_output": "Structured outputs",
    "backend_claude_haiku_4_5": "Claude Haiku 4.5",
    "backend_claude_sonnet_4_6": "Claude Sonnet 4.6",
    "backend_claude_sonnet_5_5": "Claude Sonnet 5.5",
    "raw_prompting_claude_haiku_5_5": "Direct prompting (Claude Haiku 5.5)",
    "structured_output_claude_haiku_5_5": "Structured outputs (Claude Haiku 5.5)",
}
ARCH_ARMS = [
    "full", "no_rule_injection", "no_post_processing", "no_context_join",
    "strategy_dense", "strategy_diverse", "ponder_strategy", "full_logged", "boundaries", "smote",
]
BACKEND_ARMS = ["full", "backend_gpt_4_1_mini", "backend_gpt_4o", "backend_gpt_4_1"]
# Non-LLM generators against the reference and the strongest backend.
BASELINE_ARMS = ["full", "backend_gpt_4_1", "smote", "ctgan", "tvae"]
# Reported only once all six datasets and ten repetitions are complete.
OPTIONAL_ARMS = ["raw_prompting"]
# The same LLM without the control layer, with free and schema-constrained decoding.
PROMPTING_ARMS = ["full", "raw_prompting", "structured_output"]
# Second provider, reference configuration only (Claude Haiku 5.5 refused most requests
# under the control layer and completed one dataset, so it is reported separately).
CLAUDE_ARMS = ["backend_claude_haiku_4_5", "backend_claude_sonnet_4_6", "backend_claude_sonnet_5_5"]
CLAUDE_PROMPTING_ARMS = ["raw_prompting_claude_haiku_5_5", "structured_output_claude_haiku_5_5"]


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
            "Gen MIA Attack AUC", "Gen Valid Rows",
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


def mask_unrun_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Drop XGBoost on liver disease from every arm.

    The LLM arms ran before XGBoost label re-encoding was added and have no XGBoost
    result on liver disease; later baseline runs do. Masking the cell everywhere
    keeps every arm averaged over the same 59 classifier-dataset cells.
    """
    df = df.copy()
    liver = df.dataset == "indian_liver_patient_dataset.csv"
    cols = [c for c in df.columns if c.startswith("xgb_")]
    df.loc[liver, cols] = np.nan
    return df


def mask_failed_generated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Treat generated-only metrics as missing where their computation failed.

    The SMOTE arm reports 0% generated-cell density, 0 JSD and placeholder
    correlation values on the two tables with missing cells (heart disease and
    liver disease), although augmented training ran.  Averaging those zeros
    would understate SMOTE fidelity, so these cells are excluded instead.
    """
    df = df.copy()
    failed = df["Gen Data Density_mean"] == 0
    gen_cols = [c for c in df.columns if c.startswith("Gen ")]
    df.loc[failed, gen_cols] = np.nan
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


def division_level_tests(arms: list[str], family: str):
    """Sensitivity analysis: the same tests with every division as a block.

    All arms share the divisions (random_state = 42 + division), so the 60
    dataset-division pairs are matched.  Divisions of one dataset overlap in
    their records, so these blocks are not independent and the dataset-level
    tests remain the primary analysis.  XGBoost on liver disease is masked as
    in mask_unrun_cells.
    """
    omnibus, comparisons = [], []
    for metric in ("accuracy", "f1"):
        aug, delta = {}, {}
        for arm in arms:
            a_vals, d_vals = [], []
            for dataset in DATASET_LABELS:
                fold = pd.read_csv(RUN / arm / f"{Path(dataset).stem}_per_fold.csv")
                models = [m for m in MODELS
                          if not (m == "xgb" and dataset == "indian_liver_patient_dataset.csv")]
                a = fold[[f"{m}_{metric}_augmented" for m in models]].mean(axis=1)
                r = fold[[f"{m}_{metric}_original" for m in models]].mean(axis=1)
                a_vals.extend(a)
                d_vals.extend(a - r)
            aug[arm], delta[arm] = a_vals, d_vals
        aug, delta = pd.DataFrame(aug).dropna(), pd.DataFrame(delta)
        delta = delta.loc[aug.index]
        ranks = np.vstack([rankdata(-row, method="average") for row in aug.to_numpy()])
        avg = pd.Series(ranks.mean(axis=0), index=arms).sort_values()
        stat, p = friedmanchisquare(*(aug[a].to_numpy() for a in arms))
        best = avg.index[0]
        omnibus.append({"family": family, "metric": metric, "n_blocks": len(aug),
                        "n_arms": len(arms), "friedman_chi2": float(stat),
                        "friedman_p": float(p), "best": best})
        rows, raw = [], []
        for arm in avg.index[1:]:
            diff = aug[best] - aug[arm]
            w, pr = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
            raw.append(float(pr))
            rows.append({"family": family, "metric": metric, "best": best, "comparison": arm,
                         "best_mean_delta_pp": 100 * delta[best].mean(),
                         "comparison_mean_delta_pp": 100 * delta[arm].mean(),
                         "mean_difference_pp": 100 * diff.mean(),
                         "median_difference_pp": 100 * diff.median(),
                         "average_rank": float(avg[arm]), "wilcoxon_w": float(w), "p_raw": float(pr)})
        for row, ph in zip(rows, holm_adjust(raw)):
            row["p_holm"] = ph
        comparisons.extend(rows)
    return omnibus, comparisons


# ---------------------------------------------------------------- print styling
# Figures are drawn at their printed size (IEEE column 3.5 in, text 7.16 in) and
# included at 100% scale, so the font sizes below are the sizes on paper.
COL_W, TEXT_W = 3.5, 7.16
SHORT_DS = {
    "heart_disease.csv": "Heart dis.",
    "breast_cancer_wisconsin_diagnostic.csv": "Breast",
    "heart_failure_clinical_records.csv": "Heart fail.",
    "indian_diabetes.csv": "Diabetes",
    "indian_liver_patient_dataset.csv": "Liver",
    "parkinsons.csv": "Parkinson's",
}
SHORT_ARM = {
    "full": "Synthagent (ref.)",
    "no_rule_injection": "No type rules",
    "no_post_processing": "No post-proc.",
    "no_context_join": "No context join",
    "strategy_dense": "Dense context",
    "strategy_diverse": "Diverse context",
    "ponder_strategy": "LLM-chosen settings",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "raw_prompting": "Direct prompting",
    "structured_output": "Structured outputs",
    "full_logged": "Logged rerun",
    "boundaries": "Range enforcement",
    "backend_gpt_4_1_mini": "GPT-4.1 mini",
    "backend_gpt_4o": "GPT-4o",
    "backend_gpt_4_1": "GPT-4.1",
    "backend_claude_haiku_4_5": "Claude Haiku 4.5",
    "backend_claude_sonnet_4_6": "Claude Sonnet 4.6",
    "backend_claude_sonnet_5_5": "Claude Sonnet 5.5",
}
NON_LLM = {"smote", "ctgan", "tvae"}
C_REF, C_LLM, C_BASE, C_BACK = "#1f4e79", "#7fb3d5", "#e67e22", "#2e8b57"
C_NOCTL = "#8e44ad"
NO_CONTROL = {"raw_prompting", "structured_output"}


def print_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": .6, "grid.linewidth": .4, "lines.linewidth": 1.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def arm_color(arm: str) -> str:
    if arm == "full":
        return C_REF
    if arm in NON_LLM:
        return C_BASE
    if arm.startswith("backend_"):
        return C_BACK
    if arm in NO_CONTROL:
        return C_NOCTL
    return C_LLM


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def save_real_baselines(df: pd.DataFrame):
    """Real-only baselines: one row per classifier, Acc./F1 pairs per dataset."""
    full = df[df.arm == "full"].set_index("dataset")
    records = []
    body = []
    for model in MODELS:
        cells = []
        for dataset in DATASET_LABELS:
            for metric in ("accuracy", "f1"):
                mean = full.loc[dataset, f"{model}_{metric}_real"]
                std = full.loc[dataset, f"{model}_{metric}_real_sd"]
                records.append({"model": MODEL_LABELS[model], "metric": metric,
                                "dataset": DATASET_LABELS[dataset], "mean": mean, "std": std})
                cells.append("\\multicolumn{1}{c}{--}" if pd.isna(mean)
                             else f"{100 * mean:.1f}\\sd{{{100 * std:.1f}}}")
        body.append(f"{MODEL_LABELS[model]} & " + " & ".join(cells) + r" \\")
    pd.DataFrame(records).to_csv(OUT / "real_only_baselines.csv", index=False)
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT_DS[d]}}}" for d in DATASET_LABELS)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASET_LABELS)))
    sub = " & ".join(["Acc.", "F1"] * len(DATASET_LABELS))
    tex = "\n".join([
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "rr" * len(DATASET_LABELS) + "@{}}",
        r"\toprule", f" & {groups} \\\\", rules, f"Classifier & {sub} \\\\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular*}", "",
    ])
    (OUT / "real_only_baselines.tex").write_text(tex, encoding="utf-8")


def plot_utility_heatmap(df: pd.DataFrame):
    full = df[df.arm == "full"].set_index("dataset")
    mats = {}
    for metric, title in (("accuracy", "Accuracy change (pp)"), ("f1", "F1 change (pp)")):
        mat = pd.DataFrame({SHORT_DS[d]: [100 * full.loc[d, f"{m}_{metric}_delta"] for m in MODELS]
                            for d in DATASET_LABELS}, index=[MODEL_LABELS[m] for m in MODELS])
        mats[title] = mat
    vmax = max(np.nanmax(np.abs(m.to_numpy())) for m in mats.values())
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), sharey=True,
                             gridspec_kw={"wspace": .04})
    cbar_ax = fig.add_axes([.915, .2, .012, .62])
    for i, (ax, (title, mat)) in enumerate(zip(axes, mats.items())):
        sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    annot=True, fmt=".1f", annot_kws={"size": 6.3}, linewidths=.4,
                    linecolor="white", cbar=i == 1, cbar_ax=cbar_ax if i == 1 else None,
                    cbar_kws={"label": "percentage points"})
        for (r, c), v in np.ndenumerate(mat.to_numpy()):
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, facecolor="0.93",
                                           hatch="////", edgecolor="0.75", lw=0))
                ax.text(c + .5, r + .5, "n/a", ha="center", va="center", fontsize=6, color="0.4")
        ax.set_title(title, loc="left", pad=3)
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=0, length=0)
        ax.tick_params(axis="y", rotation=0, length=0)
    cbar_ax.tick_params(labelsize=6, length=2)
    cbar_ax.yaxis.label.set_size(6.5)
    fig.subplots_adjust(left=.07, right=.9, top=.9, bottom=.1)
    save(fig, "utility_heatmap")


def plot_augmentation_uncertainty():
    """Holdout-level utility changes after averaging over classifiers."""
    records = []
    for dataset in DATASET_LABELS:
        fold = pd.read_csv(RUN / "full" / f"{Path(dataset).stem}_per_fold.csv")
        for metric in ("accuracy", "f1"):
            for _, row in fold.iterrows():
                deltas = [row[f"{m}_{metric}_augmented"] - row[f"{m}_{metric}_original"]
                          for m in MODELS
                          if pd.notna(row.get(f"{m}_{metric}_augmented"))
                          and pd.notna(row.get(f"{m}_{metric}_original"))]
                records.append({"dataset": SHORT_DS[dataset], "metric": metric,
                                "delta": 100 * np.mean(deltas)})
    values = pd.DataFrame(records)
    order = [SHORT_DS[d] for d in DATASET_LABELS]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.15), sharey=True, gridspec_kw={"wspace": .08})
    palette = sns.color_palette("colorblind", len(order))
    for ax, metric, title in zip(axes, ("accuracy", "f1"), ("Accuracy", "F1")):
        sub = values[values.metric == metric]
        for y, (dataset, color) in enumerate(zip(order, palette)):
            x = sub[sub.dataset == dataset].delta.to_numpy()
            ci = t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            ax.scatter(x, y + np.linspace(-.18, .18, len(x)), s=7, color=color, alpha=.45, lw=0)
            ax.errorbar(x.mean(), y, xerr=ci, fmt="D", ms=3.6, color=color, mec="white",
                        mew=.5, capsize=2, lw=1.1)
        ax.axvline(0, color="0.3", lw=.7)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("change vs. real only (pp)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.2, right=.99, top=.9, bottom=.2)
    save(fig, "augmentation_uncertainty")


def plot_configuration_scorecard(df: pd.DataFrame):
    """Rank each architecture arm on every reported objective."""
    rows = []
    for arm in ARCH_ARMS:
        g = df[df.arm == arm]
        rows.append({
            "arm": SHORT_ARM[arm],
            "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "Corr.": g["Gen Correlation Stability_mean"].mean(),
            "C2ST": -abs(g["Gen C2ST Accuracy_mean"].mean() - .5),
            "MIA": -abs(g["Gen MIA Attack AUC_mean"].mean() - .5),
            "ΔAcc.": g["mean_accuracy_delta"].mean(),
            "ΔF1": g["mean_f1_delta"].mean(),
            "Time": -g["gen_stats_generation_seconds_mean"].mean(),
            "Tokens": -g["gen_stats_total_tokens_mean"].mean(),
        })
    raw = pd.DataFrame(rows).set_index("arm")
    # SMOTE emits only a synthetic minority tail; its raw C2ST is not commensurate.
    raw.loc[SHORT_ARM["smote"], "C2ST"] = np.nan
    ranks = raw.rank(axis=0, ascending=False, method="average")
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    sns.heatmap(ranks, ax=ax, cmap="Blues_r", vmin=1, vmax=len(ARCH_ARMS), annot=True,
                fmt=".1f", annot_kws={"size": 6.3}, linewidths=.4, cbar=False)
    for (r, c), v in np.ndenumerate(ranks.to_numpy()):
        if np.isnan(v):
            ax.text(c + .5, r + .5, "n/a", ha="center", va="center", fontsize=6, color="0.4")
    ax.set(xlabel="", ylabel="")
    ax.grid(False)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, rotation=0)
    save(fig, "configuration_scorecard")


def plot_ranks(rank_results):
    acc, f1 = rank_results["accuracy"], rank_results["f1"]
    order = ((acc + f1) / 2).sort_values().index[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    y = np.arange(len(order))
    ax.hlines(y, [min(acc[a], f1[a]) for a in order], [max(acc[a], f1[a]) for a in order],
              color="0.75", lw=1.2, zorder=1)
    ax.scatter([acc[a] for a in order], y, marker="o", s=22, color=C_REF, label="Accuracy", zorder=2)
    ax.scatter([f1[a] for a in order], y, marker="s", s=20, facecolor="white",
               edgecolor=C_BASE, lw=1.1, label="F1", zorder=3)
    ax.set_yticks(y, [SHORT_ARM[a] for a in order])
    for lab, arm in zip(ax.get_yticklabels(), order):
        if arm == "full":
            lab.set_fontweight("bold")
        if arm in NON_LLM:
            lab.set_color(C_BASE)
    ax.set_xlim(.8, len(order) + .2)
    ax.set_xlabel("Average rank over six datasets (1 = best)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True, framealpha=.95, borderpad=.3, handletextpad=.2)
    fig.subplots_adjust(left=.3, right=.98, top=.97, bottom=.2)
    save(fig, "architecture_ranks")


def plot_frontier(df: pd.DataFrame):
    """Fidelity, utility and token cost of every complete LLM arm; baselines noted."""
    llm_arms = [a for a in ARCH_ARMS + BACKEND_ARMS[1:] + PROMPTING_ARMS[1:] if a not in NON_LLM]
    rows = []
    for arm in llm_arms + [a for a in BASELINE_ARMS if a in NON_LLM]:
        g = df[df.arm == arm]
        rows.append({"arm": arm, "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
                     "F1": 100 * g["mean_f1_delta"].mean(),
                     "tokens": g["gen_stats_total_tokens_mean"].mean()})
    p = pd.DataFrame(rows).set_index("arm")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True, gridspec_kw={"wspace": .08})
    for arm in llm_arms:
        r = p.loc[arm]
        kw = dict(s=24, color=arm_color(arm), edgecolor="white", lw=.5, zorder=3)
        axes[0].scatter(r.JSD, r.F1, **kw)
        axes[1].scatter(r.tokens / 1000, r.F1, **kw)
    for ax in axes:
        ax.axhline(0, color="0.4", lw=.6)
    # Direct labels for the points that matter; the rest are the noise-floor cloud.
    left = {"backend_gpt_4_1"}
    for arm in ("full", "backend_gpt_4_1", "backend_gpt_4_1_mini", "backend_gpt_4o", "ponder_strategy",
                "structured_output"):
        r = p.loc[arm]
        name = SHORT_ARM[arm].replace(" (ref.)", "")
        axes[0].annotate(name, (r.JSD, r.F1), xytext=(-3 if arm in left else 3, 1),
                         textcoords="offset points", ha="right" if arm in left else "left",
                         va="center", fontsize=5.8, color=arm_color(arm))
    r = p.loc["ponder_strategy"]
    axes[1].annotate("LLM-chosen\nsettings", (r.tokens / 1000, r.F1), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=5.8,
                     color=arm_color("ponder_strategy"))
    r = p.loc["structured_output"]
    axes[1].annotate("Structured\noutputs", (r.tokens / 1000, r.F1), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=5.8,
                     color=arm_color("structured_output"))
    for ax in axes:
        ax.margins(x=.12, y=.12)
    axes[0].set(xlabel="Generated-only JSD (0–100)", ylabel="Mean ΔF1 (pp)")
    axes[1].set(xlabel="Tokens per repetition (thousands)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=4, label=l)
               for c, l in ((C_REF, "Reference"), (C_LLM, "Control-layer variants"), (C_BACK, "LLM variants"),
                                (C_NOCTL, "No control layer"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.55, 1.12), handletextpad=.05, columnspacing=1.0)
    fig.subplots_adjust(left=.14, right=.99, top=.9, bottom=.18)
    save(fig, "quality_cost_frontier")


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

    save_baseline_table(df)

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


def baseline_label(arm: str) -> str:
    if arm == "full":
        return "Synthagent (GPT-4o mini)"
    if arm == "backend_gpt_4_1":
        return "Synthagent (GPT-4.1)"
    return ARM_LABELS[arm]


def save_baseline_table(df: pd.DataFrame):
    """Summary and per-dataset F1 change for the generator comparison (RQ5)."""
    arms = BASELINE_ARMS + [a for a in OPTIONAL_ARMS if a in set(df.arm)]
    rows, per_ds, valid_ds = [], [], []
    for arm in arms:
        g = df[df.arm == arm]
        rows.append({
            "Generator": baseline_label(arm),
            "JSD": g["Gen Synthetic Fidelity (JSD)_mean"].mean(),
            "Corr.": g["Gen Correlation Stability_mean"].mean(),
            "C2ST": g["Gen C2ST Accuracy_mean"].mean(),
            "MIA": g["Gen MIA Attack AUC_mean"].mean(),
            # Logged only for runs made after the metric was added (the
            # non-LLM generators); NaN for the LLM arms.
            "Valid (%)": (g["Gen Valid Rows_mean"].mean()
                          if "Gen Valid Rows_mean" in g else np.nan),
            "$\\Delta$Acc. (pp)": 100 * g["mean_accuracy_delta"].mean(),
            "$\\Delta$F1 (pp)": 100 * g["mean_f1_delta"].mean(),
            "Seconds": g["gen_stats_generation_seconds_mean"].mean(),
            "Tokens": g["gen_stats_total_tokens_mean"].mean(),
        })
        rec = {"Generator": baseline_label(arm)}
        for dataset, label in DATASET_LABELS.items():
            cell = g[g.dataset == dataset]
            rec[label] = 100 * cell["mean_f1_delta"].iloc[0] if len(cell) else np.nan
        per_ds.append(rec)
        vrec = {"Generator": baseline_label(arm)}
        for dataset, label in DATASET_LABELS.items():
            cell = g[g.dataset == dataset]
            vrec[label] = (cell["Gen Valid Rows_mean"].iloc[0]
                           if len(cell) and "Gen Valid Rows_mean" in cell else np.nan)
        valid_ds.append(vrec)
    out = pd.DataFrame(rows)
    pd.DataFrame(valid_ds).to_csv(OUT / "baseline_validity_by_dataset.csv", index=False)
    out.to_csv(OUT / "baseline_summary.csv", index=False)
    (OUT / "baseline_summary.tex").write_text(
        out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.2f}"), encoding="utf-8")
    pd.DataFrame(per_ds).to_csv(OUT / "baseline_f1_by_dataset.csv", index=False)




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
    print_style()
    summary = pd.read_csv(RUN / "ablation_summary.csv")
    # This script covers the original six datasets; generate_extended_assets.py covers
    # the ten-dataset comparison and the sensitivity study.
    summary = summary[summary.dataset.isin(DATASET_LABELS)].copy()
    valid = completed_arms(summary)
    expected = set(ARCH_ARMS + BACKEND_ARMS + BASELINE_ARMS + PROMPTING_ARMS + CLAUDE_ARMS + CLAUDE_PROMPTING_ARMS)
    missing = expected - set(valid)
    if missing:
        raise RuntimeError(f"Required complete arms missing: {sorted(missing)}")
    summary = refresh_utility_from_per_fold(summary[summary.arm.isin(valid)].copy())
    summary = mask_failed_generated_metrics(summary)
    summary = mask_unrun_cells(summary)
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
    for family, arms in (("architecture", ARCH_ARMS), ("backend", BACKEND_ARMS),
                         ("baseline", BASELINE_ARMS), ("prompting", PROMPTING_ARMS),
                         ("claude", CLAUDE_ARMS), ("all_llms", BACKEND_ARMS + CLAUDE_ARMS)):
        for metric in ("accuracy", "f1"):
            ranks, omnibus, comparisons = rank_and_test(summary, arms, metric, family)
            omnibus_rows.append(omnibus)
            comparison_rows.extend(comparisons)
            for arm, rank in ranks.items():
                rank_rows.append({"family": family, "metric": metric, "arm": arm, "average_rank": rank})
            if family == "architecture":
                architecture_ranks[metric] = ranks
    pd.DataFrame(omnibus_rows).to_csv(OUT / "friedman_tests.csv", index=False)
    div_omnibus, div_comparisons = [], []
    for family, arms in (("backend", BACKEND_ARMS), ("all_llms", BACKEND_ARMS + CLAUDE_ARMS)):
        o, c = division_level_tests(arms, family)
        div_omnibus.extend(o)
        div_comparisons.extend(c)
    pd.DataFrame(div_omnibus).to_csv(OUT / "friedman_tests_divisions.csv", index=False)
    pd.DataFrame(div_comparisons).to_csv(OUT / "posthoc_wilcoxon_holm_divisions.csv", index=False)
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
