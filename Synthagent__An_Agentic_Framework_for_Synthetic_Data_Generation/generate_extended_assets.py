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
    ("full", "Synthagent (GPT-4o mini)"),
    ("backend_gpt_4_1", "Synthagent (GPT-4.1)"),
    ("backend_claude_sonnet_4_6", "Synthagent (Claude Sonnet 4.6)"),
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
    """F1 change per dataset as four radar panels, one per group of generators, each against
    Synthagent with Claude Sonnet 4.6. The radius is symmetric-logarithmic: linear within
    +-1 pp and logarithmic beyond, so the small differences near zero are spread out."""
    import matplotlib.pyplot as plt
    gpa.print_style()
    data = pd.DataFrame(per_dataset)
    ref = "Synthagent (Claude Sonnet 4.6)"
    panels = [
        ("LLM-based", ["Direct prompting", "Structured outputs", "CLLM"]),
        ("Deep", ["CTGAN", "TVAE", "TabDDPM"]),
        ("Oversampling", ["SMOTE", "ADASYN", "Borderline-SMOTE"]),
        ("Statistical and tree-based", ["Gaussian copula", "Bayesian network", "CART", "ARF"]),
    ]
    colors = ["#e69f00", "#56b4e9", "#009e73", "#cc79a7"]
    markers = ["s", "^", "D", "v"]
    short = {"Heart disease": "Heart dis.", "Breast cancer": "Breast", "Heart failure": "Heart fail.",
             "Diabetes": "Diabetes", "Liver disease": "Liver", "Parkinson's": "Parkinson's",
             "Breast cancer (Coimbra)": "Breast (Coimbra)", "Hepatitis": "Hepatitis",
             "Kidney disease": "Kidney", "Thoracic surgery": "Thoracic"}
    labels = list(data.index)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    lo, hi = -12, 22

    def r(x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.log10(1 + np.abs(x)) + np.log10(1 - lo)

    fig, axes = plt.subplots(1, 4, figsize=(gpa.TEXT_W, 2.6), subplot_kw={"polar": True})
    for ax, (title, members) in zip(axes, panels):
        v = data[ref].tolist()
        ax.plot(angles, r(v + v[:1]), color="#1f3b73", lw=1.6, marker="o", ms=1.8, label="Synthagent")
        for name, c, m in zip(members, colors, markers):
            v = data[name].tolist()
            ax.plot(angles, r(v + v[:1]), color=c, lw=0.9, marker=m, ms=1.6, label=name)
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, r(0)), color="0.2", lw=0.6, ls=(0, (2, 1.5)))
        ax.set_ylim(0, r(hi))
        ax.set_xticks(angles[:-1], [""] * len(labels))
        for ang, lab in zip(angles[:-1], labels):
            c_, s_ = np.cos(ang), np.sin(ang)
            ax.text(ang, r(hi) * 1.05, short.get(lab, lab), fontsize=4.6,
                    ha="left" if c_ > 0.1 else "right" if c_ < -0.1 else "center",
                    va="bottom" if s_ > 0.1 else "top" if s_ < -0.1 else "center")
        ticks = [-10, 0, 10]
        ax.set_yticks(r(ticks), [""] * len(ticks))
        ax.set_rlabel_position(18)
        for t in ax.get_yticklabels():
            t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.8))
        ax.grid(lw=0.4)
        ax.set_title(title, fontsize=6.5, pad=13)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False, fontsize=4.6,
                  handlelength=1.6, labelspacing=0.25)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.86, bottom=0.27, wspace=0.9)
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


def delivery_figure():
    """RQ1: records delivered as a percentage of the configured number, per dataset."""
    import matplotlib.pyplot as plt
    gpa.print_style()
    arms = [("raw_prompting", "Direct prompting", "#e69f00", "o"),
            ("structured_output", "Structured outputs", "#cc79a7", "s"),
            ("cllm", "CLLM", "#56b4e9", "^")]
    syn = ["full", "backend_gpt_4_1", "backend_claude_sonnet_4_6"]
    s = pd.read_csv(gpa.RUN / "ablation_summary.csv")
    s = s[s.dataset.isin(ALL10)]
    s["pct"] = 100 * s["gen_stats_delivered_rows_mean"] / s["gen_stats_requested_rows_mean"]
    order = list(ALL10)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(gpa.COL_W, 1.9))
    ax.axhline(100, color="#1f3b73", lw=1.6, zorder=1)
    logged = s[s.arm.isin(syn)].groupby("dataset")["pct"].agg(["min", "max", "count"]).reindex(order)
    ax.plot(x[logged["count"].notna().values], logged["min"].dropna(), "o", color="#1f3b73", ms=3.2, zorder=3,
            label="Synthagent (logged runs)")
    for k, (arm, label, color, m) in enumerate(arms):
        g = s[s.arm == arm].set_index("dataset")["pct"].reindex(order)
        ax.plot(x + (k - 1) * 0.18, g.values, m, color=color, ms=3, mfc="white" if k == 2 else color,
                mew=0.9, label=label, zorder=2)
    ax.set_xticks(x, [ALL10[d].replace(" (Coimbra)", " (C.)") for d in order], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("Records delivered (%)", fontsize=6.5)
    ax.set_ylim(55, 108)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.legend(fontsize=5.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(gpa.OUT / "delivery.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(gpa.OUT / "delivery.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ablation_figure():
    """RQ1: control-layer variants and sensitivity study as differences from the reference,
    against the noise floor: the reference +- the spread of five runs of the same pipeline."""
    import matplotlib.pyplot as plt
    gpa.print_style()
    a = pd.read_csv(gpa.OUT / "ablation_summary.csv").set_index("Configuration")
    z = pd.read_csv(gpa.OUT / "sensitivity.csv").set_index("arm")
    a = a.rename(columns={"Corr.": "Corr", "$\\Delta$F1 (pp)": "dF1"})
    cols = ["JSD", "Corr", "dF1", "Tokens"]
    noise = [("Synthagent", "Reference"), ("No inferred type rules", "No type rules"),
             ("No post-processing", "No post-processing"), ("No context join", "No context join"),
             ("Logged rerun", "Logged rerun")]
    design = [("Dense context", "Dense context"), ("Diverse context", "Diverse context"),
              ("LLM-selected hyperparameters", "LLM-chosen settings"), ("Range enforcement", "Range enforcement")]
    sens = [("ctx_10", "Context $\\rho=10\\%$"), ("ctx_40", "Context $\\rho=40\\%$"), ("batch_15", "Batch $b=15$"),
            ("batch_60", "Batch $b=60$"), ("temp_0", "Temperature 0"), ("temp_10", "Temperature 1.0")]
    rows = []
    for key, lab in noise + design:
        r = a.loc[key]
        rows.append((lab, r["JSD"], r["Corr"], r["dF1"], r["Tokens"] / 1000))
    for key, lab in sens:
        r = z.loc[key]
        rows.append((lab, r["JSD"], r["Corr"], r["dF1"], r["tokens"] / 1000))
    df = pd.DataFrame(rows, columns=["label"] + cols)
    groups = [0] * len(noise) + [1] * len(design) + [2] * len(sens)
    y = []
    pos = 0
    for i, g in enumerate(groups):
        if i and g != groups[i - 1]:
            pos += 0.7
        y.append(pos)
        pos += 1
    y = -np.array(y)
    band = df.iloc[:len(noise)][cols]
    titles = {"JSD": "Fidelity (JSD)", "Corr": "Correlation fidelity", "dF1": "F1 change (pp)", "Tokens": "Tokens (thousands)"}
    fig, axes = plt.subplots(1, 4, figsize=(gpa.TEXT_W, 2.2), sharey=True)
    for ax, c in zip(axes, cols):
        # noise floor: the reference value +- the spread of five runs of the same pipeline
        spread = band[c].max() - band[c].min()
        lo, hi = df[c].iloc[0] - spread, df[c].iloc[0] + spread
        ax.axvspan(lo, hi, color="0.88", lw=0, zorder=0)
        ax.axvline(df[c].iloc[0], color="#1f3b73", lw=0.8, zorder=1)
        out = (df[c] < lo) | (df[c] > hi)
        ax.scatter(df[c][~out], y[~out.values], s=10, color="0.55", zorder=2)
        ax.scatter(df[c][out], y[out.values], s=14, color="#d55e00", zorder=3)
        ax.set_title(titles[c], fontsize=6.5)
        ax.tick_params(labelsize=5.5)
        ax.grid(axis="x", lw=0.3, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y, df["label"], fontsize=5.8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(gpa.OUT / "ablation_effects.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(gpa.OUT / "ablation_effects.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tradeoff_figure():
    """RQ4: fidelity against median F1 change for every generator; Synthagent is the
    Claude Sonnet 4.6 configuration, the other two Synthagent LLMs are left out."""
    import matplotlib.pyplot as plt
    gpa.print_style()
    t = pd.read_csv(gpa.OUT / "generators10.csv").set_index("arm")
    fam = {"backend_claude_sonnet_4_6": ("Synthagent", "#1f3b73"),
           "raw_prompting": ("LLM-based", "#e69f00"), "structured_output": ("LLM-based", "#e69f00"), "cllm": ("LLM-based", "#e69f00"),
           "smote": ("Oversampling", "#cc79a7"), "adasyn": ("Oversampling", "#cc79a7"), "borderline_smote": ("Oversampling", "#cc79a7"),
           "gaussian_copula": ("Statistical and tree-based", "#009e73"), "bayesian_network": ("Statistical and tree-based", "#009e73"),
           "cart": ("Statistical and tree-based", "#009e73"), "arf": ("Statistical and tree-based", "#009e73"),
           "ctgan": ("Deep", "#56b4e9"), "tvae": ("Deep", "#56b4e9"), "tabddpm": ("Deep", "#56b4e9")}
    names = {"backend_claude_sonnet_4_6": "Synthagent", "raw_prompting": "Direct", "structured_output": "Structured",
             "cllm": "CLLM", "smote": "SMOTE", "adasyn": "ADASYN", "borderline_smote": "Borderline",
             "gaussian_copula": "Copula", "bayesian_network": "Bayes. net.", "cart": "CART", "arf": "ARF",
             "ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM"}
    off = {"backend_claude_sonnet_4_6": (0.6, 0.05, "left"), "raw_prompting": (0.5, 0.06, "left"),
           "structured_output": (0.5, -0.13, "left"), "cllm": (0.5, 0.05, "left"), "smote": (0.3, 0.17, "left"),
           "adasyn": (-0.4, -0.13, "right"), "borderline_smote": (0.5, -0.15, "left"), "gaussian_copula": (-0.5, 0.06, "right"),
           "bayesian_network": (-0.5, -0.12, "right"), "cart": (-0.5, 0.06, "right"), "arf": (0.5, -0.17, "left"),
           "ctgan": (0.5, 0.0, "left"), "tvae": (0.5, 0.05, "left"), "tabddpm": (0.0, 0.17, "center")}
    fig, ax = plt.subplots(figsize=(gpa.COL_W, 2.3))
    ax.axhline(0, color="0.5", lw=0.6, ls=(0, (2, 2)), zorder=0)
    seen = set()
    for arm, (f, c) in fam.items():
        x, y = t.loc[arm, "JSD"], t.loc[arm, "dF1_median"]
        star = arm == "backend_claude_sonnet_4_6"
        ax.scatter(x, y, s=70 if star else 22, marker="*" if star else "o", color=c, zorder=3,
                   edgecolor="white", linewidth=0.4, label=None if f in seen else f)
        seen.add(f)
        dx, dy, ha = off[arm]
        ax.text(x + dx, y + dy, names[arm], fontsize=5.6, ha=ha, va="center",
                fontweight="bold" if star else "normal", color="#1f3b73" if star else "0.2")
    ax.set_xlabel("Fidelity (JSD, 0-100)", fontsize=6.5)
    ax.set_ylabel("Median F1 change (pp)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_xlim(57, 90); ax.set_ylim(-2.5, 1.8)
    ax.grid(lw=0.3, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=5.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    fig.savefig(gpa.OUT / "tradeoff.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(gpa.OUT / "tradeoff.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    print(dataset_table().round(1).to_string(index=False))
    t, f, c = generator_comparison()
    radar(pd.read_csv(gpa.OUT / "generators10_f1.csv", index_col=0))
    delivery_figure()
    tradeoff_figure()
    ablation_figure()
    print(t.round(2).to_string(index=False)); print(f.to_string(index=False))
    print(c[c.p_holm < 0.05].to_string(index=False))
    s, sf = sensitivity()
    print(s.round(3).to_string(index=False)); print(sf.to_string(index=False))


if __name__ == "__main__":
    main()
