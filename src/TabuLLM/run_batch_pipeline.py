"""
run_batch_pipeline.py
─────────────────────
Batch runner: runs the full TabuLLM augmentation pipeline over every dataset
in the configured DATA_DIR, then saves per-dataset JSON metrics + PDF reports
and a consolidated summary CSV containing ALL fidelity and ML metrics.

This single script has three modes:

  batch          (default) — one full-pipeline run (every module enabled) across
                              every dataset in DATASET_REGISTRY.
  ablation       (--ablation or TABULLM_MODE=ablation) — runs the SAME full batch
                              pipeline once per arm in ABLATION_ARMS + the special
                              arms (module toggles, context-strategy variants, the
                              SMOTE baseline, the Ponder-hyperparameters arm, and
                              any cross-model-backend arms), so "ablation" is
                              literally "run the batch pipeline N times with
                              different module combinations." Every arm gets full
                              JSON/PDF/per-fold-CSV output (arm subfolders under
                              the run dir), plus a combined ablation_summary.csv
                              and an ablation_deltas.csv with paired-significance
                              p-values and BH-corrected q-values against "full".
  stochasticity  (--stochasticity) — isolates pure LLM generation stochasticity
                              from data-split/context-sampling variance: fixes ONE
                              split + context sample per dataset and repeats
                              generation `TABULLM_STOCHASTICITY_REPEATS` times.

Usage
─────
    cd src/TabuLLM
    python run_batch_pipeline.py                 # normal batch run
    python run_batch_pipeline.py --ablation       # ablation sweep
    python run_batch_pipeline.py --stochasticity  # generation-noise decomposition

    # Gemini backend:
    set TABULLM_PROVIDER=gemini
    set GOOGLE_API_KEY=<your-key>
    python run_batch_pipeline.py

    # Override model:
    set TABULLM_MODEL=meta-llama/llama-3-8b
    python run_batch_pipeline.py

    # Ablation: restrict datasets, or explicitly reduce fold count (accepting
    # weaker statistical power — by default ablation folds == TABULLM_FOLDS,
    # same rigor as a standalone batch run):
    set TABULLM_ABLATION_DATASETS=heart_disease.csv,parkinsons.csv
    set TABULLM_ABLATION_FOLDS=5
    python run_batch_pipeline.py --ablation

    # Stochasticity: repeats and dataset subset (default: first 2 datasets, 5 repeats):
    set TABULLM_STOCHASTICITY_REPEATS=5
    set TABULLM_STOCHASTICITY_DATASETS=heart_disease.csv,parkinsons.csv
    python run_batch_pipeline.py --stochasticity

Resuming an interrupted run
────────────────────────────
Every mode checkpoints progress to {run_dir}/checkpoint.json the moment each unit (a
dataset in batch/stochasticity mode, an arm+dataset pair in ablation mode) finishes --
so a crash, a kill, an LLM timeout, or Ctrl-C partway through a long run only ever loses
the ONE unit that was in flight, never the units already completed:

    python run_batch_pipeline.py --ablation --resume          # resume the latest run
    python run_batch_pipeline.py --ablation --resume "results/ablation_20260101_120000"

Resuming re-checks the fold/repeat count against what produced each cached unit --
pipeline_manager seeds every fold's train/test split purely from `42 + fold_num`, so a
different fold count means a different, non-comparable set of splits. Any unit whose
count doesn't match the current invocation is recomputed rather than reused.

Prerequisites
─────────────
- For 'local' provider: LM Studio (or any OpenAI-compatible server) running at
  http://127.0.0.1:1234  (override with LOCAL_LLM_URL env var).
- For 'gemini' provider: set GOOGLE_API_KEY.
"""

import os
import sys
import json
import argparse
import datetime
import traceback

import numpy as np
import pandas as pd

# Make sure the package root is importable regardless of where you call this from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabullm import TabuLLM  # noqa: E402  (after sys.path tweak)
from services.pipeline_manager import (  # noqa: E402
    _paired_pvalue, apply_fdr_correction, run_smote_dataset_benchmark, run_stochasticity_check,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET REGISTRY
#   key   : filename relative to DATA_DIR
#   target: column the ML models should predict
#   ml    : subset of ['rf','lr','xgb','nb','dt','gb'] to benchmark
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR    = os.path.join(_HERE, "data", "medical")
RESULTS_DIR = os.path.join(_HERE, "results")

_ALL_MODELS = ["rf", "lr", "xgb", "nb", "dt", "gb", "ab", "knn", "mlp", "svm"]

DATASET_REGISTRY: dict[str, dict] = {
    "heart_disease.csv": {
        "target": "num",
        "ml": _ALL_MODELS,
    },
    "breast_cancer_wisconsin_diagnostic.csv": {
        "target": "Diagnosis",
        "ml": _ALL_MODELS,
    },
    "heart_failure_clinical_records.csv": {
        "target": "death_event",
        "ml": _ALL_MODELS,
    },
    "indian_diabetes.csv": {
        "target": "Outcome",
        "ml": _ALL_MODELS,
    },
    "indian_liver_patient_dataset.csv": {
        "target": "Selector",
        "ml": _ALL_MODELS,
    },
    "parkinsons.csv": {
        "target": "status",
        "ml": _ALL_MODELS,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SETTINGS  (override any via env vars)
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE: dict = {
    "provider":     os.getenv("TABULLM_PROVIDER",   "local"),
    "model":        os.getenv("TABULLM_MODEL",       "google/gemma-4-e2b"),
    "endpoint_url": os.getenv("LOCAL_LLM_URL",       "http://127.0.0.1:1234/v1/chat/completions"),
    "folds":        int(os.getenv("TABULLM_FOLDS",   "10")),
    "strategy":     os.getenv("TABULLM_STRATEGY",    "random"),   # random | dense | diverse
    "na_strategy":  os.getenv("TABULLM_NA",          "impute"),   # impute | drop
    "ctx_size_pct": int(os.getenv("TABULLM_CTX_PCT", "20")),      # % of train split → LLM context
    "gen_size_pct": int(os.getenv("TABULLM_GEN_PCT", "100")),     # % of train split size to generate
}

# NOTE: the batch pipeline intentionally never calls TabuLLM.ponder_strategy() — that
# AI-authored augmentation plan is an interactive-UI-only feature (see callbacks/audit.py).
# The batch harness instead uses a fixed, code-derived generation plan (inferred rules +
# a one-time dataset audit — see pipeline_manager.run_single_dataset_benchmark) so every
# fold of a given dataset is generated under identical conditions.

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION ARMS
#   Each arm is a set of TabuLLM.augment() kwarg overrides layered on top of the
#   PIPELINE defaults above. "full" (no overrides) is the baseline every other
#   arm is compared against. Running the ablation sweep is exactly "run the
#   batch pipeline once per arm" — see run_ablation().
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_ARMS: dict[str, dict] = {
    "full":               {},
    "no_rule_injection":  {"inject_inferred_rules": False},
    "no_post_processing": {"apply_post_processing": False},
    "no_context_join":    {"join_context": False},
    "strategy_dense":     {"strategy": "dense"},
    "strategy_diverse":   {"strategy": "diverse"},
}

# Special arms that don't fit the simple "config override on the same LLM" mold and get
# dispatched separately in run_ablation() (see ALL_ARMS / the arm-dispatch loop below):
#   "smote"           -- no LLM at all, generates via SMOTE/SMOTENC instead.
#   "ponder_strategy" -- hyperparameters are chosen by the LLM per dataset (structured
#                        Ponder call) instead of the fixed PIPELINE defaults.
#   entries of MODEL_SWEEP_ARMS -- same pipeline, a different LLM backend.
SPECIAL_ARMS = ["smote", "ponder_strategy"]

# Cross-model sweep: arm-name -> model id. Populated from whatever else is loaded in the
# local LM Studio instance (checked via GET /v1/models) alongside PIPELINE["model"].
MODEL_SWEEP_ARMS: dict[str, str] = {
    "backend_qwen2_5": "qwen2.5-vl-7b-instruct",
}

ALL_ARMS: list[str] = list(ABLATION_ARMS) + SPECIAL_ARMS + list(MODEL_SWEEP_ARMS)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC KEYS
#   All keys produced by calculate_metrics() — used to average across folds
#   and populate the summary CSV.
# ─────────────────────────────────────────────────────────────────────────────

FIDELITY_METRICS = [
    "Synthetic Fidelity (JSD)",   # numerical distribution similarity (0-100)
    "Correlation Stability",       # multivariate correlation preservation (0-100)
    "Distribution Consistency",    # median JSD fidelity (0-100)
    "Entropy Preservation",        # Shannon entropy kept (0-100)
    "Mean Z Score Drift",          # standardised mean shift (lower = better)
    "Categorical Fidelity",        # Hellinger fidelity for categorical cols (0-100)
    "Privacy Risk",                # % synthetic rows too close to real (lower = better)
    "Data Density",                # completeness / non-null % (0-100)
    "Coverage Score",              # % synthetic rows far enough from real (0-100)
    "Feature Accuracy",            # inverse mean relative error (0-100)
    "Novelty Score",               # nearest-neighbour distance ratio
]

# Same metrics computed against the pure generated rows only (no real context blended
# in) — the clean, context-blending-free signal of generation quality.
GEN_FIDELITY_METRICS = [f"Gen {key}" for key in FIDELITY_METRICS]

# ML metrics emitted per model
ML_METRICS = ["accuracy", "f1", "rmse", "r2"]

# Generation cost/reliability fields captured per fold (see pipeline_manager's
# `generation_stats`) — tracked to give the ablation study a cost-vs-quality signal.
GENERATION_STATS_FIELDS = [
    "prompt_tokens", "candidates_tokens", "total_tokens",
    "rule_violations", "repair_attempts", "generation_seconds",
]

# Non-metric identity columns present in every summary row — excluded when computing
# ablation deltas (there's no meaningful "delta" for a dataset name or fold count).
_IDENTITY_COLS = {"dataset", "target", "rows", "cols", "folds", "model", "arm"}

# Per-class fidelity roles: minority/majority by prevalence in the original data (see
# utils.metrics.calculate_metrics_by_class) — isolates whether synthetic fidelity holds
# up on the minority class specifically, not just the pooled/aggregate signal.
CLASS_FIDELITY_ROLES = ("minority", "majority")

# Privacy/discriminability attacks against the generated-only corpus (see
# utils.metrics.compute_c2st_accuracy / compute_mia_attack_auc). ~0.5 on both = no
# distinguishability / no membership-leakage signal.
PRIVACY_ATTACK_METRICS = ["Gen C2ST Accuracy", "Gen MIA Attack AUC"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_separator(char: str = "─", width: int = 64) -> None:
    print(char * width)


def _save_json(results: dict, path: str) -> None:
    """Serialise results to JSON, skipping bulky row-level payloads."""
    _SKIP = {"synthetic_data", "augmented_data", "context_data",
             "test_data", "train_non_ctx_data"}

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in _SKIP}
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        if hasattr(obj, "item"):    # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):  # numpy array / Series
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(results), fh, indent=2, default=str)


def _flat_fidelity_row(fold_results: list[dict]) -> dict:
    """
    Average every fidelity / structural metric (blended + generated-only) across folds.
    Returns one column per metric: mean and ±std.
    E.g. 'Synthetic Fidelity (JSD)_mean', 'Synthetic Fidelity (JSD)_std'
    """
    out: dict = {}
    for key in FIDELITY_METRICS + GEN_FIDELITY_METRICS + PRIVACY_ATTACK_METRICS:
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"{key}_mean"] = round(float(np.mean(vals)), 3)
            out[f"{key}_std"]  = round(float(np.std(vals)),  3)
    return out


def _flat_generation_stats_row(fold_results: list[dict]) -> dict:
    """Average generation cost/reliability stats, and total per-fold wall-clock time, across folds."""
    out: dict = {}
    for key in GENERATION_STATS_FIELDS:
        vals = [r["generation_stats"][key] for r in fold_results
                if isinstance(r.get("generation_stats"), dict) and isinstance(r["generation_stats"].get(key), (int, float))]
        if vals:
            out[f"gen_stats_{key}_mean"] = round(float(np.mean(vals)), 3)

    fold_seconds_vals = [r["fold_seconds"] for r in fold_results if isinstance(r.get("fold_seconds"), (int, float))]
    if fold_seconds_vals:
        out["fold_seconds_mean"] = round(float(np.mean(fold_seconds_vals)), 3)
        out["fold_seconds_std"]  = round(float(np.std(fold_seconds_vals)),  3)
    return out


def _flat_ml_row(overall_ml: dict) -> dict:
    """
    Flatten ALL model × metric combinations into a flat dict for the CSV.
    Columns follow the pattern:  <model>_<metric>_real/gen/aug/delta/delta_gen
    plus paired-significance p-values. Only metrics actually computed are included.
    """
    out: dict = {}
    for model, metrics in overall_ml.items():
        if not isinstance(metrics, dict):
            continue
        for metric in ML_METRICS:
            if metric not in metrics or not isinstance(metrics[metric], dict):
                continue
            m = metrics[metric]
            out[f"{model}_{metric}_real"]      = round(float(m.get("real",  0) or 0), 4)
            out[f"{model}_{metric}_gen"]       = round(float(m.get("gen",   0) or 0), 4)
            out[f"{model}_{metric}_aug"]       = round(float(m.get("aug",   0) or 0), 4)
            out[f"{model}_{metric}_delta"]     = round(float(m.get("delta", 0) or 0), 4)
            out[f"{model}_{metric}_delta_gen"] = round(float(m.get("delta_gen", 0) or 0), 4)
            # Also capture the ±std values so the CSV is fully self-contained
            if m.get("real_sd") is not None:
                out[f"{model}_{metric}_real_sd"]      = round(float(m.get("real_sd",  0) or 0), 4)
                out[f"{model}_{metric}_gen_sd"]       = round(float(m.get("gen_sd",   0) or 0), 4)
                out[f"{model}_{metric}_aug_sd"]       = round(float(m.get("aug_sd",   0) or 0), 4)
                out[f"{model}_{metric}_delta_sd"]     = round(float(m.get("delta_sd", 0) or 0), 4)
                out[f"{model}_{metric}_delta_gen_sd"] = round(float(m.get("delta_gen_sd", 0) or 0), 4)
            # Paired significance (Wilcoxon signed-rank across folds); None if not computable
            p_aug = m.get("pvalue_aug_vs_real")
            p_gen = m.get("pvalue_gen_vs_real")
            out[f"{model}_{metric}_pvalue_aug"] = round(float(p_aug), 4) if p_aug is not None else None
            out[f"{model}_{metric}_pvalue_gen"] = round(float(p_gen), 4) if p_gen is not None else None
            # Benjamini-Hochberg-corrected q-values (see pipeline_manager.apply_fdr_correction),
            # corrected jointly across every model x metric x comparison for this dataset --
            # the honest significance call once many tests are run together, not pvalue alone.
            q_aug = m.get("qvalue_aug_vs_real")
            q_gen = m.get("qvalue_gen_vs_real")
            out[f"{model}_{metric}_qvalue_aug"] = round(float(q_aug), 4) if q_aug is not None else None
            out[f"{model}_{metric}_qvalue_gen"] = round(float(q_gen), 4) if q_gen is not None else None
    return out


def _per_class_fidelity_columns(fold_result: dict) -> dict:
    """
    Flattens fold_result['fidelity_by_class'] (generated-only fidelity, computed
    separately for the minority and majority class) into 'Gen <metric> (role)' columns.
    """
    out: dict = {}
    fbc = fold_result.get("fidelity_by_class") or {}
    for role in CLASS_FIDELITY_ROLES:
        role_metrics = fbc.get(role)
        if not isinstance(role_metrics, dict):
            continue
        for key in FIDELITY_METRICS:
            if isinstance(role_metrics.get(key), (int, float)):
                out[f"Gen {key} ({role})"] = role_metrics[key]
    return out


def _flat_class_fidelity_row(fold_results: list[dict]) -> dict:
    """Mean ± std across folds of the per-class (minority/majority) generated-only fidelity."""
    out: dict = {}
    for role in CLASS_FIDELITY_ROLES:
        for key in FIDELITY_METRICS:
            vals = []
            for r in fold_results:
                role_metrics = (r.get("fidelity_by_class") or {}).get(role)
                if isinstance(role_metrics, dict) and isinstance(role_metrics.get(key), (int, float)):
                    vals.append(role_metrics[key])
            if vals:
                col = f"Gen {key} ({role})"
                out[f"{col}_mean"] = round(float(np.mean(vals)), 3)
                out[f"{col}_std"]  = round(float(np.std(vals)),  3)
    return out


def _per_fold_rows(fold_results: list[dict]) -> list[dict]:
    """
    One row per fold: fold-level fidelity (blended + generated-only), generation cost/
    reliability stats, and raw per-model×metric×arm ML values. Satisfies "results should
    also be outputted per fold" without waiting on the folds-aggregated summary.
    """
    rows: list[dict] = []
    for i, r in enumerate(fold_results, 1):
        row: dict = {"fold": i}
        if isinstance(r.get("fold_seconds"), (int, float)):
            row["fold_seconds"] = r["fold_seconds"]
        for key in FIDELITY_METRICS + GEN_FIDELITY_METRICS + PRIVACY_ATTACK_METRICS:
            if isinstance(r.get(key), (int, float)):
                row[key] = r[key]
        row.update(_per_class_fidelity_columns(r))
        gen_stats = r.get("generation_stats", {}) or {}
        for key in GENERATION_STATS_FIELDS:
            if isinstance(gen_stats.get(key), (int, float)):
                row[f"gen_stats_{key}"] = gen_stats[key]
        for model, ml in r.get("ml_eval", {}).items():
            if not isinstance(ml, dict):
                continue
            for arm in ("original", "generated", "augmented"):
                arm_vals = ml.get(arm) or {}
                if not isinstance(arm_vals, dict):
                    continue
                for metric in ML_METRICS:
                    if metric in arm_vals:
                        row[f"{model}_{metric}_{arm}"] = arm_vals[metric]
        rows.append(row)
    return rows


def _per_repeat_rows(repeat_results: list[dict]) -> list[dict]:
    """Same flattening as _per_fold_rows, reused for the stochasticity check's
    repeat_results (one row per repeated generation at a fixed split, instead of per fold)."""
    rows = _per_fold_rows(repeat_results)
    for row in rows:
        row["repeat"] = row.pop("fold")
    return rows


def _per_fold_series(fold_results: list[dict], ml_models: list[str]) -> dict:
    """
    Raw per-fold values (NOT averaged) for the generated-only fidelity metrics and the
    ML deltas, index-aligned by fold number. Folds share the same train/test split across
    every ablation arm of a dataset (pipeline_manager's random_state = 42 + fold_num
    depends only on the fold number), so index i in one arm's series lines up with index i
    in another arm's series — that pairing is what makes cross-arm significance testing
    (see run_ablation()) valid, rather than just comparing means.
    """
    series: dict = {}
    for key in GEN_FIDELITY_METRICS:
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if len(vals) == len(fold_results):
            series[key] = vals

    for model in ml_models:
        for metric in ML_METRICS:
            deltas, deltas_gen = [], []
            for r in fold_results:
                me = (r.get("ml_eval") or {}).get(model)
                if not me:
                    continue
                o = (me.get("original") or {}).get(metric)
                g = (me.get("generated") or {}).get(metric)
                a = (me.get("augmented") or {}).get(metric)
                if o is not None and a is not None:
                    deltas.append(a - o)
                if o is not None and g is not None:
                    deltas_gen.append(g - o)
            if len(deltas) == len(fold_results):
                series[f"{model}_{metric}_delta"] = deltas
            if len(deltas_gen) == len(fold_results):
                series[f"{model}_{metric}_delta_gen"] = deltas_gen
    return series


def _ml_summary_str(overall_ml: dict) -> str:
    """One-liner human-readable ML summary for console output."""
    parts = []
    for model, metrics in overall_ml.items():
        for metric in ("accuracy", "f1", "r2"):
            if metric in metrics and isinstance(metrics[metric], dict):
                delta     = metrics[metric].get("delta", 0) or 0
                delta_gen = metrics[metric].get("delta_gen", 0) or 0
                parts.append(f"{model.upper()} {metric} aug={delta:+.3f} gen={delta_gen:+.3f}")
                break
    return " | ".join(parts) if parts else "(no ML results)"


def _write_summary_csv(summary_rows: list[dict], out_dir: str, csv_name: str = "batch_summary.csv") -> "pd.DataFrame | None":
    """Adds the cross-dataset rollup row, writes the CSV, prints a condensed console table."""
    if not summary_rows:
        return None
    summary_df = pd.DataFrame(summary_rows)

    # Cross-dataset rollup: mean of every numeric metric across all successful datasets,
    # giving a single "how did the framework do overall" row.
    rollup = summary_df.select_dtypes(include=[np.number]).mean(numeric_only=True)
    rollup_row = {c: (round(float(rollup[c]), 4) if c in rollup else None) for c in summary_df.columns}
    rollup_row["dataset"] = "__ROLLUP__"
    rollup_row["target"]  = f"{len(summary_rows)} datasets"
    rollup_row["model"]   = summary_rows[0].get("model", PIPELINE["model"])
    summary_df = pd.concat([summary_df, pd.DataFrame([rollup_row])], ignore_index=True)

    csv_path = os.path.join(out_dir, csv_name)
    summary_df.to_csv(csv_path, index=False)
    print(f"  Summary CSV → {csv_path}")

    _KEY_COLS = ["dataset", "target", "rows",
                 "Synthetic Fidelity (JSD)_mean",
                 "Correlation Stability_mean",
                 "Privacy Risk_mean",
                 "Coverage Score_mean"]
    display_cols = [c for c in _KEY_COLS if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT / RESUME
#   batch/ablation/stochasticity all persist one JSON record per completed unit
#   (a dataset in batch/stochasticity mode, an arm+dataset pair in ablation mode) to
#   {run_dir}/checkpoint.json the moment that unit finishes -- so a crash, kill, or
#   Ctrl-C partway through a long run only ever loses the ONE unit that was in flight,
#   never the units already done. --resume replays a run dir's checkpoint and skips
#   any unit already recorded there, recomputing only what's missing/failed.
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """Recursively converts numpy scalars/arrays to native Python types. Needed because
    checkpointed per-fold series are later re-used numerically (Wilcoxon deltas), so a
    str()-fallback encoder (fine for the human-readable *_metrics.json) would silently
    break that math on resume."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(i) for i in obj]
    if hasattr(obj, "item"):    # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array / Series
        return obj.tolist()
    return obj


def _checkpoint_path(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoint.json")


def _load_checkpoint(run_dir: str) -> dict:
    path = _checkpoint_path(run_dir)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("completed"), dict):
                return data
        except Exception as e:
            print(f"  [WARN] Checkpoint at {path} is unreadable ({e}); treating this run dir as fresh.")
    return {"completed": {}}


def _save_checkpoint(run_dir: str, checkpoint: dict) -> None:
    """Atomic write (temp file + os.replace) -- a crash mid-save can never corrupt the
    checkpoint already on disk from the previous successfully-completed unit."""
    path = _checkpoint_path(run_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, indent=2)
    os.replace(tmp_path, path)


def _mark_unit_done(
    run_dir: str, checkpoint: dict, key: str,
    summary_row: dict, per_fold_series: "dict | None", unit_config: dict,
) -> None:
    """
    `unit_config` records the seed-determining parameters this unit was run with (e.g.
    {"folds": 10} -- pipeline_manager derives every fold's train/test split purely from
    `random_state = 42 + fold_num`, so the fold COUNT is what fixes the actual set of
    seeds used). Stored alongside the result so a later resume can tell whether cached
    data was produced under the same seed basis as the run asking for it -- see
    _checkpoint_lookup().
    """
    checkpoint["completed"][key] = _json_safe({
        "unit_config": unit_config,
        "summary_row": summary_row,
        "per_fold_series": per_fold_series or {},
    })
    _save_checkpoint(run_dir, checkpoint)


def _checkpoint_lookup(checkpoint: dict, key: str, expected_unit_config: dict) -> "dict | None":
    """
    Returns the checkpointed entry for `key` ONLY if it was produced under the same
    seed-determining configuration this invocation is using (see _mark_unit_done). A
    mismatch -- e.g. resuming with a different --ablation fold count than the original
    run used -- means the cached folds aren't the same seeded splits the rest of this run
    will use, so the unit is treated as not-yet-done and gets recomputed rather than
    silently mixing incompatible data into the same comparison.
    """
    entry = checkpoint["completed"].get(key)
    if entry is None:
        return None
    if entry.get("unit_config") != expected_unit_config:
        print(f"  [WARN] Checkpoint entry {key!r} was computed with {entry.get('unit_config')}, "
              f"not the {expected_unit_config} this run is using -- recomputing it "
              f"(different config means different/incomparable fold seeds).")
        return None
    return entry


def _find_resumable_run_dir(prefix: str) -> "str | None":
    """Most recent {prefix}_* directory under RESULTS_DIR that has a checkpoint.json --
    i.e. a previous run of this mode we can pick back up. Doesn't distinguish finished
    from unfinished runs: resuming an already-finished run just finds nothing left to do
    and immediately re-derives the final summary CSVs from the checkpoint."""
    if not os.path.isdir(RESULTS_DIR):
        return None
    candidates = sorted(
        (d for d in os.listdir(RESULTS_DIR) if d.startswith(f"{prefix}_")),
        reverse=True,
    )
    for d in candidates:
        run_dir = os.path.join(RESULTS_DIR, d)
        if os.path.exists(_checkpoint_path(run_dir)):
            return run_dir
    return None


def _resolve_run_dir(resume: "str | None", prefix: str, timestamp: str) -> "tuple[str, bool]":
    """Returns (run_dir, resumed). resume=None (default) always starts a fresh timestamped
    dir -- unchanged behaviour for anyone not opting in. resume="__latest__" (bare
    --resume on the CLI) picks the most recent resumable dir for this mode. Any other
    string is treated as an explicit run-dir path to resume. Falls back to starting fresh
    (with a warning) if the requested run dir doesn't exist."""
    if not resume:
        run_dir = os.path.join(RESULTS_DIR, f"{prefix}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, False

    run_dir = _find_resumable_run_dir(prefix) if resume == "__latest__" else resume
    if run_dir and os.path.isdir(run_dir):
        return run_dir, True

    where = "(none found under results/)" if resume == "__latest__" else repr(resume)
    print(f"  [WARN] No resumable {prefix} run {where}; starting fresh.")
    run_dir = os.path.join(RESULTS_DIR, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, False


def _run_dataset_arm(
    lib: TabuLLM,
    filename: str,
    ds_cfg: dict,
    out_dir: str,
    arm_name: str = "full",
    config_overrides: "dict | None" = None,
    folds: "int | None" = None,
    ponder: bool = False,
) -> "dict | None":
    """
    Runs one (dataset, arm) augmentation end to end: loads the CSV, calls the full
    TabuLLM pipeline, and writes the same JSON/PDF/per-fold-CSV outputs a normal batch
    run would. Used by both run_batch() (single "full" arm) and run_ablation() (looped
    over every arm) — the ablation sweep is exactly N calls to this function with
    different config_overrides, i.e. "running several batch pipelines."

    ponder=True replaces `config_overrides` with a fresh call to
    lib.ponder_hyperparameters(df, ...) for THIS dataset -- the structured, empirically
    testable counterpart to the free-text ponder_strategy(), letting the LLM pick
    ctx_size_pct/strategy/temperature instead of the fixed PIPELINE defaults.

    Returns {"summary_row": ..., "per_fold_series": ...} or None if skipped/failed.
    """
    config_overrides = config_overrides or {}
    csv_path  = os.path.join(DATA_DIR, filename)
    target    = ds_cfg["target"]
    ml_models = ds_cfg.get("ml", ["rf"])
    name      = os.path.splitext(filename)[0]
    folds     = folds if folds is not None else PIPELINE["folds"]

    print()
    _print_separator()
    print(f"  Dataset  : {filename}   [arm={arm_name}]")
    print(f"  Target   : {target}  |  Models : {ml_models}")
    _print_separator()

    if not os.path.exists(csv_path):
        print(f"  [SKIP] File not found: {csv_path}")
        return None

    try:
        df = lib.load_df(csv_path)
        print(f"  Loaded   : {len(df)} rows × {len(df.columns)} cols")

        if target not in df.columns:
            print(f"  [SKIP] Target column '{target}' not in dataset.")
            return None

        kwargs = dict(
            targets=[target],
            folds=folds,
            strategy=PIPELINE["strategy"],
            ml_models=ml_models,
            na_strategy=PIPELINE["na_strategy"],
            ctx_size_pct=PIPELINE["ctx_size_pct"],
            gen_size_pct=PIPELINE["gen_size_pct"],
        )
        if ponder:
            ponder_theta = lib.ponder_hyperparameters(
                df, targets=[target],
                defaults={
                    "ctx_size_pct": PIPELINE["ctx_size_pct"],
                    "strategy": PIPELINE["strategy"],
                    "temperature": 0.7,
                },
            )
            print(f"  Ponder   : {ponder_theta}")
            kwargs.update(ponder_theta)
        else:
            kwargs.update(config_overrides)

        print(f"  Running augmentation ({folds} folds, arm={arm_name}) …")
        results = lib.augment(df, **kwargs)

        if not results:
            print("  [WARN] Pipeline returned empty results.")
            return None

        results["filename"]     = filename
        results["dataset_name"] = name
        fold_results = results.get("results", [])

        json_path = os.path.join(out_dir, f"{name}_metrics.json")
        _save_json(results, json_path)
        print(f"  Saved JSON  → {json_path}")

        per_fold = _per_fold_rows(fold_results)
        if per_fold:
            per_fold_csv = os.path.join(out_dir, f"{name}_per_fold.csv")
            pd.DataFrame(per_fold).to_csv(per_fold_csv, index=False)
            print(f"  Saved Folds → {per_fold_csv}")

        pdf_path = os.path.join(out_dir, f"{name}_report.pdf")
        lib.generate_pdf_report(results, pdf_path, original_df=df)
        print(f"  Saved PDF   → {pdf_path}")

        ml_str  = _ml_summary_str(results.get("overall_ml", {}))
        avg_fid = results.get("avg_fidelity", 0)
        print(f"  ✓  Avg Fidelity (JSD): {avg_fid:.1f}%  |  {ml_str}")

        summary_row: dict = {
            "dataset": filename,
            "target":  target,
            "rows":    len(df),
            "cols":    len(df.columns),
            "folds":   folds,
            "model":   lib.default_model,
            "arm":     arm_name,
        }
        summary_row.update(_flat_fidelity_row(fold_results))
        summary_row.update(_flat_class_fidelity_row(fold_results))
        summary_row.update(_flat_generation_stats_row(fold_results))
        summary_row.update(_flat_ml_row(results.get("overall_ml", {})))

        return {
            "summary_row": summary_row,
            "per_fold_series": _per_fold_series(fold_results, ml_models),
        }

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        traceback.print_exc()
        return None


def _run_smote_arm(lib: TabuLLM, filename: str, ds_cfg: dict, out_dir: str, folds: "int | None" = None) -> "dict | None":
    """
    Runs one (dataset, "smote") augmentation end to end via the SMOTE baseline path
    (services.pipeline_manager.run_smote_dataset_benchmark) instead of the LLM -- the
    same JSON/PDF/per-fold-CSV outputs and return contract as _run_dataset_arm, so it
    slots into the exact same dispatch/aggregation code in run_ablation(). `lib` is only
    used for load_df() and generate_pdf_report() (pure I/O, no LLM calls involved) --
    SMOTE never touches the LLM at all.
    """
    csv_path  = os.path.join(DATA_DIR, filename)
    target    = ds_cfg["target"]
    ml_models = ds_cfg.get("ml", ["rf"])
    name      = os.path.splitext(filename)[0]
    folds     = folds if folds is not None else PIPELINE["folds"]
    arm_name  = "smote"

    print()
    _print_separator()
    print(f"  Dataset  : {filename}   [arm={arm_name}]")
    print(f"  Target   : {target}  |  Models : {ml_models}")
    _print_separator()

    if not os.path.exists(csv_path):
        print(f"  [SKIP] File not found: {csv_path}")
        return None

    try:
        df = lib.load_df(csv_path)
        print(f"  Loaded   : {len(df)} rows × {len(df.columns)} cols")

        if target not in df.columns:
            print(f"  [SKIP] Target column '{target}' not in dataset.")
            return None

        col_defs = [{"field": c, "headerName": c} for c in df.columns]
        config = {
            "folds": folds,
            "targets": [target],
            "na_strategy": PIPELINE["na_strategy"],
            "ml_config": {"selected_models": ml_models, "params": {}},
        }

        print(f"  Running SMOTE baseline ({folds} folds, arm={arm_name}) …")
        results = run_smote_dataset_benchmark(("smote-lib-task", filename, df.to_dict('records'), col_defs, config))

        if not results:
            print("  [WARN] Pipeline returned empty results.")
            return None

        results["dataset_name"] = name
        fold_results = results.get("results", [])

        json_path = os.path.join(out_dir, f"{name}_metrics.json")
        _save_json(results, json_path)
        print(f"  Saved JSON  → {json_path}")

        per_fold = _per_fold_rows(fold_results)
        if per_fold:
            per_fold_csv = os.path.join(out_dir, f"{name}_per_fold.csv")
            pd.DataFrame(per_fold).to_csv(per_fold_csv, index=False)
            print(f"  Saved Folds → {per_fold_csv}")

        pdf_path = os.path.join(out_dir, f"{name}_report.pdf")
        lib.generate_pdf_report(results, pdf_path, original_df=df)
        print(f"  Saved PDF   → {pdf_path}")

        ml_str  = _ml_summary_str(results.get("overall_ml", {}))
        avg_fid = results.get("avg_fidelity", 0)
        print(f"  ✓  Avg Fidelity (JSD): {avg_fid:.1f}%  |  {ml_str}")

        summary_row: dict = {
            "dataset": filename,
            "target":  target,
            "rows":    len(df),
            "cols":    len(df.columns),
            "folds":   folds,
            "model":   "smote",
            "arm":     arm_name,
        }
        summary_row.update(_flat_fidelity_row(fold_results))
        summary_row.update(_flat_class_fidelity_row(fold_results))
        summary_row.update(_flat_generation_stats_row(fold_results))
        summary_row.update(_flat_ml_row(results.get("overall_ml", {})))

        return {
            "summary_row": summary_row,
            "per_fold_series": _per_fold_series(fold_results, ml_models),
        }

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — batch mode
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(resume: "str | None" = None) -> None:
    """Single full-pipeline (every module enabled) run across every dataset."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, resumed = _resolve_run_dir(resume, "run", timestamp)
    checkpoint = _load_checkpoint(run_dir)

    _print_separator("═")
    print(f"  TabuLLM Batch Pipeline  —  {timestamp}")
    print(f"  Provider : {PIPELINE['provider']}  |  Model : {PIPELINE['model']}")
    print(f"  Folds    : {PIPELINE['folds']}  |  Strategy : {PIPELINE['strategy']}")
    print(f"  Datasets : {len(DATASET_REGISTRY)}")
    print(f"  Output   : {run_dir}")
    if resumed:
        print(f"  Resuming : {len(checkpoint['completed'])}/{len(DATASET_REGISTRY)} datasets already completed")
    _print_separator("═")

    lib = TabuLLM(
        provider=PIPELINE["provider"],
        model=PIPELINE["model"],
        endpoint_url=PIPELINE["endpoint_url"],
    )

    unit_config = {"folds": PIPELINE["folds"]}
    summary_rows: list[dict] = []
    failed: list[str] = []

    for filename, ds_cfg in DATASET_REGISTRY.items():
        cached = _checkpoint_lookup(checkpoint, filename, unit_config)
        if cached is not None:
            print(f"  [SKIP] {filename}: already completed (resumed from checkpoint).")
            summary_rows.append(cached["summary_row"])
            continue

        out = _run_dataset_arm(lib, filename, ds_cfg, run_dir, arm_name="full", folds=PIPELINE["folds"])
        if out is None:
            failed.append(filename)
        else:
            summary_rows.append(out["summary_row"])
            _mark_unit_done(run_dir, checkpoint, filename, out["summary_row"], out["per_fold_series"], unit_config)

    print()
    _print_separator("═")
    if summary_rows:
        print(f"  BATCH COMPLETE  —  {len(summary_rows)} succeeded, {len(failed)} failed")
        _write_summary_csv(summary_rows, run_dir, "batch_summary.csv")
    else:
        print("  BATCH COMPLETE  —  No datasets were processed successfully.")
    _print_separator("═")
    if failed:
        print(f"\n  Failed datasets: {failed}")
        print(f"  Re-run to retry just the failed/remaining ones:")
        print(f"    python run_batch_pipeline.py --resume \"{run_dir}\"")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ablation mode
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(resume: "str | None" = None) -> None:
    """
    Runs the full batch pipeline once per ABLATION_ARMS entry (each arm = one module
    toggled off, or a context-strategy variant). Defaults to the SAME fold count as a
    standalone batch run (TABULLM_FOLDS) so every arm carries the same statistical power
    — cutting folds to save time would undermine the paired significance test below.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, resumed = _resolve_run_dir(resume, "ablation", timestamp)

    ablation_folds = int(os.getenv("TABULLM_ABLATION_FOLDS", str(PIPELINE["folds"])))
    _datasets_env  = os.getenv("TABULLM_ABLATION_DATASETS", "")
    dataset_subset = (
        {f.strip() for f in _datasets_env.split(",") if f.strip()}
        if _datasets_env else set(DATASET_REGISTRY.keys())
    )

    checkpoint = _load_checkpoint(run_dir)
    total_units = len(ALL_ARMS) * len(dataset_subset)

    _print_separator("═")
    print(f"  TabuLLM Ablation Study  —  {timestamp}")
    print(f"  Provider : {PIPELINE['provider']}  |  Model : {PIPELINE['model']}")
    print(f"  Folds    : {ablation_folds}  (same rigor as a standalone batch run)")
    print(f"  Arms     : {ALL_ARMS}")
    print(f"  Datasets : {len(dataset_subset)}")
    print(f"  Output   : {run_dir}")
    if resumed:
        print(f"  Resuming : {len(checkpoint['completed'])}/{total_units} (arm, dataset) units already completed")
    _print_separator("═")

    lib = TabuLLM(
        provider=PIPELINE["provider"],
        model=PIPELINE["model"],
        endpoint_url=PIPELINE["endpoint_url"],
    )

    # Cross-model sweep arms each need their own TabuLLM instance (different model id) --
    # built lazily and cached here so every dataset in that arm reuses the same instance.
    model_sweep_libs: dict[str, TabuLLM] = {}

    def _lib_for_arm(arm_name: str) -> TabuLLM:
        model_id = MODEL_SWEEP_ARMS[arm_name]
        if model_id not in model_sweep_libs:
            model_sweep_libs[model_id] = TabuLLM(
                provider=PIPELINE["provider"],
                model=model_id,
                endpoint_url=PIPELINE["endpoint_url"],
            )
        return model_sweep_libs[model_id]

    all_summary_rows: list[dict] = []                    # every (dataset, arm) row
    series_by_key: dict[tuple, dict] = {}                 # (dataset, arm) -> per-fold series
    unit_config = {"folds": ablation_folds}               # what fixes each unit's fold seeds

    for arm_name in ALL_ARMS:
        overrides = ABLATION_ARMS.get(arm_name, {})
        arm_dir = os.path.join(run_dir, arm_name)
        os.makedirs(arm_dir, exist_ok=True)

        print()
        _print_separator("═")
        if arm_name == "smote":
            print(f"  ARM: {arm_name}   (SMOTE/SMOTENC baseline, no LLM)")
        elif arm_name == "ponder_strategy":
            print(f"  ARM: {arm_name}   (LLM-chosen hyperparameters via ponder_hyperparameters())")
        elif arm_name in MODEL_SWEEP_ARMS:
            print(f"  ARM: {arm_name}   (cross-model sweep, model={MODEL_SWEEP_ARMS[arm_name]})")
        else:
            print(f"  ARM: {arm_name}   overrides={overrides or '(none — full pipeline)'}")
        _print_separator("═")

        arm_summary_rows: list[dict] = []
        arm_failed: list[str] = []

        for filename, ds_cfg in DATASET_REGISTRY.items():
            if filename not in dataset_subset:
                continue

            unit_key = f"{arm_name}::{filename}"
            cached = _checkpoint_lookup(checkpoint, unit_key, unit_config)
            if cached is not None:
                print(f"  [{arm_name}] {filename}: already completed — skipping (resumed from checkpoint).")
                arm_summary_rows.append(cached["summary_row"])
                all_summary_rows.append(cached["summary_row"])
                series_by_key[(filename, arm_name)] = cached["per_fold_series"]
                continue

            if arm_name == "smote":
                out = _run_smote_arm(lib, filename, ds_cfg, arm_dir, folds=ablation_folds)
            elif arm_name == "ponder_strategy":
                out = _run_dataset_arm(
                    lib, filename, ds_cfg, arm_dir,
                    arm_name=arm_name, folds=ablation_folds, ponder=True,
                )
            elif arm_name in MODEL_SWEEP_ARMS:
                out = _run_dataset_arm(
                    _lib_for_arm(arm_name), filename, ds_cfg, arm_dir,
                    arm_name=arm_name, folds=ablation_folds,
                )
            else:
                out = _run_dataset_arm(
                    lib, filename, ds_cfg, arm_dir,
                    arm_name=arm_name, config_overrides=overrides, folds=ablation_folds,
                )
            if out is None:
                arm_failed.append(filename)
                continue
            arm_summary_rows.append(out["summary_row"])
            all_summary_rows.append(out["summary_row"])
            series_by_key[(filename, arm_name)] = out["per_fold_series"]
            _mark_unit_done(run_dir, checkpoint, unit_key, out["summary_row"], out["per_fold_series"], unit_config)

        print()
        if arm_summary_rows:
            _write_summary_csv(arm_summary_rows, arm_dir, "batch_summary.csv")
        if arm_failed:
            print(f"  [{arm_name}] Failed datasets: {arm_failed}")
            print(f"  Re-run to retry just the failed/remaining units:")
            print(f"    python run_batch_pipeline.py --ablation --resume \"{run_dir}\"")

    # ── Cross-arm comparison ────────────────────────────────────────────────
    print()
    _print_separator("═")
    if not all_summary_rows:
        print("  ABLATION COMPLETE — no results were produced.")
        _print_separator("═")
        return

    comparison_csv = os.path.join(run_dir, "ablation_summary.csv")
    pd.DataFrame(all_summary_rows).to_csv(comparison_csv, index=False)
    print(f"  Combined arm × dataset summary → {comparison_csv}")

    # Marginal contribution of each module: metric - full_baseline_metric, per dataset,
    # plus a paired Wilcoxon p-value computed fold-for-fold against "full" (valid because
    # every arm ran the same folds with identical train/test splits — see _per_fold_series).
    delta_rows = []
    for filename in DATASET_REGISTRY:
        if filename not in dataset_subset:
            continue
        baseline_row = next((r for r in all_summary_rows if r["dataset"] == filename and r["arm"] == "full"), None)
        if baseline_row is None:
            continue
        baseline_series = series_by_key.get((filename, "full"), {})

        dataset_delta_rows: dict[str, dict] = {}   # arm_name -> delta_row
        raw_pvalues: dict[tuple, float] = {}        # (arm_name, key) -> pvalue

        for arm_name in ALL_ARMS:
            if arm_name == "full":
                continue
            row = next((r for r in all_summary_rows if r["dataset"] == filename and r["arm"] == arm_name), None)
            if row is None:
                continue

            delta_row = {"dataset": filename, "arm": arm_name}
            for key, val in row.items():
                if key in _IDENTITY_COLS or not isinstance(val, (int, float)):
                    continue
                base_val = baseline_row.get(key)
                if isinstance(base_val, (int, float)):
                    delta_row[f"{key}_vs_full"] = round(val - base_val, 4)

            arm_series = series_by_key.get((filename, arm_name), {})
            for key, base_vals in baseline_series.items():
                arm_vals = arm_series.get(key)
                if arm_vals and len(arm_vals) == len(base_vals):
                    p = _paired_pvalue(base_vals, arm_vals)
                    delta_row[f"{key}_pvalue_vs_full"] = round(p, 4) if p is not None else None
                    if p is not None:
                        raw_pvalues[(arm_name, key)] = p

            dataset_delta_rows[arm_name] = delta_row

        # Benjamini-Hochberg correction across this dataset's full family of (arm, metric)
        # comparisons vs "full" — avoids inflating the false-discovery rate that comes
        # from reading each raw p-value in isolation across many arms and metrics at once.
        apply_fdr_correction(
            list(raw_pvalues.keys()),
            get_fn=lambda ref: raw_pvalues.get(ref),
            set_fn=lambda ref, q: dataset_delta_rows[ref[0]].__setitem__(f"{ref[1]}_qvalue_vs_full", q),
        )

        delta_rows.extend(dataset_delta_rows.values())

    if delta_rows:
        deltas_csv = os.path.join(run_dir, "ablation_deltas.csv")
        pd.DataFrame(delta_rows).to_csv(deltas_csv, index=False)
        print(f"  Ablation deltas (marginal contribution per module) → {deltas_csv}")

    _print_separator("═")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — stochasticity mode
# ─────────────────────────────────────────────────────────────────────────────

def run_stochasticity_study(resume: "str | None" = None) -> None:
    """
    Generation-stochasticity decomposition (see pipeline_manager.run_stochasticity_check):
    for a handful of datasets, fixes ONE train/test split and ONE context sample, then
    repeats LLM generation on those identical inputs. The resulting std is pure LLM
    sampling noise — compare it against the same metric's fold-to-fold std in a normal
    batch_summary.csv run (which mixes data-split AND generation variance) to see how
    much of the reported fold-to-fold spread is actually generation noise.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, resumed = _resolve_run_dir(resume, "stochasticity", timestamp)

    repeats = int(os.getenv("TABULLM_STOCHASTICITY_REPEATS", "5"))
    _datasets_env = os.getenv("TABULLM_STOCHASTICITY_DATASETS", "")
    dataset_subset = (
        [f.strip() for f in _datasets_env.split(",") if f.strip()]
        if _datasets_env else list(DATASET_REGISTRY.keys())[:2]
    )

    checkpoint = _load_checkpoint(run_dir)

    _print_separator("═")
    print(f"  TabuLLM Generation-Stochasticity Check  —  {timestamp}")
    print(f"  Provider : {PIPELINE['provider']}  |  Model : {PIPELINE['model']}")
    print(f"  Repeats  : {repeats}  |  Datasets : {dataset_subset}")
    print(f"  Output   : {run_dir}")
    if resumed:
        print(f"  Resuming : {len(checkpoint['completed'])}/{len(dataset_subset)} datasets already completed")
    _print_separator("═")

    lib = TabuLLM(
        provider=PIPELINE["provider"],
        model=PIPELINE["model"],
        endpoint_url=PIPELINE["endpoint_url"],
    )

    unit_config = {"repeats": repeats}   # count of independent repeats -- see _mark_unit_done
    summary_rows: list[dict] = []
    for filename in dataset_subset:
        cached = _checkpoint_lookup(checkpoint, filename, unit_config)
        if cached is not None:
            print(f"  [SKIP] {filename}: already completed (resumed from checkpoint).")
            summary_rows.append(cached["summary_row"])
            continue

        ds_cfg = DATASET_REGISTRY.get(filename)
        if ds_cfg is None:
            print(f"  [SKIP] '{filename}' not found in DATASET_REGISTRY.")
            continue
        csv_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(csv_path):
            print(f"  [SKIP] File not found: {csv_path}")
            continue

        target = ds_cfg["target"]
        ml_models = ds_cfg.get("ml", ["rf"])
        name = os.path.splitext(filename)[0]

        print()
        _print_separator()
        print(f"  Dataset  : {filename}  |  Target : {target}")
        _print_separator()

        df = lib.load_df(csv_path)
        if target not in df.columns:
            print(f"  [SKIP] Target column '{target}' not in dataset.")
            continue

        col_defs = [{"field": c, "headerName": c} for c in df.columns]
        config = {
            "targets": [target],
            "strategy": PIPELINE["strategy"],
            "algorithm": PIPELINE["model"],
            "na_strategy": PIPELINE["na_strategy"],
            "ctx_size_pct": PIPELINE["ctx_size_pct"],
            "gen_size_pct": PIPELINE["gen_size_pct"],
            "ml_config": {"selected_models": ml_models, "params": {}},
            "repeats": repeats,
        }

        payload = ("stochasticity-task", filename, df.to_dict('records'), col_defs, config)
        result = run_stochasticity_check(payload, repeats=repeats)

        repeat_results = result.get("repeat_results", [])
        if not repeat_results:
            print("  [WARN] No repeats succeeded.")
            continue

        per_repeat = _per_repeat_rows(repeat_results)
        per_repeat_csv = os.path.join(run_dir, f"{name}_per_repeat.csv")
        pd.DataFrame(per_repeat).to_csv(per_repeat_csv, index=False)
        print(f"  Saved Repeats → {per_repeat_csv}")

        summary_row: dict = {
            "dataset": filename,
            "target": target,
            "repeats": len(repeat_results),
            "split_train": result["split_sizes"]["train"],
            "split_test": result["split_sizes"]["test"],
            "split_context": result["split_sizes"]["context"],
        }
        summary_row.update(_flat_fidelity_row(repeat_results))
        summary_row.update(_flat_class_fidelity_row(repeat_results))
        summary_row.update(_flat_generation_stats_row(repeat_results))
        summary_row.update(_flat_ml_row(result.get("overall_ml", {})))
        summary_rows.append(summary_row)
        _mark_unit_done(run_dir, checkpoint, filename, summary_row, None, unit_config)

    print()
    _print_separator("═")
    if summary_rows:
        summary_csv = os.path.join(run_dir, "stochasticity_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"  Stochasticity summary → {summary_csv}")
        print()
        print("  NOTE: the '_std' columns above are computed across REPEATS at a single")
        print("  fixed train/test split and context sample — i.e. pure LLM generation")
        print("  noise. Compare them against the same metric's '_std' column in a normal")
        print("  batch_summary.csv run (which mixes data-split variance AND generation")
        print("  variance) to see how much of the reported fold-to-fold spread is")
        print("  actually LLM sampling noise rather than genuine data-split sensitivity.")
    else:
        print("  STOCHASTICITY CHECK COMPLETE — no datasets were processed successfully.")
    _print_separator("═")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabuLLM batch pipeline / ablation study runner.")
    parser.add_argument("--ablation", action="store_true",
                         help="Run the ablation sweep (repeats the full batch pipeline once "
                              "per module combination) instead of a single full-module batch run.")
    parser.add_argument("--stochasticity", action="store_true",
                         help="Run the generation-stochasticity decomposition (repeats LLM "
                              "generation on a fixed split/context to isolate sampling noise).")
    parser.add_argument("--resume", nargs="?", const="__latest__", default=None, metavar="RUN_DIR",
                         help="Resume an interrupted run instead of starting fresh. Bare --resume "
                              "picks the most recent run of the same mode under results/ that has "
                              "a checkpoint.json; --resume <path> resumes that exact run directory. "
                              "Units already recorded in checkpoint.json are skipped -- UNLESS the "
                              "fold/repeat count now differs from what produced them, in which case "
                              "they're recomputed (fold train/test splits are seeded purely from the "
                              "fold count, so a mismatched count means different, incomparable seeds).")
    args = parser.parse_args()

    try:
        if args.stochasticity or os.getenv("TABULLM_MODE", "batch").lower() == "stochasticity":
            run_stochasticity_study(resume=args.resume)
        elif args.ablation or os.getenv("TABULLM_MODE", "batch").lower() == "ablation":
            run_ablation(resume=args.resume)
        else:
            run_batch(resume=args.resume)
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Every unit completed so far is already saved to this run's "
              "checkpoint.json. Re-run with --resume to pick up where this left off.")
        sys.exit(130)
