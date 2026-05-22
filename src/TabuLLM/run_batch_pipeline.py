"""
run_batch_pipeline.py
─────────────────────
Batch runner: runs the full TabuLLM augmentation pipeline over every dataset
in the configured DATA_DIR, then saves per-dataset JSON metrics + PDF reports
and a consolidated summary CSV containing ALL fidelity and ML metrics.

Usage
─────
    cd src/TabuLLM
    python run_batch_pipeline.py

    # Gemini backend:
    set TABULLM_PROVIDER=gemini
    set GOOGLE_API_KEY=<your-key>
    python run_batch_pipeline.py

    # Override model:
    set TABULLM_MODEL=meta-llama/llama-3-8b
    python run_batch_pipeline.py

Prerequisites
─────────────
- For 'local' provider: LM Studio (or any OpenAI-compatible server) running at
  http://127.0.0.1:1234  (override with LOCAL_LLM_URL env var).
- For 'gemini' provider: set GOOGLE_API_KEY.
"""

import os
import sys
import json
import datetime
import traceback

import numpy as np
import pandas as pd

# Make sure the package root is importable regardless of where you call this from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabullm import TabuLLM  # noqa: E402  (after sys.path tweak)

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
    "gen_size_pct": int(os.getenv("TABULLM_GEN_PCT", "80")),      # % of train split size to generate
}

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

# ML metrics emitted per model
ML_METRICS = ["accuracy", "f1", "rmse", "r2"]


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
    Average every fidelity / structural metric across folds.
    Returns one column per metric: mean and ±std.
    E.g. 'Synthetic Fidelity (JSD)_mean', 'Synthetic Fidelity (JSD)_std'
    """
    out: dict = {}
    for key in FIDELITY_METRICS:
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"{key}_mean"] = round(float(np.mean(vals)), 3)
            out[f"{key}_std"]  = round(float(np.std(vals)),  3)
    return out


def _flat_ml_row(overall_ml: dict) -> dict:
    """
    Flatten ALL model × metric combinations into a flat dict for the CSV.
    Columns follow the pattern:  <model>_<metric>_real/aug/delta
    Only the metrics that were actually computed for that model are included.
    """
    out: dict = {}
    for model, metrics in overall_ml.items():
        if not isinstance(metrics, dict):
            continue
        for metric in ML_METRICS:
            if metric not in metrics or not isinstance(metrics[metric], dict):
                continue
            m = metrics[metric]
            out[f"{model}_{metric}_real"]  = round(float(m.get("real",  0) or 0), 4)
            out[f"{model}_{metric}_aug"]   = round(float(m.get("aug",   0) or 0), 4)
            out[f"{model}_{metric}_delta"] = round(float(m.get("delta", 0) or 0), 4)
            # Also capture the ±std values so the CSV is fully self-contained
            if m.get("real_sd") is not None:
                out[f"{model}_{metric}_real_sd"]  = round(float(m.get("real_sd",  0) or 0), 4)
                out[f"{model}_{metric}_aug_sd"]   = round(float(m.get("aug_sd",   0) or 0), 4)
                out[f"{model}_{metric}_delta_sd"] = round(float(m.get("delta_sd", 0) or 0), 4)
    return out


def _ml_summary_str(overall_ml: dict) -> str:
    """One-liner human-readable ML summary for console output."""
    parts = []
    for model, metrics in overall_ml.items():
        for metric in ("accuracy", "f1", "r2"):
            if metric in metrics and isinstance(metrics[metric], dict):
                delta = metrics[metric].get("delta", 0) or 0
                parts.append(f"{model.upper()} {metric}={delta:+.3f}")
                break
    return " | ".join(parts) if parts else "(no ML results)"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_batch() -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    _print_separator("═")
    print(f"  TabuLLM Batch Pipeline  —  {timestamp}")
    print(f"  Provider : {PIPELINE['provider']}  |  Model : {PIPELINE['model']}")
    print(f"  Folds    : {PIPELINE['folds']}  |  Strategy : {PIPELINE['strategy']}")
    print(f"  Datasets : {len(DATASET_REGISTRY)}")
    print(f"  Output   : {run_dir}")
    _print_separator("═")

    lib = TabuLLM(
        provider=PIPELINE["provider"],
        model=PIPELINE["model"],
        endpoint_url=PIPELINE["endpoint_url"],
    )

    summary_rows: list[dict] = []
    failed: list[str] = []

    for filename, ds_cfg in DATASET_REGISTRY.items():
        csv_path  = os.path.join(DATA_DIR, filename)
        target    = ds_cfg["target"]
        ml_models = ds_cfg.get("ml", ["rf"])
        name      = os.path.splitext(filename)[0]

        print()
        _print_separator()
        print(f"  Dataset  : {filename}")
        print(f"  Target   : {target}  |  Models : {ml_models}")
        _print_separator()

        # ── guard: file exists ────────────────────────────────────────────
        if not os.path.exists(csv_path):
            print(f"  [SKIP] File not found: {csv_path}")
            failed.append(filename)
            continue

        try:
            # 1. Load
            df = lib.load_df(csv_path)
            print(f"  Loaded   : {len(df)} rows × {len(df.columns)} cols")

            if target not in df.columns:
                print(f"  [SKIP] Target column '{target}' not in dataset.")
                failed.append(filename)
                continue

            # 2. Run pipeline
            print(f"  Running augmentation ({PIPELINE['folds']} folds) …")
            results = lib.augment(
                df,
                targets=[target],
                folds=PIPELINE["folds"],
                strategy=PIPELINE["strategy"],
                ml_models=ml_models,
                na_strategy=PIPELINE["na_strategy"],
                ctx_size_pct=PIPELINE["ctx_size_pct"],
                gen_size_pct=PIPELINE["gen_size_pct"],
            )

            if not results:
                print("  [WARN] Pipeline returned empty results.")
                failed.append(filename)
                continue

            results["filename"]     = filename
            results["dataset_name"] = name

            # 3. Save JSON (metrics only — no raw row payloads)
            json_path = os.path.join(run_dir, f"{name}_metrics.json")
            _save_json(results, json_path)
            print(f"  Saved JSON  → {json_path}")

            # 4. Export PDF report
            pdf_path = os.path.join(run_dir, f"{name}_report.pdf")
            lib.generate_pdf_report(results, pdf_path, original_df=df)
            print(f"  Saved PDF   → {pdf_path}")

            # 5. Console summary
            ml_str = _ml_summary_str(results.get("overall_ml", {}))
            avg_fid = results.get("avg_fidelity", 0)
            print(f"  ✓  Avg Fidelity (JSD): {avg_fid:.1f}%  |  {ml_str}")

            # 6. Accumulate summary row
            #    Identity columns first, then every fidelity metric (mean ± std
            #    across folds), then every ML model × metric combination.
            summary_row: dict = {
                "dataset":  filename,
                "target":   target,
                "rows":     len(df),
                "cols":     len(df.columns),
                "folds":    PIPELINE["folds"],
                "model":    PIPELINE["model"],
            }
            summary_row.update(_flat_fidelity_row(results.get("results", [])))
            summary_row.update(_flat_ml_row(results.get("overall_ml", {})))
            summary_rows.append(summary_row)

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            failed.append(filename)

    # ── Write consolidated summary CSV ────────────────────────────────────────
    print()
    _print_separator("═")
    if summary_rows:
        summary_df  = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(run_dir, "batch_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        print(f"  BATCH COMPLETE  —  {len(summary_rows)} succeeded, {len(failed)} failed")
        print(f"  Summary CSV → {summary_csv}")
        _print_separator("═")

        # Show a condensed view in the terminal (identity + key metrics only)
        _KEY_COLS = ["dataset", "target", "rows",
                     "Synthetic Fidelity (JSD)_mean",
                     "Correlation Stability_mean",
                     "Privacy Risk_mean",
                     "Coverage Score_mean"]
        display_cols = [c for c in _KEY_COLS if c in summary_df.columns]
        print(summary_df[display_cols].to_string(index=False))
        print(f"\n  Full metrics saved to: {summary_csv}")
    else:
        print("  BATCH COMPLETE  —  No datasets were processed successfully.")
        _print_separator("═")

    if failed:
        print(f"\n  Failed datasets: {failed}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_batch()
