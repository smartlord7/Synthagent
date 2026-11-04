import pandas as pd
import numpy as np
import sys
import os
from typing import Callable, Optional

# Ensure the root services and utils are accessible to the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.llm_client import BaseLLMProvider, LocalOpenAIProvider, GeminiProvider, OpenAIProvider, AnthropicProvider
from services.pipeline_manager import PipelineManager
from services.report_generator import generate_benchmark_pdf
from services.auditor import DatasetAuditor
from services.prompt_manager import prompt_manager
from utils.json_utils import robust_json_loads
from utils.logging_config import logger

# Numeric keys produced by calculate_metrics() — used to average across folds
_METRIC_KEYS = [
    "Synthetic Fidelity (JSD)", "Correlation Stability", "Distribution Consistency",
    "Entropy Preservation", "Mean Z Score Drift", "Categorical Fidelity",
    "Privacy Risk", "Data Density", "Coverage Score", "Feature Accuracy", "Novelty Score",
]


class TabuLLM:
    """
    The primary developer interface for TabuLLM.
    This class encapsulates data loading, AI-driven insights,
    synthetic data generation, and audit reporting.
    """

    llm: BaseLLMProvider

    def __init__(self, provider: str = "local", model: str = "google/gemma-3-4b", endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the TabuLLM engine.

        Args:
            provider (str): LLM provider to use ('local', 'gemini', or 'openai'). Default 'local'.
            model (str): Model ID to use. Default 'google/gemma-3-4b'.
            endpoint_url (str, optional): Custom endpoint for local LLM (e.g., LM Studio) or an
                OpenAI-compatible proxy (e.g., Azure OpenAI / OpenRouter) when provider='openai'.
            api_key (str, optional): API key for provider='openai'. Falls back to OPENAI_API_KEY
                if not given.
        """
        if provider == "local":
            self.llm = LocalOpenAIProvider(
                endpoint_url=endpoint_url or os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions"),
                model_name=model
            )
        elif provider == "gemini":
            self.llm = GeminiProvider(model_name=model)
        elif provider == "openai":
            self.llm = OpenAIProvider(model_name=model, api_key=api_key, endpoint_url=endpoint_url)
        elif provider == "anthropic":
            self.llm = AnthropicProvider(model_name=model, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.default_model = model

    # ──────────────────────────────────────────────────────────────────────────
    # Data
    # ──────────────────────────────────────────────────────────────────────────

    def get_models(self) -> list[dict]:
        """
        Queries the backend for a list of available AI models and their context windows.

        Returns:
            list[dict]: List of model objects with 'id' and 'max_context'.
        """
        return self.llm.get_available_models() or []

    def load_df(self, path: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV or Excel file.

        Args:
            path (str): Local path to the data file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(path)
        raise ValueError("Unsupported format. Use .csv or .xlsx")

    # ──────────────────────────────────────────────────────────────────────────
    # AI
    # ──────────────────────────────────────────────────────────────────────────

    def suggest_targets(self, df: pd.DataFrame, single_target: bool = True) -> list[str]:
        """
        Uses AI to automatically suggest potential target columns for ML.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            single_target (bool): If True, prioritize a single primary target.

        Returns:
            list[str]: A list of suggested target column names.
        """
        return self.llm.detect_target_columns(df.head(50), single_target=single_target, algorithm=self.default_model) or []

    def get_insights(self, df: pd.DataFrame, metadata: str = "") -> str:
        """
        Generates a structured natural language description of the dataset.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            metadata (str): Optional user-provided context or hints.

        Returns:
            str: AI-generated markdown insights.
        """
        return self.llm.get_dataset_description(df.head(50), provided_metadata=metadata, algorithm=self.default_model) or ""

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def augment(
        self,
        df: pd.DataFrame,
        targets: list[str],
        num_rows: int = 0,
        folds: int = 3,
        strategy: str = "random",
        ml_models: Optional[list[str]] = None,
        na_strategy: str = "impute",
        ctx_size_pct: int = 20,
        gen_size_pct: int = 80,
        join_context: bool = True,
        inject_inferred_rules: bool = True,
        apply_post_processing: bool = True,
        raw_prompting: bool = False,
        enforce_boundaries: bool = False,
        structured_output: bool = False,
        temperature: float = 0.3,
        resume_fold_results: Optional[dict] = None,
        on_fold_done: Optional[Callable[[int, dict], None]] = None,
    ) -> dict:
        """
        Runs the full High-Reliability Augmentation pipeline.
        This includes cross-validation, fidelity auditing, and ML performance comparison.

        Args:
            df (pd.DataFrame): Original dataset.
            targets (list[str]): Target column(s) for ML evaluation.
            num_rows (int): Explicit synthetic rows per fold (0 = use gen_size_pct).
            folds (int): Number of cross-validation folds. Default 3.
            strategy (str): Context selection strategy ('random', 'dense', 'diverse').
            ml_models (list[str]): Models to evaluate ('rf', 'lr', 'tree', 'nb', 'xgb').
            na_strategy (str): Strategy for missing values ('impute' or 'drop').
            ctx_size_pct (int): % of train split used as LLM context rows. Default 20.
            gen_size_pct (int): % of train split size to generate (when num_rows==0). Default 80.
            join_context (bool): Whether generated rows are blended with real context rows
                for fidelity/appendix purposes. Default True. Ablation toggle.
            inject_inferred_rules (bool): Whether inferred dtype/boundary rules are sent to
                the LLM as generation constraints. Default True. Ablation toggle.
            apply_post_processing (bool): Whether code-side rule post-processing (noise,
                rounding, etc.) is applied to generated rows. Default True. Ablation toggle.
            raw_prompting (bool): No-controller ablation. The same prompts and context,
                but strict JSON parsing with no repair, no schema bridge, rule violations
                counted but not corrected, no overflow retry, no top-up and no
                post-processing. Default False.
            enforce_boundaries (bool): Whether the validator also enforces the inferred
                numeric ranges (observed min/max relaxed by 10%) and categorical domains,
                not only feature types. Default False.
            structured_output (bool): Structured-output baseline. The no-controller path,
                but the endpoint constrains decoding to a JSON schema built from the
                real columns and types (OpenAI response_format json_schema). Default False.
            temperature (float): LLM sampling temperature for generation. Default 0.3.
                See ponder_hyperparameters() for an LLM-recommended alternative.
            resume_fold_results (dict | None): {fold_num: fold_result} for folds already
                computed by a prior interrupted call -- those folds are reused as-is instead
                of being regenerated. Default None (no folds cached).
            on_fold_done (Callable[[int, dict], None] | None): called with (fold_num,
                fold_result) immediately after each NEWLY computed fold, so a caller can
                persist per-fold progress (e.g. to a checkpoint file) without waiting for
                every fold in this call to finish. Default None.

        Returns:
            dict: Results including augmented_data, fidelity scores, and ML benchmarks.
        """
        if ml_models is None:
            ml_models = ["rf"]

        # Strip internal UI columns
        df_clean = df.copy()
        df_clean.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')

        col_defs = [{"field": c, "headerName": c} for c in df_clean.columns]
        config = {
            "folds": folds,
            "strategy": strategy,
            "algorithm": self.default_model,
            "targets": targets,
            "num_empty": num_rows,       # 0 → pipeline falls back to gen_size_pct
            "ctx_size_pct": ctx_size_pct,
            "gen_size_pct": gen_size_pct,
            "join_context": join_context,
            "inject_inferred_rules": inject_inferred_rules,
            "apply_post_processing": apply_post_processing,
            "raw_prompting": raw_prompting,
            "enforce_boundaries": enforce_boundaries,
            "structured_output": structured_output,
            "temperature": temperature,
            "ml_config": {"selected_models": ml_models, "params": {}},
            "na_strategy": na_strategy,
            "instructions": "",
            "desc_metadata": "",
            "_resume_fold_results": resume_fold_results or {},
            "_on_fold_done": on_fold_done,
        }

        task = ("lib-task", "dataset.csv", df_clean.to_dict('records'), col_defs, config)
        results = PipelineManager.run_pipeline([task])
        return results[0] if results else {}

    # ──────────────────────────────────────────────────────────────────────────
    # Audit
    # ──────────────────────────────────────────────────────────────────────────

    def run_audit(self, df: pd.DataFrame, targets: Optional[list[str]] = None) -> dict:
        """
        Runs a comprehensive Health & Ethics audit on the dataset.

        Args:
            df (pd.DataFrame): The dataset to audit.
            targets (list[str] | None): Target columns for fairness parity analysis.

        Returns:
            dict: Audit results including Structural, Fairness, Privacy, and Quality metrics.
        """
        auditor_service = DatasetAuditor()
        target_col: Optional[str] = targets[0] if targets else None
        sensitive_cols = DatasetAuditor.detect_sensitive_columns(df)

        return {
            "structural":       auditor_service.get_structural_metrics(df),
            "outliers":         auditor_service.get_outlier_metrics(df),
            "fairness":         auditor_service.get_fairness_metrics(df, target_col, sensitive_cols),  # type: ignore[arg-type]
            "privacy":          auditor_service.get_privacy_metrics(df),
            "proxies":          auditor_service.get_proxy_risks(df, sensitive_cols),
            "quality":          auditor_service.get_quality_metrics(df, target_col),  # type: ignore[arg-type]
            "quasi_identifiers": auditor_service.get_quasi_identifiers(df),
        }

    def ponder_strategy(self, df: pd.DataFrame, targets: Optional[list[str]] = None) -> str:
        """
        AI Strategy: Performs a deep ethical analysis of the audit results and
        generates a prioritized augmentation plan.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            targets (list[str] | None): Target columns for bias context.

        Returns:
            str: AI-generated strategic guidance (Markdown).
        """
        audit = self.run_audit(df, targets)
        struct = audit["structural"]

        context = {
            "report": audit,
            "overall_sparsity": struct['overall_sparsity'] * 100,
            "top_sparse_cols": [c for c, s in struct['sparsity'].items() if s > 0.05][:3],
            "skewness": {c: v for c, v in struct.get('skewness', {}).items() if abs(v) > 2},
            "target_col": targets[0] if targets else "None Selected",
            "fairness": audit["fairness"],
            "privacy": audit["privacy"],
            "proxies": audit["proxies"],
            "quality": audit["quality"],
            "outliers": audit["outliers"],
        }

        prompt = prompt_manager.get_prompt("ponder_audit.j2", **context)
        return self.llm.call_llm(
            messages=[{"role": "user", "content": prompt}],
            model_id=self.default_model,
            temperature=0.3,
        ) or ""

    def ponder_hyperparameters(
        self,
        df: pd.DataFrame,
        targets: Optional[list[str]] = None,
        defaults: Optional[dict] = None,
    ) -> dict:
        """
        The structured, empirically-testable counterpart to ponder_strategy(): asks the
        LLM to recommend concrete generation hyperparameters (context ratio, sampling
        strategy, temperature) from the dataset's audit profile, returned as a
        validated dict instead of free-text prose. Never raises -- any missing/invalid
        field, or a totally unparseable response, silently falls back to `defaults` (or
        safe hard-coded defaults), so a malformed LLM response degrades to "use the
        fixed baseline config" rather than breaking the caller.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            targets (list[str] | None): Target columns for fairness/audit context.
            defaults (dict | None): Fallback values for any field the LLM omits or gets
                wrong. Defaults to {"ctx_size_pct": 20, "strategy": "random", "temperature": 0.7}.

        Returns:
            dict: {"ctx_size_pct": int, "strategy": str, "temperature": float}
        """
        result = {"ctx_size_pct": 20, "strategy": "random", "temperature": 0.7}
        if defaults:
            result.update(defaults)

        try:
            audit = self.run_audit(df, targets)
            context = {
                "report": audit,
                "target_col": targets[0] if targets else "None Selected",
            }
            prompt = prompt_manager.get_prompt("ponder_hyperparams.j2", **context)
            raw = self.llm.call_llm(
                messages=[{"role": "user", "content": prompt}],
                model_id=self.default_model,
                temperature=0.2,
            )
            parsed = robust_json_loads(raw) if raw else None
            if not isinstance(parsed, dict):
                logger.warning("ponder_hyperparameters: LLM response was not a JSON object; using defaults.")
                return result

            ctx_size_pct = parsed.get("ctx_size_pct")
            if isinstance(ctx_size_pct, (int, float)) and 5 <= ctx_size_pct <= 50:
                result["ctx_size_pct"] = int(ctx_size_pct)

            strategy = parsed.get("strategy")
            if strategy in ("random", "dense", "diverse"):
                result["strategy"] = strategy

            temperature = parsed.get("temperature")
            if isinstance(temperature, (int, float)) and 0.1 <= temperature <= 1.0:
                result["temperature"] = float(temperature)

        except Exception as e:
            logger.warning(f"ponder_hyperparameters failed ({e}); using defaults.")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def generate_pdf_report(
        self,
        pipeline_results: dict,
        output_path: str,
        original_df: Optional[pd.DataFrame] = None,
    ):
        """
        Exports the benchmark results and fidelity audits to a professional PDF.

        Args:
            pipeline_results (dict): The dictionary returned by .augment().
            output_path (str): Destination file path for the PDF.
            original_df (pd.DataFrame | None): The original dataset used for augmentation.
                When provided, enriches the report with real row/col counts,
                a per-feature inference table, and a baseline data sample.
        """
        fold_results: list[dict] = pipeline_results.get("results", [])

        # ── 1. Average numeric metrics across folds ────────────────────────
        avg_metrics: dict = {}
        for key in _METRIC_KEYS:
            vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
            if vals:
                mean_v = float(np.mean(vals))
                std_v  = float(np.std(vals))
                avg_metrics[key] = f"{mean_v:.1f} ± {std_v:.2f}"

        # ── 2. Dataset portrait ────────────────────────────────────────────
        if original_df is not None:
            num_rows_ds  = len(original_df)
            num_cols_ds  = len(original_df.columns)
            num_numeric  = len(original_df.select_dtypes(include='number').columns)
            num_cat      = num_cols_ds - num_numeric
            metadata = {
                "Rows": num_rows_ds,
                "Columns": num_cols_ds,
                "Numeric features": num_numeric,
                "Categorical features": num_cat,
                "Missing cells (%)": f"{original_df.isna().mean().mean() * 100:.1f}%",
            }
        else:
            metadata = {"Rows": "N/A", "Columns": "N/A"}

        # ── 3. Generation params ───────────────────────────────────────────
        first_fold = fold_results[0] if fold_results else {}
        gen_params = {
            "Algorithm": self.default_model,
            "Folds": len(fold_results),
            "Context %": f"{first_fold.get('ctx_size_pct', 20)}%",
            "Generation %": f"{first_fold.get('gen_size_pct', 80)}%",
            "Join context": str(first_fold.get('join_context', True)),
        }

        # ── 4. Per-feature inference table ────────────────────────────────
        inference_table: list = []
        if original_df is not None and fold_results:
            synth_rows = fold_results[0].get("synthetic_data", [])
            synth_df   = pd.DataFrame(synth_rows) if synth_rows else pd.DataFrame()
            numeric_cols = original_df.select_dtypes(include='number').columns.tolist()
            for col in numeric_cols[:12]:   # cap at 12 rows to keep PDF readable
                if col not in synth_df.columns:
                    continue
                orig_mean  = original_df[col].mean()
                synth_mean = synth_df[col].mean()
                shift      = synth_mean - orig_mean
                rel_shift  = abs(shift / orig_mean) if orig_mean != 0 else 0
                status     = "OK" if rel_shift < 0.10 else ("Warn" if rel_shift < 0.25 else "Drift")
                inference_table.append([
                    col,
                    f"{orig_mean:.3f}",
                    f"{synth_mean:.3f}",
                    f"{shift:+.3f}",
                    f"{rel_shift * 100:.1f}%",
                    status,
                ])

        # ── 5. Appendix: colour-coded fold samples ─────────────────────────
        # Colours: Blue=context, Green=generated, Yellow=train-non-ctx, Grey=test
        _BLUE   = (173, 216, 230)
        _GREEN  = (144, 238, 144)
        _YELLOW = (255, 255, 153)
        _GREY   = (220, 220, 220)

        appendix: list = []
        for i, fold in enumerate(fold_results, 1):
            rows: list  = []
            colors: list = []
            for row in fold.get("context_data", [])[:6]:
                rows.append(row); colors.append(_BLUE)
            for row in fold.get("synthetic_data", [])[:6]:
                rows.append(row); colors.append(_GREEN)
            for row in fold.get("train_non_ctx_data", [])[:4]:
                rows.append(row); colors.append(_YELLOW)
            for row in fold.get("test_data", [])[:4]:
                rows.append(row); colors.append(_GREY)

            if rows:
                appendix.append({
                    "name":        f"Fold {i}",
                    "sample":      pd.DataFrame(rows),
                    "row_colors":  colors,
                    "ctx_size_pct": fold.get("ctx_size_pct"),
                    "gen_size_pct": fold.get("gen_size_pct"),
                })

        # ── 6. Assemble report_data and render ────────────────────────────
        report_data = {
            "executive_summary": {
                "avg_fidelity":  pipeline_results.get("avg_fidelity", 0),
                "global_metrics": avg_metrics,
            },
            "datasets": {
                pipeline_results.get("dataset_id", "lib-ds"): {
                    "filename":        pipeline_results.get("filename", "dataset.csv"),
                    "metadata":        metadata,
                    "params":          {"generation": gen_params},
                    "metrics":         avg_metrics,
                    "inference_table": inference_table,
                    "ml_results":      pipeline_results.get("overall_ml", {}),
                    "appendix":        appendix,
                    "samples":         {
                        "orig": original_df.head(20) if original_df is not None else pd.DataFrame()
                    },
                }
            },
        }

        pdf_bytes = generate_benchmark_pdf(report_data)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"Audit Report exported to: {output_path}")


# Convenience alias
TabuLLM_Lib = TabuLLM
