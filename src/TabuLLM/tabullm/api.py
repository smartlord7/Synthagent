import pandas as pd
import sys
import os

# Ensure the root services and utils are accessible to the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.data_manager import parse_uploaded_file, df_to_aggrid_format
from services.llm_client import LocalOpenAIProvider, GeminiProvider
from services.pipeline_manager import run_single_dataset_benchmark, PipelineManager
from services.report_generator import generate_benchmark_pdf
from services.auditor import DatasetAuditor
from services.prompt_manager import prompt_manager

class TabuLLM:
    """
    The primary developer interface for TabuLLM.
    This class encapsulates data loading, AI-driven insights, 
    synthetic data generation, and audit reporting.
    """
    
    def __init__(self, provider="local", model="google/gemma-3-4b", endpoint_url=None):
        """
        Initialize the TabuLLM engine.
        
        Args:
            provider (str): LLM provider to use ('local' or 'gemini'). Default 'local'.
            model (str): Model ID to use. Default 'google/gemma-3-4b'.
            endpoint_url (str, optional): Custom endpoint for local LLM (e.g., LM Studio).
        """
        if provider == "local":
            self.llm = LocalOpenAIProvider(
                endpoint_url=endpoint_url or os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions"),
                model_name=model
            )
        elif provider == "gemini":
            self.llm = GeminiProvider(model_name=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.default_model = model

    def get_models(self) -> list[dict]:
        """
        Queries the backend for a list of available AI models and their context windows.
        
        Returns:
            list[dict]: List of model objects with 'id' and 'max_context'.
        """
        return self.llm.get_available_models()

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

    def suggest_targets(self, df: pd.DataFrame, single_target: bool = True) -> list[str]:
        """
        Uses AI to automatically suggest potential target columns for ML.
        
        Args:
            df (pd.DataFrame): The dataset to analyze.
            single_target (bool): If True, prioritized a single primary target.
            
        Returns:
            list[str]: A list of suggested target column names.
        """
        return self.llm.detect_target_columns(df.head(50), single_target=single_target, algorithm=self.default_model)

    def get_insights(self, df: pd.DataFrame, metadata: str = "") -> str:
        """
        Generates a structured natural language description of the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze.
            metadata (str): Optional user-provided context or hints.
            
        Returns:
            str: AI-generated markdown insights.
        """
        return self.llm.get_dataset_description(df.head(50), provided_metadata=metadata, algorithm=self.default_model)

    def augment(self, df: pd.DataFrame, targets: list[str], num_rows: int = 100, folds: int = 3, strategy: str = "random", ml_models: list[str] = ["rf"], na_strategy: str = "impute") -> dict:
        """
        Runs the full High-Reliability Augmentation pipeline.
        This includes cross-validation, fidelity auditing, and ML performance comparison.
        
        Args:
            df (pd.DataFrame): Original dataset.
            targets (list[str]): Target column(s) for ML evaluation.
            num_rows (int): Total number of synthetic rows to generate per fold.
            folds (int): Number of cross-validation folds. Default 3.
            strategy (str): Context selection strategy ('random', 'dense', 'diverse').
            ml_models (list[str]): List of models to evaluate ('rf', 'lr', 'tree', 'nb', 'xgb').
            na_strategy (str): Strategy for missing values ('impute' or 'drop').
            
        Returns:
            dict: The results dictionary including augmented_data, fidelity scores, and benchmarks.
        """
        # Filter internal cols
        df_clean = df.copy()
        if '_row_id' in df_clean.columns:
            df_clean.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')

        col_defs = [{"field": c, "headerName": c} for c in df_clean.columns]
        config = {
            "folds": folds,
            "strategy": strategy,
            "algorithm": self.default_model,
            "targets": targets,
            "num_empty": num_rows,
            "ctx_ratio": 20, 
            "ml_config": {"selected_models": ml_models, "params": {}},
            "na_strategy": na_strategy,
            "instructions": "",
            "desc_metadata": ""
        }
        
        task = ("lib-task", "dataset.csv", df_clean.to_dict('records'), col_defs, config)
        
        results = PipelineManager.run_pipeline([task])
        return results[0] if results else {}

    def run_audit(self, df: pd.DataFrame, targets: list[str] = None) -> dict:
        """
        Runs a comprehensive Health & Ethics audit on the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to audit.
            targets (list[str]): Target columns for fairness parity analysis.
            
        Returns:
            dict: Audit results including Structural, Fairness, Privacy, and Quality metrics.
        """
        auditor_service = DatasetAuditor()
        target_col = targets[0] if targets else None
        
        sensitive_keywords = ['gender', 'sex', 'age', 'race', 'ethnicity', 'nationality', 'religion', 'zip', 'income', 'stage', 'location', 'origin', 'education', 'marital', 'disability']
        sensitive_cols = [c for c in df.columns if any(k in c.lower() for k in sensitive_keywords) and c not in ['is_context', 'is_target']]

        return {
            "structural": auditor_service.get_structural_metrics(df),
            "outliers": auditor_service.get_outlier_metrics(df),
            "fairness": auditor_service.get_fairness_metrics(df, target_col, sensitive_cols),
            "privacy": auditor_service.get_privacy_metrics(df),
            "proxies": auditor_service.get_proxy_risks(df, sensitive_cols),
            "quality": auditor_service.get_quality_metrics(df, target_col),
            "quasi_identifiers": auditor_service.get_quasi_identifiers(df)
        }

    def ponder_strategy(self, df: pd.DataFrame, targets: list[str] = None) -> str:
        """
        AI Strategy: Performs a deep ethical analysis of the audit results and 
        generates a prioritized augmentation plan.
        
        Args:
            df (pd.DataFrame): The dataset to analyze.
            targets (list[str]): Target columns for bias context.
            
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
            "outliers": audit["outliers"]
        }
        
        prompt = prompt_manager.get_prompt("ponder_audit.j2", **context)
        return self.llm.call_llm(messages=[{"role": "user", "content": prompt}], model_id=self.default_model, temperature=0.3)

    def generate_pdf_report(self, pipeline_results: dict, output_path: str):
        """
        Exports the benchmark results and fidelity audits to a professional PDF.
        
        Args:
            pipeline_results (dict): The dictionary returned by the .augment() method.
            output_path (str): File path for the resulting PDF.
        """
        report_data = {
            "executive_summary": {
                "avg_fidelity": pipeline_results.get("avg_fidelity", 0),
                "global_metrics": {}
            },
            "datasets": {
                pipeline_results.get("dataset_id", "lib-ds"): {
                    "filename": pipeline_results.get("filename", "dataset.csv"),
                    "metadata": {"Rows": 0, "Cols": 0}, # Placeholder or extract from df
                    "params": {"generation": {"Algorithm": self.default_model}},
                    "metrics": {"Fidelity": pipeline_results.get("avg_fidelity", 0)},
                    "ml_results": pipeline_results.get("overall_ml", {})
                }
            }
        }
        
        pdf_bytes = generate_benchmark_pdf(report_data)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"Audit Report exported to: {output_path}")

# Convenience Alias
TabuLLM_Lib = TabuLLM
