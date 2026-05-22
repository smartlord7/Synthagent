import pandas as pd
import numpy as np
from typing import List, Dict, Any
from utils.logging_config import logger

class RuleInference:
    """
    Service to automatically infer hierarchical rules from a DataFrame.

    Level 1: Core Integrity (Datatypes)
    Level 2: Dataset Boundaries (Ranges and Categories)
    """

    @staticmethod
    def infer_dataset_rules(
        df: pd.DataFrame,
        categorical_threshold: int = 20,
        boundary_tolerance: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a DataFrame and returns a list of rule metadata.
        Each rule has a 'level' attribute (1 or 2).

        Args:
            df: Source dataframe.
            categorical_threshold: Max unique values for a column to be treated as
                categorical (otherwise it gets no enum constraint).
            boundary_tolerance: Fraction by which numeric min/max are relaxed beyond
                the observed data extremes (default 10 %).  This prevents spurious
                violations when the LLM generates values that are statistically
                plausible but lie just outside the training-set extremes.
        """
        inferred_rules = []

        # Filter out internal tracking columns
        excluded_cols = ['_row_id', 'is_context', 'is_target', 'ai_audit']
        cols_to_analyze = [c for c in df.columns if c not in excluded_cols and not c.startswith('is_ai_')]

        if not cols_to_analyze:
            return inferred_rules

        for col in cols_to_analyze:
            dtype = df[col].dtype

            # --- Level 1: Core Integrity (Datatype) ---
            if pd.api.types.is_numeric_dtype(dtype):
                base_type = "number"
            else:
                base_type = "category"

            inferred_rules.append({
                "name": col,
                "type": base_type,
                "level": 1,  # Integrity
                "message": f"Column '{col}' must be of type {base_type}."
            })

            # --- Level 2: Dataset Boundaries ---
            if base_type == "number":
                valid_data = df[col].dropna()
                if not valid_data.empty:
                    c_min_raw = valid_data.min()
                    c_max_raw = valid_data.max()

                    # Relax boundaries by `boundary_tolerance` so the LLM is not
                    # penalised for generating values that are statistically plausible
                    # but lie just outside the observed training extremes.
                    spread = c_max_raw - c_min_raw
                    # Guard against zero-spread columns (e.g. a constant column).
                    buffer = max(spread * boundary_tolerance, abs(c_min_raw) * boundary_tolerance, 1.0)
                    c_min = c_min_raw - buffer
                    c_max = c_max_raw + buffer

                    # Keep integers as integers for clean rule messages
                    if float(c_min_raw).is_integer() and float(c_max_raw).is_integer():
                        c_min = int(round(c_min))
                        c_max = int(round(c_max))

                    inferred_rules.append({
                        "name": col,
                        "type": "number",
                        "level": 2,  # Boundary
                        "min": c_min,
                        "max": c_max,
                        "message": f"Value for '{col}' should be between {c_min} and {c_max}."
                    })

            else:
                # Categorical Domains
                unique_vals = df[col].dropna().unique()
                if 0 < len(unique_vals) <= categorical_threshold:
                    try:
                        unique_list = sorted([str(v) for v in unique_vals])
                    except Exception:
                        unique_list = [str(v) for v in unique_vals]

                    inferred_rules.append({
                        "name": col,
                        "type": "category",
                        "level": 2,  # Boundary
                        "text": ", ".join(unique_list),
                        "message": f"Value for '{col}' should be one of: {', '.join(unique_list)}."
                    })

        n_integrity  = len([r for r in inferred_rules if r['level'] == 1])
        n_boundaries = len([r for r in inferred_rules if r['level'] == 2])
        logger.info(
            f"Rule Inference: {n_integrity} integrity rules, "
            f"{n_boundaries} boundary rules "
            f"(numeric tolerance ±{boundary_tolerance*100:.0f}%)."
        )
        return inferred_rules
