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
    def infer_dataset_rules(df: pd.DataFrame, categorical_threshold: int = 20) -> List[Dict[str, Any]]:
        """
        Analyzes a DataFrame and returns a list of rule metadata.
        Each rule has a 'level' attribute (1 or 2).
        """
        inferred_rules = []
        
        if df.empty:
            return inferred_rules

        # Filter out internal tracking columns
        excluded_cols = ['_row_id', 'is_context', 'is_target', 'ai_audit']
        cols_to_analyze = [c for c in df.columns if c not in excluded_cols and not c.startswith('is_ai_')]

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
                # Numeric Ranges
                valid_data = df[col].dropna()
                if not valid_data.empty:
                    c_min, c_max = valid_data.min(), valid_data.max()
                    # Handle float/int differences for clean rules
                    if float(c_min).is_integer() and float(c_max).is_integer():
                        c_min, c_max = int(c_min), int(c_max)
                    
                    inferred_rules.append({
                        "name": col,
                        "type": "number",
                        "level": 2, # Boundary
                        "min": c_min,
                        "max": c_max,
                        "message": f"Value for '{col}' should be between {c_min} and {c_max}."
                    })
            
            else:
                # Categorical Domains
                unique_vals = df[col].dropna().unique()
                if 0 < len(unique_vals) <= categorical_threshold:
                    # Sort for consistent prompting
                    try:
                        unique_list = sorted([str(v) for v in unique_vals])
                    except:
                        unique_list = [str(v) for v in unique_vals]
                        
                    inferred_rules.append({
                        "name": col,
                        "type": "category",
                        "level": 2, # Boundary
                        "text": ", ".join(unique_list),
                        "message": f"Value for '{col}' should be one of: {', '.join(unique_list)}."
                    })

        logger.info(f"Rule Inference: Detected {len([r for r in inferred_rules if r['level'] == 1])} integrity rules and {len([r for r in inferred_rules if r['level'] == 2])} boundary rules.")
        return inferred_rules
