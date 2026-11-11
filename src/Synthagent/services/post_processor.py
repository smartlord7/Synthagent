
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from utils.logging_config import logger

class PostProcessor:
    """
    Service to apply active transformations (noise, rounding, balancing) 
    to synthetic data based on extracted rules.
    """

    @staticmethod
    def apply_active_rules(data: List[Dict[str, Any]], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data or not rules:
            return data
        
        df = pd.DataFrame(data)
        
        for rule in rules:
            rule_name = rule.get("name", "").lower()
            rule_value = rule.get("value", "")
            
            # 1. Noise Injection (Privacy Safeguard)
            if "noise" in rule_name:
                # Expecting value like "10%" or just "10"
                try:
                    noise_pct = 0.1 # Default
                    if "%" in str(rule_value):
                        noise_pct = float(rule_value.replace("%", "")) / 100.0
                    elif str(rule_value).replace(".", "").isdigit():
                        noise_pct = float(rule_value) / 100.0 if float(rule_value) > 1 else float(rule_value)
                    
                    # Apply to specific column if named, else to QIDs if name is generic
                    target_cols = []
                    # Check if rule name starts with a column name
                    potential_col = rule_name.split("_noise")[0].strip()
                    if potential_col in df.columns:
                        target_cols = [potential_col]
                    else:
                        # Fallback: find common QID names if it's a general 'quasi_identifier_noise' rule
                        target_cols = [c for c in df.columns if any(k in c.lower() for k in ['zip', 'id', 'addr', 'loc', 'phone', 'mail'])]
                    
                    for col in target_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            logger.info(f"[PostProcess] Adding {noise_pct*100}% Gaussian noise to {col}")
                            std = df[col].std()
                            noise = np.random.normal(0, std * noise_pct, size=len(df))
                            df[col] = df[col] + noise
                        else:
                            # For categorical, "noise" means random swapping/shuffling a percentage of values
                            logger.info(f"[PostProcess] Applying {noise_pct*100}% swap noise to categorical {col}")
                            mask = np.random.rand(len(df)) < noise_pct
                            shuffled = df[col].sample(frac=1).values
                            df.loc[mask, col] = shuffled[mask]
                except Exception as e:
                    logger.warning(f"Failed to apply noise rule {rule_name}: {e}")

            # 2. Rounding / Precision
            elif "round" in rule_name or "precision" in rule_name:
                try:
                    # Extract decimal places
                    decimals = 0
                    import re
                    match = re.search(r'\d+', str(rule_value))
                    if match:
                        decimals = int(match.group())
                    
                    potential_col = rule_name.split("_round")[0].split("_precision")[0].strip()
                    if potential_col in df.columns and pd.api.types.is_numeric_dtype(df[potential_col]):
                        logger.info(f"[PostProcess] Rounding {potential_col} to {decimals} decimals")
                        df[potential_col] = df[potential_col].round(decimals)
                except: pass

            # 3. Specific Normalization
            elif "normalize" in rule_name:
                try:
                    potential_col = rule_name.split("_normalize")[0].strip()
                    if potential_col in df.columns and pd.api.types.is_numeric_dtype(df[potential_col]):
                        logger.info(f"[PostProcess] Min-Max Normalizing {potential_col}")
                        c_min = df[potential_col].min()
                        c_max = df[potential_col].max()
                        if c_max > c_min:
                            df[potential_col] = (df[potential_col] - c_min) / (c_max - c_min)
                except: pass

        return df.to_dict("records")
