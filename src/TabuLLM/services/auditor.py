import pandas as pd
import numpy as np
from utils.logging_config import logger

class DatasetAuditor:
    """Service for calculating structural, fairness, and privacy metrics of a dataset."""

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Excludes all is_ai_* columns, _row_id, ai_audit, is_context, and is_target from the audit process."""
        cols_to_drop = [c for c in df.columns if c.startswith('is_ai_') or c in ['_row_id', 'ai_audit', 'is_context', 'is_target']]
        return df.drop(columns=cols_to_drop, errors='ignore')

    def get_structural_metrics(self, df: pd.DataFrame) -> dict:
        """Calculates skewed distributions and integrity profiles."""
        df_clean = self._filter_data(df)
        if df_clean.empty: return {}
        
        metrics = {}
        # 1. Sparsity
        missing = df_clean.isnull().mean()
        metrics['sparsity'] = missing.to_dict()
        metrics['overall_sparsity'] = float(missing.mean())
        
        # 2. Skewness (Numeric Asymmetry)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        metrics['skewness'] = {col: float(df_clean[col].skew()) for col in num_cols if not df_clean[col].isnull().all()}
        
        # 3. Gini Stability (Categorical Dominance)
        stability = {}
        for col in df_clean.select_dtypes(include=['object', 'category']).columns:
            probs = df_clean[col].value_counts(normalize=True)
            gini_impurity = 1 - np.sum(probs**2)
            stability[col] = float(1 - gini_impurity)
        metrics['stability'] = stability
        
        return metrics

    def get_quality_metrics(self, df: pd.DataFrame, target_col: str = None) -> dict:
        """Analyzes dataset cleanliness: Duplicates, Conflicts, and Constant features."""
        df_clean = self._filter_data(df)
        if df_clean.empty: return {}
        
        # 1. Duplicate Records (Absolute)
        total_dupes = int(df_clean.duplicated().sum())
        
        # 2. Conflicting Labels (Noise)
        conflicts = 0
        if target_col and target_col in df_clean.columns:
            features = [c for c in df_clean.columns if c != target_col]
            if features:
                # Group by features and check if target is unique in each group
                conflicts = int(df_clean.groupby(features)[target_col].nunique().gt(1).sum())
        
        # 3. Constant Features (Zero Variance)
        constant_features = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
        
        return {
            "total_duplicates": total_dupes,
            "conflicting_labels": conflicts,
            "constant_features": constant_features,
            "quality_score": max(0, 100 - (total_dupes/len(df_clean)*50) - (conflicts*5)) # Penalty-based
        }

    def get_outlier_metrics(self, df: pd.DataFrame) -> dict:
        """Detects rows that may skew generation by being statistically extreme."""
        df_clean = self._filter_data(df)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        if not len(num_cols): return {"total_outliers": 0, "outlier_ratio": 0.0}
        
        # Simple Z-score logic
        z_scores = np.abs((df_clean[num_cols] - df_clean[num_cols].mean()) / (df_clean[num_cols].std() + 1e-9))
        outliers = (z_scores > 3).any(axis=1)
        count = int(outliers.sum())
        
        return {
            "total_outliers": count,
            "outlier_ratio": float(count / len(df_clean)),
            "outlier_rows": df_clean.index[outliers].tolist()[:5]
        }

    def get_cardinality_metrics(self, df: pd.DataFrame) -> dict:
        """Flags features with too many unique values for LLM processing."""
        df_clean = self._filter_data(df)
        risks = {}
        for col in df_clean.columns:
            n_unique = df_clean[col].nunique()
            if n_unique > 50:
                risks[col] = {
                    "count": int(n_unique),
                    "risk": "High" if n_unique > 200 else "Medium"
                }
        return risks

    def get_fairness_metrics(self, df: pd.DataFrame, target_col: str, sensitive_cols: list) -> dict:
        """
        Calculates Disparate Impact and Statistical Parity Difference (Excludes metadata).
        Standard threshold for DI is 0.8 (Four-Fifths Rule).
        """
        df_clean = self._filter_data(df)
        # Remove target_col if it was an is_ai column (unlikely but safe)
        if target_col not in df_clean.columns or df_clean.empty or not sensitive_cols:
            return {}

        results = {}
        
        if pd.api.types.is_numeric_dtype(df_clean[target_col]):
            median = df_clean[target_col].median()
            df_binary = (df_clean[target_col] >= median).astype(int)
        else:
            pos_label = df_clean[target_col].mode()[0]
            df_binary = (df_clean[target_col] == pos_label).astype(int)

        for s_col in sensitive_cols:
            # Re-verify that s_col is in current clean columns
            if s_col not in df_clean.columns or s_col == target_col: continue
            
            counts = df_clean[s_col].value_counts()
            if counts.empty: continue
            
            majority_class = counts.index[0]
            prob_maj = df_binary[df_clean[s_col] == majority_class].mean()
            prob_min = df_binary[df_clean[s_col] != majority_class].mean()
            
            di = prob_min / (prob_maj + 1e-9)
            spd = prob_maj - prob_min
            
            results[s_col] = {
                "majority_group": str(majority_class),
                "disparate_impact": float(di),
                "parity_gap": float(spd),
                "bias_detected": di < 0.8 or di > 1.25
            }
            
        return results

    def get_proxy_risks(self, df: pd.DataFrame, sensitive_cols: list) -> dict:
        """Identifies columns that act as proxies for sensitive attributes (0.30+ correlation)."""
        df_clean = self._filter_data(df)
        if df_clean.empty or not sensitive_cols: return {}
        
        risks = {}
        # Lower threshold for 'High Awareness' audit: 0.30
        df_encoded = df_clean.apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)
        
        for s_col in sensitive_cols:
            if s_col not in df_encoded.columns: continue
            corrs = df_encoded.corr()[s_col].abs().sort_values(ascending=False)
            potential_proxies = corrs[(corrs > 0.30) & (corrs < 1.0)].to_dict()
            if potential_proxies:
                risks[s_col] = potential_proxies
                
        return risks

    def get_privacy_metrics(self, df: pd.DataFrame) -> dict:
        """Calculates uniqueness profiles including numeric QIDs (Excludes metadata)."""
        df_clean = self._filter_data(df)
        if df_clean.empty: return {}
        
        # 1. Identify Quasi-Identifiers (All non-text columns with manageable cardinality)
        qid_cols = []
        for col in df_clean.columns:
            n_unique = df_clean[col].nunique()
            # If it's a unique ID (count == records) or too many categories, skip as QID
            if n_unique > 1 and n_unique < (len(df_clean) * 0.95):
                qid_cols.append(col)
        
        if not qid_cols: 
            return {"uniqueness_ratio": 0.0, "risk_level": "Low", "at_risk_count": 0}
        
        # 2. Calculate Uniqueness (Rows that are perfectly unique in these columns)
        is_unique_row = ~df_clean.duplicated(subset=qid_cols, keep=False)
        at_risk_count = int(is_unique_row.sum())
        ratio = at_risk_count / len(df_clean)
        
        # 3. Adopt Strict Risk Thresholds (High > 15%, Med > 5%)
        level = "High" if ratio > 0.15 else ("Medium" if ratio > 0.05 else "Low")
        
        return {
            "at_risk_count": at_risk_count,
            "uniqueness_ratio": float(ratio),
            "risk_level": level,
            "qid_subset": qid_cols[:5] # Sample of columns used for check
        }

    def get_quasi_identifiers(self, df: pd.DataFrame) -> list:
        """Detects specific columns that act as strong re-identification keys."""
        df_clean = self._filter_data(df)
        high_risk_keys = ['zip', 'addr', 'loc', 'lat', 'long', 'dob', 'birth', 'date', 'phone', 'mail', 'ip', 'device']
        
        qids = []
        for col in df_clean.columns:
            # Match by name
            if any(k in col.lower() for k in high_risk_keys):
                qids.append(col)
                continue
            
            # Match by entropy (high but not unique)
            n_unique = df_clean[col].nunique()
            if 50 < n_unique < (len(df_clean) * 0.9):
                qids.append(col)
                
        return sorted(list(set(qids)))

# Singleton instance
auditor = DatasetAuditor()
