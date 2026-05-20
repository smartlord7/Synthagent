import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatasetAuditor:
    """Modular engine for comprehensive dataset health and ethics auditing."""
    
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Excludes all is_ai_* columns and internal IDs from the audit."""
        cols_to_drop = [c for c in df.columns if c.startswith('is_ai_') or c in ['_row_id', 'ai_audit', 'is_context', 'is_target']]
        return df.drop(columns=cols_to_drop, errors='ignore')

    def get_audit_report(self, df: pd.DataFrame, target_col: str = None, sensitive_cols: List[str] = None) -> Dict[str, Any]:
        """Entry point for all audit metrics."""
        df_clean = self._filter_data(df)
        if df_clean.empty: 
            return {
                "structural": {},
                "quality": {},
                "fairness": {"status": "No data available after filtering."},
                "privacy": {}
            }
        
        return {
            "structural": self.get_structural_metrics(df),
            "quality": self.get_quality_metrics(df, target_col),
            "fairness": self.get_fairness_metrics(df, target_col, sensitive_cols or []),
            "privacy": self.get_privacy_metrics(df)
        }

    def get_structural_metrics(self, df: pd.DataFrame) -> dict:
        """Exposes _audit_structure with legacy keys for backward compatibility."""
        df_clean = self._filter_data(df)
        if df_clean.empty: 
            return {
                "sparsity": {}, "overall_sparsity": 0, "skewness": {}, "stability": 0,
                "row_count": 0, "col_count": 0, "total_cells": 0, "empty_rows": 0,
                "density_ratio": 0, "sparsity_ratio": 0, "memory_readable": "0 MB",
                "type_distribution": {"numeric": 0, "categorical": 0, "object": 0},
                "avg_cardinality": 0, "constant_columns": []
            }
        s = self._audit_structure(df_clean)
        
        # Calculate skewness and filter out NaNs (e.g. for all-NaN columns)
        skew_vals = {}
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            val = df_clean[col].skew()
            if not pd.isna(val):
                skew_vals[col] = float(val)
                
        return {
            "sparsity": {col: float(df_clean[col].isnull().mean()) for col in df_clean.columns},
            "overall_sparsity": s.get("sparsity_ratio", 0),
            "skewness": skew_vals,
            "stability": 1.0, 
            **s
        }

    def get_quality_metrics(self, df: pd.DataFrame, target_col: str = None) -> dict:
        """Exposes _audit_quality with legacy keys for backward compatibility."""
        df_clean = self._filter_data(df)
        if df_clean.empty: 
            return {
                "quality_score": 0, "duplicate_count": 0, "duplicate_ratio": 0,
                "outlier_count": 0, "outlier_ratio": 0, "avg_skewness": 0,
                "avg_kurtosis": 0, "conflicting_labels": 0, "columns_with_missing": 0,
                "total_duplicates": 0
            }
        q = self._audit_quality(df_clean, target_col)
        return {
            "total_duplicates": q.get("duplicate_count", 0),
            "conflicting_labels": q.get("conflicting_labels", 0),
            **q
        }

    def get_outlier_metrics(self, df: pd.DataFrame) -> dict:
        """Calculates outliers specifically (Legacy compatibility)."""
        df_clean = self._filter_data(df)
        if df_clean.empty: return {"total_outliers": 0, "outlier_ratio": 0.0}
        q = self._audit_quality(df_clean, None)
        return {
            "total_outliers": q.get("outlier_count", 0),
            "outlier_ratio": q.get("outlier_ratio", 0.0)
        }

    def get_fairness_metrics(self, df: pd.DataFrame, target_col: str = None, sensitive_cols: List[str] = None) -> dict:
        """Exposes _audit_fairness with legacy keys for backward compatibility."""
        df_clean = self._filter_data(df)
        if df_clean.empty: 
            return {
                "status": "No data available.",
                "metrics_per_attr": {},
                "detected_bias_count": 0,
                "sensitive_proxies": {},
                "group_sparsity": {}
            }
        f = self._audit_fairness(df_clean, target_col, sensitive_cols or [])
        # Map metrics_per_attr items to top-level for legacy tests
        legacy_f = dict(f.get("metrics_per_attr", {}))
        for col_name, metrics in legacy_f.items():
            metrics["bias_detected"] = metrics.get("disparate_impact", 1.0) < 0.8 or metrics.get("disparate_impact", 1.0) > 1.25
        return {**f, **legacy_f}

    def get_privacy_metrics(self, df: pd.DataFrame) -> dict:
        """Exposes _audit_privacy with legacy keys for backward compatibility."""
        df_clean = self._filter_data(df)
        if df_clean.empty: 
            return {
                "uniqueness_ratio": 0, "at_risk_record_count": 0, "identity_risk_level": "N/A",
                "attr_identifiability": {}, "avg_k_anonymity": 0, "avg_column_entropy": 0,
                "max_column_entropy": 0, "known_sensitive_patterns": [], "risk_level": "N/A"
            }
        p = self._audit_privacy(df_clean)
        return {
            "risk_level": p.get("identity_risk_level", "Low"),
            **p
        }

    def _audit_structure(self, df: pd.DataFrame) -> dict:
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        
        # Memory profiling
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 * 1024)
        
        # Type distribution
        type_summary = {
            "numeric": int(sum(1 for t in df.dtypes if pd.api.types.is_numeric_dtype(t))),
            "categorical": int(sum(1 for t in df.dtypes if not pd.api.types.is_numeric_dtype(t))),
            "object": int(sum(1 for t in df.dtypes if t == 'object'))
        }

        # Dynamic cardinality
        cardinality = {col: int(df[col].nunique()) for col in df.columns}
        avg_card = np.mean(list(cardinality.values())) if cardinality else 0

        return {
            "row_count": len(df),
            "col_count": len(df.columns),
            "total_cells": int(total_cells),
            "empty_rows": int((df.isnull().all(axis=1)).sum()),
            "density_ratio": float((total_cells - null_cells) / total_cells if total_cells > 0 else 0),
            "sparsity_ratio": float(null_cells / total_cells if total_cells > 0 else 0),
            "memory_readable": f"{mem_mb:.2f} MB",
            "type_distribution": type_summary,
            "avg_cardinality": float(avg_card),
            "constant_columns": [c for c, v in cardinality.items() if v <= 1]
        }

    def _audit_quality(self, df: pd.DataFrame, target_col: str) -> dict:
        dup_count = df.duplicated().sum()
        
        # Outlier detection (Z-score > 3 for numeric)
        numerics = df.select_dtypes(include=[np.number])
        outlier_mask = pd.Series(False, index=df.index)
        if not numerics.empty:
            std = numerics.std() + 1e-9
            z_scores = np.abs((numerics - numerics.mean()) / std)
            outlier_mask = (z_scores > 3).any(axis=1)
        
        # Target conflict
        conflicts = 0
        if target_col and target_col in df.columns:
            feat_cols = [c for c in df.columns if c != target_col]
            if feat_cols:
                conflicts = int(df.groupby(feat_cols)[target_col].nunique().gt(1).sum())

        skew = numerics.skew().mean() if not numerics.empty else 0
        kurt = numerics.kurt().mean() if not numerics.empty else 0

        return {
            "quality_score": float(max(0, 100 - (dup_count/len(df)*20 + outlier_mask.sum()/len(df)*20 + df.isnull().mean().mean()*50))),
            "duplicate_count": int(dup_count),
            "duplicate_ratio": float(dup_count / len(df) if len(df) > 0 else 0),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_ratio": float(outlier_mask.mean()),
            "avg_skewness": float(skew),
            "avg_kurtosis": float(kurt),
            "conflicting_labels": int(conflicts),
            "columns_with_missing": int((df.isnull().sum() > 0).sum())
        }

    def _audit_fairness(self, df: pd.DataFrame, target_col: str, sensitive_cols: List[str]) -> dict:
        if not target_col or not sensitive_cols:
            return {
                "status": "Target or sensitive features not selected.",
                "metrics_per_attr": {},
                "detected_bias_count": 0,
                "sensitive_proxies": {},
                "group_sparsity": {}
            }
        
        metrics = {
            "metrics_per_attr": {},
            "detected_bias_count": 0,
            "sensitive_proxies": self.get_proxy_risks(df, sensitive_cols),
            "group_sparsity": {} 
        }
        
        # Pre-calculate binary target proxy
        if pd.api.types.is_numeric_dtype(df[target_col]):
            median = df[target_col].median()
            df_binary = (df[target_col] >= median).astype(int)
        else:
            pos_label = df[target_col].mode()[0]
            df_binary = (df[target_col] == pos_label).astype(int)

        for col in sensitive_cols:
            if col not in df.columns: continue
            groups = df[col].unique()
            if len(groups) < 2: continue
            
            counts = df[col].value_counts()
            maj = counts.idxmax()
            min_grp = counts.idxmin()
            
            maj_success = df_binary[df[col] == maj].mean()
            min_success = df_binary[df[col] == min_grp].mean()
            di = min_success / (maj_success + 1e-9)
            
            # Distributional Shift proxy
            shifts = {}
            numerics = df.select_dtypes(include=[np.number]).columns
            for n_col in numerics:
                if n_col == col or n_col == target_col: continue
                m_mean = df[df[col] == maj][n_col].mean()
                mi_mean = df[df[col] == min_grp][n_col].mean()
                denom = df[n_col].std() + 1e-9
                shifts[n_col] = float(abs(m_mean - mi_mean) / denom)

            metrics["group_sparsity"][str(maj)] = float(df[df[col] == maj].isnull().mean().mean())
            metrics["group_sparsity"][str(min_grp)] = float(df[df[col] == min_grp].isnull().mean().mean())

            metrics["metrics_per_attr"][col] = {
                "majority_group": str(maj),
                "minority_group": str(min_grp),
                "disparate_impact": float(di),
                "parity_gap": float(abs(maj_success - min_success)),
                "representation_ratio": float(len(df[df[col] == min_grp]) / len(df)),
                "distribution_shift": shifts 
            }
            if di < 0.8 or di > 1.25:
                metrics["detected_bias_count"] += 1
                
        return metrics

    def _audit_privacy(self, df: pd.DataFrame) -> dict:
        total = len(df)
        # In privacy context, 'uniques' or 'at risk' typically refers to rows that appear only once
        at_risk_mask = ~df.duplicated(keep=False)
        uniques_count = at_risk_mask.sum()
        
        attr_metrics = {}
        for col in df.columns:
            try:
                pk = df[col].value_counts(normalize=True)
                entropy = -(pk * np.log2(pk + 1e-10)).sum()
                u_ratio = df[col].nunique() / total
                attr_metrics[col] = {
                    "entropy": float(entropy),
                    "uniqueness_signal": float(u_ratio),
                    "identifiability": "High" if u_ratio > 0.7 else ("Medium" if u_ratio > 0.3 else "Low")
                }
            except: continue

        qids = self.get_quasi_identifiers(df)
        k_score = 0
        if qids:
            try:
                k_score = float(df.groupby(qids).size().mean())
            except: pass

        return {
            "uniqueness_ratio": float(uniques_count / total if total > 0 else 0),
            "at_risk_record_count": int(uniques_count),
            "identity_risk_level": "High" if uniques_count/total > 0.4 else ("Medium" if uniques_count/total > 0.1 else "Low"),
            "attr_identifiability": attr_metrics,
            "avg_k_anonymity": k_score,
            "avg_column_entropy": float(np.mean([m['entropy'] for m in attr_metrics.values()])) if attr_metrics else 0,
            "max_column_entropy": float(np.max([m['entropy'] for m in attr_metrics.values()])) if attr_metrics else 0,
            "known_sensitive_patterns": qids
        }

    def get_proxy_risks(self, df: pd.DataFrame, sensitive_cols: List[str]) -> dict:
        risks = {}
        if not sensitive_cols: return risks
        df_s = df.sample(min(len(df), 2000))
        df_encoded = df_s.apply(lambda x: pd.factorize(x)[0] if not pd.api.types.is_numeric_dtype(x) else x)
        for s_col in sensitive_cols:
            if s_col not in df_encoded.columns: continue
            corrs = df_encoded.corr()[s_col].abs().sort_values(ascending=False)
            potentials = corrs[(corrs > 0.3) & (corrs < 1.0)].to_dict()
            if potentials: risks[s_col] = potentials
        return risks

    def get_quasi_identifiers(self, df: pd.DataFrame) -> list:
        qids = []
        for col in df.columns:
            if any(k in col.lower() for k in ['zip', 'id', 'addr', 'loc', 'phone', 'mail']):
                qids.append(col)
            elif 2 < df[col].nunique() < (len(df) * 0.8):
                qids.append(col)
        return sorted(list(set(qids)))

auditor = DatasetAuditor()
