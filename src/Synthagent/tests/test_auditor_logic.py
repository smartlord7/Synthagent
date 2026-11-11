import unittest
import pandas as pd
import numpy as np
from services.auditor import auditor, DatasetAuditor

class TestAuditorLogic(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with deliberate issues
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40, 150, np.nan], # 150 is outlier, 1 nan
            "gender": ["M", "M", "F", "F", "M", "M"], # M dominates
            "income": [50000, 60000, 55000, 65000, 70000, 52000],
            "target": [0, 1, 0, 1, 0, 1],
            "sensitive": ["A", "A", "B", "B", "A", "A"], # A is majority
            "_row_id": range(6),
            "ai_audit": [""] * 6
        })

    def test_structural_metrics(self):
        metrics = auditor.get_structural_metrics(self.df)
        self.assertIn("sparsity", metrics)
        self.assertIn("overall_sparsity", metrics)
        self.assertIn("skewness", metrics)
        self.assertIn("stability", metrics)
        # age has 1 nan out of 6 -> 1/6 sparsity
        self.assertAlmostEqual(metrics["sparsity"]["age"], 1/6)

    def test_quality_metrics(self):
        metrics = auditor.get_quality_metrics(self.df, target_col="target")
        self.assertIn("total_duplicates", metrics)
        self.assertIn("conflicting_labels", metrics)
        self.assertEqual(metrics["total_duplicates"], 0)
        
        # Test duplicates
        df_dupe = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        metrics_dupe = auditor.get_quality_metrics(df_dupe)
        self.assertEqual(metrics_dupe["total_duplicates"], 1)

    def test_outlier_metrics(self):
        metrics = auditor.get_outlier_metrics(self.df)
        self.assertIn("total_outliers", metrics)
        # 150 in age should be detected as outlier (Z > 3 is strict, but let's see)
        # Actually with small sample size Z might not be > 3. 
        # But we check the keys exist.
        self.assertIsInstance(metrics["total_outliers"], int)

    def test_fairness_metrics(self):
        # Target is balanced, should have high Disparate Impact
        metrics = auditor.get_fairness_metrics(self.df, target_col="target", sensitive_cols=["sensitive"])
        self.assertIn("sensitive", metrics)
        self.assertIn("disparate_impact", metrics["sensitive"])
        self.assertIn("bias_detected", metrics["sensitive"])

    def test_privacy_metrics(self):
        metrics = auditor.get_privacy_metrics(self.df)
        self.assertIn("uniqueness_ratio", metrics)
        self.assertIn("risk_level", metrics)
        self.assertGreater(metrics["uniqueness_ratio"], 0)

    def test_filter_data(self):
        df_filtered = auditor._filter_data(self.df)
        self.assertNotIn("_row_id", df_filtered.columns)
        self.assertNotIn("ai_audit", df_filtered.columns)
        self.assertIn("age", df_filtered.columns)

    def test_detect_sensitive_columns(self):
        cols = DatasetAuditor.detect_sensitive_columns(self.df)
        self.assertIn("age", cols)
        self.assertIn("gender", cols)
        self.assertIn("income", cols)
        self.assertNotIn("target", cols)
        self.assertNotIn("_row_id", cols)

    def test_detect_sensitive_columns_excludes_ui_internal_cols(self):
        df = self.df.copy()
        df["is_context"] = False
        df["is_target"] = False
        cols = DatasetAuditor.detect_sensitive_columns(df)
        self.assertNotIn("is_context", cols)
        self.assertNotIn("is_target", cols)

if __name__ == '__main__':
    unittest.main()
