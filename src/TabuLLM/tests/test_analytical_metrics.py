import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metrics import (
    compute_jensenshannon_fidelity,
    compute_z_drift,
    compute_correlation_stability,
    compute_privacy_metrics,
    compute_hellinger_fidelity,
    calculate_metrics
)

class TestAnalyticalMetrics(unittest.TestCase):
    
    def setUp(self):
        # Create a simple original dataset
        self.df_orig = pd.DataFrame({
            "num1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "cat1": ["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"]
        })
        # Create a "good" augmented dataset (similar distributions)
        self.df_aug_good = pd.DataFrame({
            "num1": [11, 21, 31, 41, 51, 61, 71, 81, 91, 101],
            "cat1": ["A", "A", "B", "B", "B", "C", "C", "C", "C", "A"]
        })
        # Create a "bad" augmented dataset (very different distributions)
        self.df_aug_bad = pd.DataFrame({
            "num1": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            "cat1": ["Z", "Z", "Z", "Z", "Z", "Z", "Z", "Z", "Z", "Z"]
        })

    def test_jensenshannon_fidelity(self):
        """Verify JSD fidelity score (1.0 = identical)."""
        score_good = compute_jensenshannon_fidelity(self.df_orig["num1"], self.df_aug_good["num1"])
        score_bad = compute_jensenshannon_fidelity(self.df_orig["num1"], self.df_aug_bad["num1"])
        
        self.assertGreater(score_good, 0.7)
        self.assertLess(score_bad, 0.2)

    def test_z_drift(self):
        """Verify Z-drift (standardized mean shift)."""
        drift_good = compute_z_drift(self.df_orig["num1"], self.df_aug_good["num1"])
        drift_bad = compute_z_drift(self.df_orig["num1"], self.df_aug_bad["num1"])
        
        # Good drift should be small (< 0.1 std dev)
        self.assertLess(drift_good, 0.1)
        # Bad drift should be massive
        self.assertGreater(drift_bad, 10.0)

    def test_correlation_stability(self):
        """Verify correlation matrix similarity."""
        df_orig = pd.DataFrame({"a": [1,2,3,4,5], "b": [2,4,6,8,10]}) # Perfect correlation
        df_aug = pd.DataFrame({"a": [1,2,3,4,5], "b": [2,4,6,8,11]})  # Near perfect
        
        score = compute_correlation_stability(df_orig, df_aug, ["a", "b"])
        self.assertGreater(score, 95.0)

    def test_privacy_memorization_risk(self):
        """Verify that identical rows trigger high privacy risk."""
        df_orig = pd.DataFrame({"f1": [1, 2, 3], "f2": [10, 20, 30]})
        # Data that is exactly the same as original
        df_aug_leak = pd.DataFrame({"f1": [1, 2, 3], "f2": [10, 20, 30]})
        
        res = compute_privacy_metrics(df_orig, df_aug_leak, ["f1", "f2"])
        # Privacy risk should be 100% since all rows are within 0.05 distance
        self.assertEqual(res["privacy_risk"], 100)
        # NNDR should be 0 because nearest neighbor in real is distance 0
        self.assertEqual(res["nndr"], 0.0)

    def test_hellinger_categorical_fidelity(self):
        """Verify categorical fidelity (Hellinger)."""
        score_good = compute_hellinger_fidelity(self.df_orig["cat1"], self.df_aug_good["cat1"])
        score_bad = compute_hellinger_fidelity(self.df_orig["cat1"], self.df_aug_bad["cat1"])
        
        self.assertGreater(score_good, 0.7)
        self.assertLess(score_bad, 0.1)

    def test_calculate_metrics_integration(self):
        """Verify the master metrics aggregator."""
        res = calculate_metrics(self.df_orig, self.df_aug_good)
        self.assertIn("Synthetic Fidelity (JSD)", res)
        self.assertIn("Privacy Risk", res)
        self.assertIn("Novelty Score", res)

if __name__ == "__main__":
    unittest.main()
