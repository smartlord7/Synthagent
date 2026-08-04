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
    calculate_metrics,
    calculate_metrics_by_class,
    compute_c2st_accuracy,
    compute_mia_attack_auc,
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

    def test_calculate_metrics_by_class_splits_minority_majority(self):
        """Verify the minority/majority split is by prevalence in df_orig, not df_aug."""
        rng = np.random.RandomState(0)
        df_orig = pd.DataFrame({
            "num1": rng.rand(30) * 100,
            "target": [0] * 20 + [1] * 10,   # majority=0 (20 rows), minority=1 (10 rows)
        })
        df_aug = pd.DataFrame({
            "num1": rng.rand(30) * 100,
            "target": [0] * 20 + [1] * 10,
        })
        result = calculate_metrics_by_class(df_orig, df_aug, "target")
        self.assertEqual(result["minority_label"], "1")
        self.assertEqual(result["majority_label"], "0")
        self.assertIn("minority", result)
        self.assertIn("majority", result)
        self.assertIn("Synthetic Fidelity (JSD)", result["minority"])
        self.assertIn("Synthetic Fidelity (JSD)", result["majority"])

    def test_calculate_metrics_by_class_missing_target_returns_empty(self):
        result = calculate_metrics_by_class(self.df_orig, self.df_aug_good, "nonexistent_col")
        self.assertEqual(result, {})

    def test_calculate_metrics_by_class_single_class_returns_empty(self):
        df_orig = pd.DataFrame({"num1": [1, 2, 3, 4, 5, 6], "target": [0] * 6})
        df_aug = pd.DataFrame({"num1": [1, 2, 3, 4, 5, 6], "target": [0] * 6})
        result = calculate_metrics_by_class(df_orig, df_aug, "target")
        self.assertEqual(result, {})

    def test_calculate_metrics_by_class_skips_too_few_rows(self):
        """A class subset under 6 rows shouldn't produce a (misleadingly noisy) metrics block."""
        df_orig = pd.DataFrame({"num1": list(range(8)), "target": [0] * 6 + [1] * 2})
        df_aug = pd.DataFrame({"num1": list(range(8)), "target": [0] * 6 + [1] * 2})
        result = calculate_metrics_by_class(df_orig, df_aug, "target")
        self.assertIn("minority_label", result)
        self.assertNotIn("minority", result)  # only 2 rows in the minority class, below threshold


class TestC2STAndMIA(unittest.TestCase):
    """Privacy/discriminability attack metrics: C2ST (classifier two-sample test) and a
    distance-based membership-inference-attack (MIA) simulation."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.df_real = pd.DataFrame({
            "num1": rng.normal(0, 1, 200),
            "num2": rng.normal(0, 1, 200),
        })
        # Indistinguishable synthetic: same generating distribution.
        self.df_synth_similar = pd.DataFrame({
            "num1": rng.normal(0, 1, 200),
            "num2": rng.normal(0, 1, 200),
        })
        # Trivially distinguishable synthetic: shifted far away.
        self.df_synth_different = pd.DataFrame({
            "num1": rng.normal(100, 1, 200),
            "num2": rng.normal(100, 1, 200),
        })

    def test_c2st_near_chance_on_identical_distributions(self):
        acc = compute_c2st_accuracy(self.df_real, self.df_synth_similar)
        self.assertIsNotNone(acc)
        self.assertLess(acc, 0.7)

    def test_c2st_high_on_clearly_distinguishable_distributions(self):
        acc = compute_c2st_accuracy(self.df_real, self.df_synth_different)
        self.assertIsNotNone(acc)
        self.assertGreater(acc, 0.9)

    def test_c2st_none_on_too_few_rows(self):
        acc = compute_c2st_accuracy(self.df_real.head(5), self.df_synth_similar.head(5))
        self.assertIsNone(acc)

    def test_c2st_none_on_no_common_columns(self):
        df_other = pd.DataFrame({"other_col": [1, 2, 3] * 10})
        acc = compute_c2st_accuracy(self.df_real, df_other)
        self.assertIsNone(acc)

    def test_mia_auc_near_chance_when_synthetic_equidistant(self):
        """Synthetic drawn from the same distribution as both members and non-members
        should give the attacker no advantage (~0.5 AUC)."""
        rng = np.random.RandomState(1)
        members = pd.DataFrame({"num1": rng.normal(0, 1, 50), "num2": rng.normal(0, 1, 50)})
        nonmembers = pd.DataFrame({"num1": rng.normal(0, 1, 50), "num2": rng.normal(0, 1, 50)})
        synth = pd.DataFrame({"num1": rng.normal(0, 1, 50), "num2": rng.normal(0, 1, 50)})
        auc = compute_mia_attack_auc(members, nonmembers, synth)
        self.assertIsNotNone(auc)
        self.assertGreater(auc, 0.2)
        self.assertLess(auc, 0.8)

    def test_mia_auc_high_when_synthetic_leaks_membership(self):
        """Synthetic data that clusters tightly around the members (and far from
        non-members) should be an easy, high-AUC attack."""
        rng = np.random.RandomState(1)
        members = pd.DataFrame({"num1": rng.normal(0, 0.1, 50), "num2": rng.normal(0, 0.1, 50)})
        nonmembers = pd.DataFrame({"num1": rng.normal(50, 0.1, 50), "num2": rng.normal(50, 0.1, 50)})
        synth = pd.DataFrame({"num1": rng.normal(0, 0.1, 50), "num2": rng.normal(0, 0.1, 50)})
        auc = compute_mia_attack_auc(members, nonmembers, synth)
        self.assertIsNotNone(auc)
        self.assertGreater(auc, 0.9)

    def test_mia_auc_none_on_too_few_rows(self):
        auc = compute_mia_attack_auc(self.df_real.head(2), self.df_real.head(2), self.df_synth_similar.head(2))
        self.assertIsNone(auc)

    def test_mia_auc_none_when_no_numeric_common_columns(self):
        members = pd.DataFrame({"cat": ["a"] * 10})
        nonmembers = pd.DataFrame({"cat": ["b"] * 10})
        synth = pd.DataFrame({"cat": ["a"] * 10})
        auc = compute_mia_attack_auc(members, nonmembers, synth)
        self.assertIsNone(auc)


if __name__ == "__main__":
    unittest.main()
