import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.json_utils import robust_json_loads
from services.auditor import auditor
from services.ml_evaluator import MLEvaluator

class TestRobustnessSuite(unittest.TestCase):
    
    # --- JSON ROBUSTNESS ---
    def test_json_truncated_brackets(self):
        """Test auto-closing of truncated JSON blocks."""
        mangled = '{"cols": ["a", "b"], "data": [[1, 2], [3'
        result = robust_json_loads(mangled)
        self.assertIsNotNone(result)
        self.assertIn("data", result)
        self.assertEqual(result["data"][0], [1, 2])
        # The auto-closer should have added ]} to make it valid
    
    def test_json_missing_commas(self):
        """Test repair of missing commas between objects/arrays."""
        mangled = '{"a": 1} {"b": 2}' # Fixed to [{"a":1}, {"b":2}] or similar
        # Our regex fixes '}{' to '}, {'
        result = robust_json_loads(mangled)
        # result should be [{"a": 1}, {"b": 2}] if extracted correctly
        self.assertIsInstance(result, list)
    
    def test_json_python_literals(self):
        """Test handling of Python-style False/None/True."""
        mangled = '[{"active": False, "value": None, "valid": True}]'
        result = robust_json_loads(mangled)
        self.assertEqual(result[0]["active"], False)
        self.assertIsNone(result[0]["value"])
        self.assertEqual(result[0]["valid"], True)

    # --- AUDITOR ROBUSTNESS ---
    def test_auditor_empty_df(self):
        """Ensure auditor doesn't crash on an empty DataFrame and returns structured defaults."""
        df = pd.DataFrame()
        result = auditor.get_structural_metrics(df)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("row_count"), 0)
        self.assertIn("type_distribution", result)
        
    def test_auditor_all_nan_column(self):
        """Ensure auditor handles columns containing only NaNs."""
        df = pd.DataFrame({"nan_col": [np.nan, np.nan]})
        result = auditor.get_structural_metrics(df)
        self.assertEqual(result["sparsity"]["nan_col"], 1.0)
        # Skewness should be empty for a column with no numbers
        self.assertNotIn("nan_col", result["skewness"])

    # --- ML ROBUSTNESS ---
    def test_ml_single_class_classification(self):
        """Ensure ML evaluator handles target columns with only one unique value."""
        df = pd.DataFrame({"f1": [1, 2, 3], "target": [1, 1, 1]})
        # For a single class, train_test_split or the model will fail
        perf, _, err = MLEvaluator.train_and_evaluate(df, ["target"], "rf", {}, "classification")
        self.assertIsNotNone(err)
        self.assertIn("Single class in train", err)

    def test_ml_constant_regression(self):
        """Ensure ML evaluator handles target columns with constant numeric values."""
        df = pd.DataFrame({"f1": [1, 2, 3], "target": [10.5, 10.5, 10.5]})
        perf, _, err = MLEvaluator.train_and_evaluate(df, ["target"], "rf", {}, "regression")
        # Regression should technically work but return R2=0 or similar
        self.assertIn("rmse", perf)
        self.assertEqual(perf["rmse"], 0.0)

if __name__ == "__main__":
    unittest.main()
