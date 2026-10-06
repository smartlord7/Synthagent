import unittest
import pandas as pd
import numpy as np
from services.ml_evaluator import MLEvaluator

class TestMLEvaluator(unittest.TestCase):
    def setUp(self):
        # Classification Data
        self.df_clf = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Regression Data
        self.df_reg = pd.DataFrame({
            "feature1": np.arange(20),
            "target": np.arange(20) * 1.5 + 2
        })

    def test_detect_task_type(self):
        # Numeric target with many unique values -> regression
        self.assertEqual(MLEvaluator.detect_task_type(self.df_reg, ["target"]), "regression")
        
        # Categorical target -> classification
        self.assertEqual(MLEvaluator.detect_task_type(self.df_clf, ["target"]), "classification")
        
        # Numeric target but low cardinality -> classification
        df_low_card = pd.DataFrame({"target": [1, 2, 1, 2]})
        self.assertEqual(MLEvaluator.detect_task_type(df_low_card, ["target"]), "classification")

    def test_preprocess_data_impute(self):
        df_nan = pd.DataFrame({
            "num": [1, 2, np.nan, 4],
            "cat": ["A", "B", np.nan, "A"]
        })
        
        # Impute strategy
        df_clean = MLEvaluator.preprocess_data(df_nan, strategy="impute")
        self.assertFalse(df_clean.isnull().values.any())
        self.assertEqual(df_clean["num"][2], 7/3) # Mean of 1, 2, 4
        self.assertEqual(df_clean["cat"][2], "A") # Mode of A, B, A

    def test_preprocess_data_drop(self):
        df_nan = pd.DataFrame({
            "num": [1, 2, np.nan, 4],
            "cat": ["A", "B", "C", "A"]
        })
        
        # Drop strategy
        df_clean = MLEvaluator.preprocess_data(df_nan, strategy="drop")
        self.assertEqual(len(df_clean), 3)
        self.assertFalse(df_clean.isnull().values.any())

    def test_train_and_evaluate_classification(self):
        # Test Random Forest Classifier
        metrics, metadata, error = MLEvaluator.train_and_evaluate(
            self.df_clf, ["target"], "rf", task_type="classification"
        )
        
        self.assertIsNone(error)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertEqual(metadata["model_type"], "rf")
        self.assertIn("feature_importances", metadata)

    def test_train_and_evaluate_regression(self):
        # Test Ridge Regressor
        metrics, metadata, error = MLEvaluator.train_and_evaluate(
            self.df_reg, ["target"], "linear", task_type="regression"
        )
        
        self.assertIsNone(error)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertEqual(metadata["model_type"], "linear")

    def test_get_model_adaptation(self):
        # Small dataset -> should pick smaller n_estimators
        model = MLEvaluator.get_model("rf", {}, "classification", X_train=self.df_clf[["feature1"]], y_train=self.df_clf["target"])
        self.assertLessEqual(model.n_estimators, 100) # adapted from n_samples/5 if n_samples=10

    def test_invalid_model_type(self):
        metrics, metadata, error = MLEvaluator.train_and_evaluate(
            self.df_clf, ["target"], "invalid-model", task_type="classification"
        )
        self.assertIsNone(metrics)
        self.assertIn("Invalid model type", error)

if __name__ == "__main__":
    unittest.main()


class TestXGBNonZeroBasedLabels(unittest.TestCase):
    """XGBClassifier rejects class labels that don't start at 0 (e.g. the {1, 2} `Selector`
    target in indian_liver_patient_dataset). The evaluator encodes for fit and decodes
    predictions back so metrics stay in the original label space."""

    def _frame(self, labels):
        rng = np.random.RandomState(0)
        n = 60
        y = np.array([labels[0]] * 30 + [labels[1]] * 30)
        x = rng.randn(n) + (y == labels[1]) * 3.0   # separable, so accuracy should be high
        return pd.DataFrame({"x": x, "target": y})

    def test_xgb_trains_on_labels_1_and_2(self):
        df = self._frame([1, 2])
        metrics, _, err = MLEvaluator.train_and_evaluate(df, ["target"], "xgb", {}, "classification", test_df=df)
        self.assertIsNone(err)
        self.assertGreater(metrics["accuracy"], 0.9)

    def test_xgb_f1_matches_rf_on_shifted_labels(self):
        # f1 uses pos_label=1. If XGB were scored in encoded space, pos_label would silently
        # become original class 2 and f1 would diverge from every other model's.
        df = self._frame([1, 2])
        xgb_m, _, _ = MLEvaluator.train_and_evaluate(df, ["target"], "xgb", {}, "classification", test_df=df)
        rf_m, _, _ = MLEvaluator.train_and_evaluate(df, ["target"], "rf", {}, "classification", test_df=df)
        self.assertAlmostEqual(xgb_m["f1"], rf_m["f1"], delta=0.1)

    def test_xgb_still_works_on_zero_based_labels(self):
        df = self._frame([0, 1])
        metrics, _, err = MLEvaluator.train_and_evaluate(df, ["target"], "xgb", {}, "classification", test_df=df)
        self.assertIsNone(err)
        self.assertGreater(metrics["accuracy"], 0.9)
