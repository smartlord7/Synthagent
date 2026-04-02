import unittest
import pandas as pd
import numpy as np
from services.ml_evaluator import MLEvaluator

class TestMLEvaluatorTypes(unittest.TestCase):
    def setUp(self):
        # Classification dataset
        self.df_clf = pd.DataFrame({
            "f1": np.random.rand(100),
            "f2": np.random.rand(100),
            "target": np.random.choice([0, 1], size=100)
        })
        # Regression dataset
        self.df_reg = pd.DataFrame({
            "f1": np.random.rand(100),
            "f2": np.random.rand(100),
            "target": np.random.rand(100) * 100
        })
        self.model_types = ["logistic", "rf", "xgb", "mlp", "svm", "knn", "nb", "dt", "gb", "ab"]

    def test_classification_models(self):
        for m_type in self.model_types:
            metrics, metadata, err = MLEvaluator.train_and_evaluate(
                self.df_clf, ["target"], m_type, task_type="classification"
            )
            self.assertIsNone(err, f"Model {m_type} failed with error: {err}")
            self.assertIn("accuracy", metrics)
            self.assertIn("f1", metrics)
            self.assertIn("model_type", metadata)

    def test_regression_models(self):
        # Note: 'nb' (Naive Bayes) is popped out/replaced with Ridge in our implementation for regression
        reg_models = [m for m in self.model_types if m != "logistic"] # logistic is clf only, linear should use Ridge
        for m_type in reg_models:
            metrics, metadata, err = MLEvaluator.train_and_evaluate(
                self.df_reg, ["target"], m_type, task_type="regression"
            )
            self.assertIsNone(err, f"Model {m_type} failed with error: {err}")
            self.assertIn("rmse", metrics)
            self.assertIn("r2", metrics)

    def test_impute_strategy(self):
        df_nan = self.df_clf.copy()
        df_nan.loc[0, "f1"] = np.nan
        # Should not crash with impute
        metrics, metadata, err = MLEvaluator.train_and_evaluate(
            df_nan, ["target"], "rf", na_strategy="impute"
        )
        self.assertIsNone(err)

    def test_drop_strategy(self):
        df_nan = self.df_clf.copy()
        df_nan.loc[0, "f1"] = np.nan
        # Should drop first row
        metrics, metadata, err = MLEvaluator.train_and_evaluate(
            df_nan, ["target"], "rf", na_strategy="drop"
        )
        self.assertIsNone(err)

if __name__ == '__main__':
    unittest.main()
