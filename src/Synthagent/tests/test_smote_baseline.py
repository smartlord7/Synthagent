import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.smote_baseline import generate_smote_rows


class TestGenerateSmoteRows(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        # Imbalanced numeric + categorical dataset: 30 majority rows, 8 minority rows.
        self.df_train = pd.DataFrame({
            "num1": rng.rand(38) * 100,
            "num2": rng.rand(38) * 10,
            "sex": (["M"] * 19 + ["F"] * 19),
            "target": [0] * 30 + [1] * 8,
        })

    def test_returns_only_synthetic_minority_rows(self):
        synth = generate_smote_rows(self.df_train, "target", categorical_cols=["sex"], random_state=42)
        self.assertGreater(len(synth), 0)
        # SMOTE only ever synthesizes minority-class rows.
        self.assertTrue((synth["target"] == 1).all())

    def test_synthetic_rows_have_valid_categorical_labels(self):
        synth = generate_smote_rows(self.df_train, "target", categorical_cols=["sex"], random_state=42)
        self.assertTrue(set(synth["sex"].unique()).issubset({"M", "F"}))

    def test_returns_same_columns_as_input(self):
        synth = generate_smote_rows(self.df_train, "target", categorical_cols=["sex"], random_state=42)
        self.assertEqual(list(synth.columns), list(self.df_train.columns))

    def test_no_categorical_columns_uses_plain_smote(self):
        df_numeric = self.df_train.drop(columns=["sex"])
        synth = generate_smote_rows(df_numeric, "target", categorical_cols=[], random_state=42)
        self.assertGreater(len(synth), 0)
        self.assertTrue((synth["target"] == 1).all())

    def test_missing_target_column_returns_empty(self):
        synth = generate_smote_rows(self.df_train, "nonexistent", categorical_cols=["sex"], random_state=42)
        self.assertTrue(synth.empty)

    def test_single_class_returns_empty(self):
        df_single = pd.DataFrame({"num1": range(10), "target": [0] * 10})
        synth = generate_smote_rows(df_single, "target", categorical_cols=[], random_state=42)
        self.assertTrue(synth.empty)

    def test_too_few_minority_rows_returns_empty(self):
        """k_neighbors = min(5, minority_count - 1); a single minority row gives 0, below
        the k_neighbors >= 1 floor, so SMOTE cannot run at all."""
        df_tiny_minority = pd.DataFrame({
            "num1": list(range(10)),
            "target": [0] * 9 + [1] * 1,
        })
        synth = generate_smote_rows(df_tiny_minority, "target", categorical_cols=[], random_state=42)
        self.assertTrue(synth.empty)

    def test_empty_dataframe_returns_empty(self):
        df_empty = pd.DataFrame(columns=["num1", "target"])
        synth = generate_smote_rows(df_empty, "target", categorical_cols=[], random_state=42)
        self.assertTrue(synth.empty)


if __name__ == "__main__":
    unittest.main()
