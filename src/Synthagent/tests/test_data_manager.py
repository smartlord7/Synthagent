import unittest
import pandas as pd
import numpy as np
import base64
from services.data_manager import parse_uploaded_file, get_eda_metrics, select_context_rows

class TestDataManager(unittest.TestCase):
    
    def test_parse_uploaded_file_csv(self):
        # Create a simple CSV
        csv_content = "a,b,c\n1,2,3\n4,5,6"
        encoded = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
        contents = f"data:text/csv;base64,{encoded}"
        
        df = parse_uploaded_file(contents, "test.csv")
        self.assertEqual(df.shape, (2, 3))
        self.assertEqual(list(df.columns), ["a", "b", "c"])

    def test_get_eda_metrics(self):
        df = pd.DataFrame({
            "num": [1, 2, np.nan],
            "cat": ["X", "Y", "X"]
        })
        metrics = get_eda_metrics(df)
        self.assertIn("columns", metrics)
        self.assertEqual(metrics["shape"], (3, 2))
        self.assertEqual(metrics["missing_values"]["num"], 1)

    def test_select_context_rows_random(self):
        df = pd.DataFrame({"f": range(100)})
        selected = select_context_rows(df, strategy="random", n_rows=10)
        self.assertEqual(len(selected), 10)

    def test_select_context_rows_dense(self):
        # Row 2 is most dense (no NaNs)
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [np.nan, 2, 3],
            "f3": [np.nan, np.nan, 3]
        })
        # Note: 1 is most sparse, 3 is most dense
        selected = select_context_rows(df, strategy="dense", n_rows=1)
        # index 2 corresponds to [3, 3, 3]
        self.assertEqual(selected[0]["f1"], 3)

    def test_select_context_rows_diverse(self):
        # Points in two clusters
        df = pd.DataFrame({
            "f1": [1, 1.1, 10, 10.1],
            "f2": [1, 0.9, 10, 9.9]
        })
        # Diverse strategy should pick one from each cluster if n_rows=2
        selected = select_context_rows(df, strategy="diverse", n_rows=2)
        self.assertEqual(len(selected), 2)
        
        # Check that we got one from low and one from high
        vals = [s["f1"] for s in selected]
        self.assertTrue(any(v < 2 for v in vals))
        self.assertTrue(any(v > 9 for v in vals))

if __name__ == "__main__":
    unittest.main()
