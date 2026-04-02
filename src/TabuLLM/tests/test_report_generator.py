import unittest
import pandas as pd
import io
from services.report_generator import generate_benchmark_pdf

class TestReportGenerator(unittest.TestCase):
    def test_generate_pdf_structure(self):
        # Prepare dummy but correctly shaped report data
        report_data = {
            "executive_summary": {
                "avg_fidelity": 0.88,
                "global_metrics": {"Avg Precision": "0.82 \u00b1 0.05"}
            },
            "datasets": {
                "dataset_alpha": {
                    "filename": "alpha_gen.csv",
                    "metadata": {"Rows": "50", "Columns": "5"},
                    "params": {
                        "generation": {"algorithm": "gemma-3-4b", "tone": "Professional", "consistency": "Strict"}
                    },
                    "metrics": {
                        "Synthetic Fidelity (JSD)": 0.92,
                        "Privacy Risk": 0.05
                    },
                    "plots": [], # Skip plots for headless unit test
                    "samples": {
                        "orig": pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]}),
                        "aug": pd.DataFrame({"col_a": [1.1, 2.1, 3.1], "col_b": ["x", "y", "z"]})
                    },
                    "inference_table": [
                        ["col_a", "2.0", "2.1", "+5%", "0.45", "Stable"]
                    ],
                    "ml_results": {
                        "rf": {
                            "accuracy": {"real": 0.90, "aug": 0.88, "delta": -0.02},
                            "f1": {"real": 0.89, "aug": 0.87, "delta": -0.02}
                        }
                    },
                    "appendix": [
                        {
                            "name": "Audit Trace: Fold 1",
                            "sample": pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]}),
                            "row_colors": [(220, 240, 255), (220, 255, 220)],
                            "ctx_ratio": 20
                        }
                    ]
                }
            }
        }
        
        # This will return a bytearray/string of the PDF content
        pdf_out = generate_benchmark_pdf(report_data)
        
        self.assertIsInstance(pdf_out, (bytearray, bytes, str))
        self.assertGreater(len(pdf_out), 1000) # Basic check that content exists
        
        # Check PDF header magic bytes if bytes
        if isinstance(pdf_out, (bytes, bytearray)):
            self.assertTrue(pdf_out.startswith(b'%PDF'))

if __name__ == '__main__':
    unittest.main()
