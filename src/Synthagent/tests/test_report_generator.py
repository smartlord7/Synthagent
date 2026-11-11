import unittest
import os
from services.report_generator import generate_benchmark_pdf

class TestReportGenerator(unittest.TestCase):
    def test_generate_pdf_full_data(self):
        # Mock report data
        report_data = {
            "executive_summary": {
                "avg_fidelity": 0.85,
                "global_metrics": {"Stability": "High", "Privacy": "Secure"}
            },
            "datasets": {
                "ds1": {
                    "filename": "test_data.csv",
                    "metadata": {"Rows": 100, "Cols": 10},
                    "params": {"generation": {"Model": "GPT-4", "Temperature": 0.7}},
                    "metrics": {"Fidelity": 0.9, "Privacy Risk": 0.1},
                    "ml_results": {
                        "random_forest": {
                            "accuracy": {"real": 0.8, "aug": 0.82, "delta": "+2.5%"},
                            "f1": {"real": 0.75, "aug": 0.78, "delta": "+4.0%"}
                        }
                    },
                    "samples": {
                        "orig": None # Testing None handling
                    }
                }
            }
        }
        
        try:
            pdf_bytes = generate_benchmark_pdf(report_data)
            self.assertTrue(len(pdf_bytes) > 0)
            
            # Save to disk for manual inspection if needed
            output_path = "test_report_output.pdf"
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            self.assertTrue(os.path.exists(output_path))
            os.remove(output_path)
            
        except Exception as e:
            self.fail(f"PDF generation failed: {e}")

if __name__ == "__main__":
    unittest.main()
