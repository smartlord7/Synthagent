import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pipeline_manager import PipelineManager, run_single_dataset_benchmark
from services.data_manager import parse_uploaded_file
from services.report_generator import PDFReportGenerator, generate_benchmark_pdf

class TestSystemIntegration(unittest.TestCase):
    
    def setUp(self):
        self.df_orig = pd.DataFrame({
            "age": [20, 30, 40, 50, 60],
            "income": [20000, 30000, 40000, 50000, 60000],
            "target": [0, 1, 0, 1, 0]
        })
        self.col_defs = [
            {"field": "age", "headerName": "Age"},
            {"field": "income", "headerName": "Income"},
            {"field": "target", "headerName": "Target"}
        ]
        self.config = {
            "folds": 2,
            "ctx_ratio": 20,
            "algorithm": "test-mock",
            "ml_config": {"selected_models": ["rf"], "params": {}},
            "targets": ["Target"],
            "desc_metadata": "Test Metadata",
            "instructions": "Test Instructions"
        }

    @patch('services.pipeline_manager.generate_bulk_data')
    def test_multiprocessing_pipeline_execution(self, mock_gen):
        """Verify that the pipeline manager survives multiprocessing fork/spawn."""
        mock_gen.side_effect = lambda *args, **kwargs: {
            "data": [{"Age": 33, "Income": 35000, "Target": 0}], 
            "stats": {"total_tokens": 100}
        }
        
        # Two identical tasks to run in parallel
        tasks = [
            ("ds1", "file1.csv", self.df_orig.to_dict('records'), self.col_defs, self.config),
            ("ds2", "file2.csv", self.df_orig.to_dict('records'), self.col_defs, self.config)
        ]
        
        results = PipelineManager.run_pipeline(tasks)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["dataset_id"], "ds1")
        self.assertEqual(results[1]["dataset_id"], "ds2")
        # Ensure folds were processed
        self.assertEqual(len(results[0]["results"]), 2)

    @patch('services.llm_client.LocalOpenAIProvider.get_dataset_description')
    @patch('services.pipeline_manager.generate_bulk_data')
    def test_the_great_chain(self, mock_gen, mock_desc):
        """Test the full flow: Upload -> Insights -> Generate -> Report."""
        # 1. Setup mocks
        mock_desc.return_value = "# Dataset Insights\n- Key trend: Age correlates with income."
        mock_gen.side_effect = lambda *args, **kwargs: {
            "data": [{"Age": 45, "Income": 45000, "Target": 1}], 
            "stats": {"total_tokens": 50}
        }
        
        # 2. Upload / Parse (Simulated base64)
        mock_contents = "data:text/csv;base64,YWdlLGluY29tZSx0YXJnZXQKMjAsMjAwMDAsMAozMCwzMDAwMCwxCjQwLDQwMDAwLDA="
        df_uploaded = parse_uploaded_file(mock_contents, "test.csv")
        self.assertEqual(len(df_uploaded), 3)
        
        # 3. Insights (Smart Define)
        # We've already mocked mock_desc. No need to call manually here, but we check logic.
        
        # 4. Pipeline Execution
        task = ("test_chain_ds", "test.csv", df_uploaded.to_dict('records'), self.col_defs, self.config)
        res = run_single_dataset_benchmark(task)
        
        # 5. Report Generation
        report_data = {
            "executive_summary": {"avg_fidelity": res["avg_fidelity"], "global_metrics": {}},
            "datasets": {
                "test_chain_ds": {
                    "filename": "test.csv",
                    "metadata": {"Rows": 3, "Cols": 3},
                    "params": {"generation": {"Algorithm": "test-mock"}},
                    "metrics": {"Fidelity": res["avg_fidelity"]},
                    "ml_results": res["overall_ml"]
                }
            }
        }
        
        out_path = os.path.join(os.path.dirname(__file__), "integration_test_report.pdf")
        pdf_bytes = generate_benchmark_pdf(report_data)
        
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        
        # Verify report file exists
        self.assertTrue(os.path.exists(out_path))
        # Cleanup
        if os.path.exists(out_path):
            os.remove(out_path)
        # Cleanup
        if os.path.exists(out_path):
            os.remove(out_path)

if __name__ == "__main__":
    unittest.main()
