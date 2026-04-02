import unittest
import pandas as pd
import numpy as np
from services.pipeline_manager import run_single_dataset_benchmark

class TestPipelineOrchestration(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "f1": np.random.rand(20),
            "f2": np.random.rand(20),
            "target": np.random.choice([0, 1], size=20)
        })
        self.col_defs = [
            {"field": "f1", "headerName": "F1"},
            {"field": "f2", "headerName": "F2"},
            {"field": "target", "headerName": "Target"}
        ]
        self.config = {
            "folds": 2,
            "ml_config": {"selected_models": ["rf"], "params": {}},
            "targets": ["target"],
            "desc_metadata": "Test Metadata",
            "algorithm": "test-mock", # This will fail real LLM calls, but we check logic around it
            "ctx_ratio": 20
        }

    def test_run_single_dataset_benchmark_structure(self):
        # We need to mock the LLM client because real calls are expensive/local
        # For this logic test, we'll see if it handles the fold splitting
        
        # This function calls generate_bulk_data inside. 
        # For a truly headless unit test, we should mock generate_bulk_data.
        
        with unittest.mock.patch('services.pipeline_manager.generate_bulk_data') as mock_gen:
            mock_gen.side_effect = lambda *args, **kwargs: {"data": [{"f1": 0.5, "f2": 0.5, "target": 1}], "stats": {}}
            
            payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, self.config)
            result = run_single_dataset_benchmark(payload)
            
            self.assertEqual(result["dataset_id"], "test_ds")
            self.assertEqual(len(result["results"]), 2) # 2 folds
            # Check if fold_id was stamped
            for row in result["results"][0]["synthetic_data"]:
                self.assertEqual(row["_fold_id"], 1)

if __name__ == '__main__':
    unittest.main()
