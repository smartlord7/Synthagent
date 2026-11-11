import unittest
import pandas as pd
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthagent import Synthagent

class TestLibraryFullPipeline(unittest.TestCase):
    
    def setUp(self):
        os.environ["TABULLM_SYNC_TEST"] = "1"
        self.lib = Synthagent(provider="local", model="test-lib-model")
        # Small dataset for real pipeline execution
        self.df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    @patch('services.llm_client.LocalOpenAIProvider.get_available_models')
    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_full_pipeline_flow(self, mock_llm, mock_models):
        """
        Verify that the library correctly orchestrates:
        1. K-Fold Augmentation
        2. ML Evaluation (Real Scikit-learn call)
        3. Comprehensive Auditing
        4. AI Pondering
        """
        # --- Mock Setup ---
        mock_models.return_value = [{"id": "test-lib-model", "max_context": 4096}]
        # 1. Models List
        # 2. Generation Logic (Success)
        # 3. Ponder Strategy Logic
        def side_effect_func(*args, **kwargs):
            # If it's a JSON mode call (generation), return valid synthetic rows
            if kwargs.get('json_mode'):
                content = json.dumps({"cols": ["age", "gender", "income", "target"], "data": [[30, "M", 50, 0]]})
                return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
            
            # Default response for ponder_strategy or unknown
            return {"choices": [{"message": {"content": "AI Strategy Plan: Your data looks good."}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}

        mock_llm.side_effect = side_effect_func

        print("\n[TEST] 1. Testing get_models()...")
        models = self.lib.get_models()
        self.assertTrue(len(models) > 0)

        print("[TEST] 2. Testing augment() with 2 Folds and RF/LR models...")
        # Note: PipelineManager.run_pipeline will be called for REAL
        results = self.lib.augment(
            self.df, 
            targets=["target"], 
            num_rows=2, 
            folds=2, 
            ml_models=["rf", "lr"]
        )
        
        # Verify K-fold results
        self.assertEqual(len(results["results"]), 2)
        self.assertIn("rf", results["overall_ml"])
        self.assertIn("lr", results["overall_ml"])
        print(f"   -> Success: Produced {len(results['results'])} folds and benchmarks for {list(results['overall_ml'].keys())}")

        print("[TEST] 3. Testing run_audit()...")
        audit = self.lib.run_audit(self.df, targets=["target"])
        self.assertIn("fairness", audit)
        self.assertIn("privacy", audit)
        self.assertIn("structural", audit)
        self.assertIn("quality", audit)
        print("   -> Success: Audit contains all 4 major axes.")

        print("[TEST] 4. Testing ponder_strategy()...")
        strategy = self.lib.ponder_strategy(self.df, targets=["target"])
        self.assertIn("AI Strategy Plan", strategy)
        print("   -> Success: AI Pondering returned valid guidance.")

if __name__ == "__main__":
    unittest.main()
