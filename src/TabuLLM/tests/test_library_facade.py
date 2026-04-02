import unittest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabullm import TabuLLM

class TestLibraryFacade(unittest.TestCase):
    
    def setUp(self):
        self.lib = TabuLLM(provider="local", model="test-lib-model")
        self.sample_df = pd.DataFrame({
            "age": [20, 30, 40],
            "income": [2000, 3000, 4000],
            "target": [0, 1, 0]
        })

    def test_load_df_csv(self):
        """Test loading from CSV (mocking pandas read_csv)."""
        with patch('pandas.read_csv') as mock_read:
            mock_read.return_value = self.sample_df
            df = self.lib.load_df("data.csv")
            self.assertEqual(len(df), 3)
            mock_read.assert_called_once()

    @patch('tabullm.api.LocalOpenAIProvider.detect_target_columns')
    def test_suggest_targets(self, mock_detect):
        """Test target suggestion delegation."""
        mock_detect.return_value = ["target"]
        targets = self.lib.suggest_targets(self.sample_df)
        self.assertEqual(targets, ["target"])
        mock_detect.assert_called_once()

    @patch('tabullm.api.LocalOpenAIProvider.get_dataset_description')
    def test_get_insights(self, mock_desc):
        """Test insights generation delegation."""
        mock_desc.return_value = "Insights description"
        res = self.lib.get_insights(self.sample_df)
        self.assertEqual(res, "Insights description")
        mock_desc.assert_called_once()

    @patch('tabullm.api.PipelineManager.run_pipeline')
    def test_augment_pipeline(self, mock_pipe):
        """Test the full augmentation pipeline delegation."""
        mock_pipe.return_value = [{"dataset_id": "lib-ds", "avg_fidelity": 0.9, "overall_ml": {}}]
        results = self.lib.augment(self.sample_df, targets=["target"], num_rows=10)
        
        self.assertEqual(results["dataset_id"], "lib-ds")
        self.assertEqual(results["avg_fidelity"], 0.9)
        mock_pipe.assert_called_once()

    @patch('tabullm.api.generate_benchmark_pdf')
    def test_generate_pdf_report(self, mock_pdf):
        """Test PDF report generation delegation."""
        mock_pdf.return_value = b"%PDF-1.4 mock content"
        out_path = "test_lib_report.pdf"
        
        # Test the wrapper
        self.lib.generate_pdf_report({"avg_fidelity": 0.8}, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        if os.path.exists(out_path):
            os.remove(out_path)
        mock_pdf.assert_called_once()

if __name__ == "__main__":
    unittest.main()
