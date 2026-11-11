import unittest
import json
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.llm_client import LocalOpenAIProvider

class TestLLMIntelligence(unittest.TestCase):
    
    def setUp(self):
        self.provider = LocalOpenAIProvider()
        self.all_cols = ["f1", "f2", "target"]
        self.ctx = {"metadata": "test", "manual_ctx": "[]", "global_ctx": []}

    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_topup_guarantee_loop(self, mock_llm):
        """Test that the Top-Up loop correctly requests missing rows until complete."""
        # 1. Initial bulk returns 1 row (instead of 2)
        mock_llm.side_effect = [
            {"choices": [{"message": {"content": '{"cols": ["f1", "f2", "target"], "data": [[1, 2, 0]]}'}}], "usage": {}},
            {"choices": [{"message": {"content": '{"cols": ["f1", "f2", "target"], "data": [[3, 4, 1]]}'}}], "usage": {}}
        ]
        
        # Request 2 rows (num_empty=2)
        result = self.provider.generate_bulk_data([], self.ctx, self.all_cols, self.all_cols, "test-model", "test-inst", 0.7, num_empty=2)
        
        # Verify result has 2 rows
        self.assertEqual(len(result["data"]), 2)
        # Verify LLM was called twice (1 initial + 1 top-up)
        self.assertEqual(mock_llm.call_count, 2)

    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_syntax_self_healing_repair(self, mock_llm):
        """Test that a JSON parse error triggers a repair turn."""
        # 1. First call returns bad JSON (irreparable by regex)
        # 2. Second call (repair) returns fixed JSON
        # We use a string that 'robust_json_loads' will fail on even after regex fixes.
        mock_llm.side_effect = [
            {"choices": [{"message": {"content": '{"cols": ["f1"], "data": [[1], [?]]}'}}], "usage": {}},
            {"choices": [{"message": {"content": '{"cols": ["f1"], "data": [[1]]}'}}], "usage": {}}
        ]
        
        # Request 1 row
        result = self.provider.generate_bulk_data([], self.ctx, ["f1"], ["f1"], "test-model", "test-inst", 0.7, num_empty=1)
        
        self.assertEqual(len(result["data"]), 1)
        # Verify 2 calls (Original + Repair)
        self.assertEqual(mock_llm.call_count, 2)

    def test_token_budgeting_calculation(self):
        """Test transparency and safety of the token budgeter."""
        messages = [{"role": "user", "content": "test"}]
        limit = 4096
        
        # _calculate_budget returns a dict with 'rows_allowed'
        budget = self.provider._calculate_budget([], [], self.all_cols, messages, "inst", limit)
        self.assertIn("rows_allowed", budget)
        # It should be a reasonable positive number
        self.assertGreater(budget["rows_allowed"], 10)

    def test_context_rotation_logic(self):
        """Test that context sliding window rotates across chunks."""
        pool = [{"id": i} for i in range(100)]
        
        # Turn 0, allowed 20 -> indices [0:20]
        slice_0 = self.provider._apply_rotation(pool, 0, 20)
        self.assertEqual(slice_0[0]["id"], 0)
        self.assertEqual(len(slice_0), 20)
        
        # Turn 1, allowed 20 -> indices [20:40]
        slice_1 = self.provider._apply_rotation(pool, 1, 20)
        self.assertEqual(slice_1[0]["id"], 20)
        
        # Overflow/Wrap logic
        slice_wrap = self.provider._apply_rotation(pool, 10, 20) # 200 % 100 = 0. Oh wait, offset is (idx * allowed) % 100
        # If 10 * 20 = 200. Offset = 0.
        self.assertEqual(slice_wrap[0]["id"], 0)

if __name__ == "__main__":
    unittest.main()
