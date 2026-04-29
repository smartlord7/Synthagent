import unittest
import json
from unittest.mock import patch, MagicMock
from services.llm_client import LocalOpenAIProvider

class TestLLMBehaviors(unittest.TestCase):
    def setUp(self):
        self.provider = LocalOpenAIProvider()

    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_topup_guarantee_logic(self, mock_call):
        """
        Scenario: Requested 5 rows, LLM returns 3 rows in first chunk.
        Provider should trigger Top-up loop to get remaining 2 rows.
        """
        # First call: returns 3 rows
        resp1 = {
            "choices": [{"message": {"content": json.dumps({"cols": ["a"], "data": [[1], [2], [3]]})}}],
            "usage": {"total_tokens": 100}
        }
        # Second call (Top-up): returns remaining 2 rows
        resp2 = {
            "choices": [{"message": {"content": json.dumps({"cols": ["a"], "data": [[4], [5]]})}}],
            "usage": {"total_tokens": 50}
        }
        
        mock_call.side_effect = [resp1, resp2]
        
        result = self.provider.generate_bulk_data(
            rows_payload=[],
            context={},
            all_columns=["a"],
            all_fields=["a"],
            algorithm="test",
            instructions="Test",
            temperature=0.7,
            num_empty=5,
            chunk_size=5
        )
        
        self.assertEqual(len(result["data"]), 5)
        self.assertEqual(mock_call.call_count, 2)

    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_self_correction_logic(self, mock_call):
        """
        Scenario: LLM returns a row that violates a rule.
        Provider should trigger RuleValidator and then a fix prompt to the LLM.
        """
        # First call: returns a violation (age=150 while max is 100)
        resp1 = {
            "choices": [{"message": {"content": json.dumps({"cols": ["age"], "data": [[150]]})}}],
            "usage": {"total_tokens": 100}
        }
        # Second call (Correction): returns fixed row
        resp2 = {
            "choices": [{"message": {"content": json.dumps({"cols": ["age"], "data": [[25]]})}}],
            "usage": {"total_tokens": 50}
        }
        
        mock_call.side_effect = [resp1, resp2]
        
        rules = [{"name": "age", "type": "number", "level": 2, "max": 100}]
        
        result = self.provider.generate_bulk_data(
            rows_payload=[],
            context={},
            all_columns=["age"],
            all_fields=["age"],
            algorithm="test",
            instructions="Test",
            temperature=0.7,
            num_empty=1,
            enforce_rules=True,
            rules_metadata={"integrity": rules}
        )
        
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["age"], 25)
        # 1 initial + 1 correction = 2 calls
        self.assertEqual(mock_call.call_count, 2)

if __name__ == "__main__":
    unittest.main()
