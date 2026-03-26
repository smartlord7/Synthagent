import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Add src/TabuLLM to path
sys.path.append(os.getcwd())

from services.llm_client import LocalOpenAIProvider
from utils.json_utils import RobustJSONError

class TestSelfCorrection(unittest.TestCase):
    
    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_local_provider_repair_success(self, mock_call):
        # 1. Setup mock responses
        # First call returns malformed JSON
        malformed_json = '{"data": [{"name": "John", "age": 30' # Missing closing brackets
        
        # Second call returns fixed JSON
        fixed_json = '{"data": [{"name": "John", "age": 30}]}'
        
        mock_call.side_effect = [
            # First response
            {
                "choices": [{"message": {"content": malformed_json}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            },
            # Repair response
            {
                "choices": [{"message": {"content": fixed_json}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            }
        ]
        
        provider = LocalOpenAIProvider()
        
        # 2. Execute
        rows_payload = [{"name": None, "age": None}]
        result = provider.generate_bulk_data(
            rows_payload=rows_payload,
            context={},
            all_columns=["name", "age"],
            algorithm="Standard",
            instructions="Generate data",
            temperature=0.7
        )
        
        # 3. Assert
        self.assertIsNotNone(result)
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["name"], "John")
        self.assertEqual(mock_call.call_count, 2)
        
        # Verify the repair prompt was sent in the second call
        repair_call_args = mock_call.call_args_list[1][0]
        messages = repair_call_args[0]
        self.assertIn("The following JSON fragment is malformed", messages[-1]["content"])

if __name__ == "__main__":
    unittest.main()
