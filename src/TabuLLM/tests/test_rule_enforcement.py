import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add src/TabuLLM to path
sys.path.append(os.getcwd())

from services.rule_validator import RuleValidator
from services.llm_client import LocalOpenAIProvider

class TestRuleEnforcement(unittest.TestCase):
    
    def setUp(self):
        self.validator = RuleValidator()
        self.rules = [
            {"name": "Age", "type": "number", "min": 18, "max": 65},
            {"name": "Status", "type": "category", "text": "Active, Inactive"}
        ]

    def test_pydantic_model_generation(self):
        """Verifies that a valid Pydantic model is created from UI rules."""
        Model = self.validator.create_pydantic_model(self.rules)
        self.assertIsNotNone(Model)
        
        # Valid data should pass
        valid_row = {"Age": 25, "Status": "Active"}
        try:
            Model(**valid_row)
        except Exception as e:
            self.fail(f"Model failed to validate correct data: {e}")

    def test_validation_logic(self):
        """Verifies that the validator correctly identifies out-of-bounds data."""
        Model = self.validator.create_pydantic_model(self.rules)
        test_data = [
            {"Age": 70, "Status": "Active"},   # Age too high
            {"Age": 25, "Status": "Pending"},  # Invalid Category
        ]
        
        violations = self.validator.validate_batch(test_data, Model)
        self.assertEqual(len(violations), 2)
        
        fields_with_errors = [v['field'] for v in violations]
        self.assertIn("Age", fields_with_errors)
        self.assertIn("Status", fields_with_errors)

    @patch('services.llm_client.LocalOpenAIProvider._call_local_llm')
    def test_self_correction_loop_integration(self, mock_call):
        """Verifies that LocalOpenAIProvider triggers a retry when rules are violated."""
        # 1. First response violates rules (Age=70)
        violation_json = '{"data": [{"Age": 70, "Status": "Active"}]}'
        # 2. Second response (retry) fixes the data (Age=30)
        fixed_json = '{"data": [{"Age": 30, "Status": "Active"}]}'
        
        mock_call.side_effect = [
            # Initial call
            {
                "choices": [{"message": {"content": violation_json}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            },
            # Correction call
            {
                "choices": [{"message": {"content": fixed_json}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            }
        ]
        
        provider = LocalOpenAIProvider()
        result = provider.generate_bulk_data(
            rows_payload=[{"Age": None, "Status": None}],
            context={},
            all_columns=["Age", "Status"],
            algorithm="Standard",
            instructions="Gen",
            temperature=0.3,
            rules_metadata=self.rules,
            enforce_rules=True
        )
        
        # Should have called twice (initial + 1 correction)
        self.assertEqual(mock_call.call_count, 2)
        self.assertEqual(result["data"][0]["Age"], 30)
        
        # Verify the feedback turn contained violation info
        correction_msg = mock_call.call_args_list[1][0][0][-1]["content"]
        self.assertIn("RULE VIOLATIONS DETECTED", correction_msg)
        self.assertIn("Value 70 is greater than maximum permitted 65", correction_msg)

if __name__ == "__main__":
    unittest.main()
