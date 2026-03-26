import sys
import os
import json
import unittest

# Add the current directory to path (assuming we run from src/TabuLLM)
sys.path.append(os.getcwd())

from utils.json_utils import robust_json_loads

class TestJsonUtils(unittest.TestCase):
    
    def test_pure_json(self):
        data = {"key": "value", "list": [1, 2, 3]}
        text = json.dumps(data)
        self.assertEqual(robust_json_loads(text), data)

    def test_markdown_json(self):
        data = {"data": [{"a": 1}]}
        text = f"```json\n{json.dumps(data)}\n```"
        self.assertEqual(robust_json_loads(text), data)

    def test_markdown_no_lang(self):
        data = {"data": [{"a": 1}]}
        text = f"```\n{json.dumps(data)}\n```"
        self.assertEqual(robust_json_loads(text), data)

    def test_conversational_padding(self):
        data = {"result": "success"}
        text = f"Here is the JSON you requested:\n{json.dumps(data)}\nI hope this is what you needed."
        self.assertEqual(robust_json_loads(text), data)

    def test_nested_markdown(self):
        data = {"items": [1, 2]}
        text = "The model said:\n```json\n{\"items\": [1, 2]}\n```\nWhich is correct."
        self.assertEqual(robust_json_loads(text), data)

    def test_nan_fix(self):
        text = '{"value": NaN, "other": inf}'
        expected = {"value": None, "other": None}
        self.assertEqual(robust_json_loads(text), expected)

    def test_trailing_comma_fix(self):
        text = '{"list": [1, 2, ], "obj": {"a": 1, }}'
        expected = {"list": [1, 2], "obj": {"a": 1}}
        self.assertEqual(robust_json_loads(text), expected)

    def test_malformed_fails_with_context(self):
        text = '{"a": 1,\n "b": [1, 2, 3\n "c": 3}'
        from utils.json_utils import RobustJSONError
        with self.assertRaises(RobustJSONError) as cm:
            robust_json_loads(text)
        self.assertIn("context", cm.exception.error_data)
        self.assertIn('"b": [1, 2, 3', cm.exception.error_data["context"])

if __name__ == "__main__":
    unittest.main()
