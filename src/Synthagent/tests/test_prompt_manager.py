import unittest
import os
from services.prompt_manager import PromptManager

class TestPromptManager(unittest.TestCase):
    def setUp(self):
        # The prompt manager expects a 'prompts' directory
        # We can test with the real one since it's read-only
        self.pm = PromptManager()

    def test_get_prompt_exists(self):
        # 'generation_final.j2' is a core prompt
        # We'll just check if it renders without error when given dummy data
        try:
            prompt = self.pm.get_prompt("generation_final.j2", 
                                        instructions="Test Instructions", 
                                        all_columns=["A", "B"], 
                                        is_empty=False,
                                        num_rows_to_generate=1,
                                        compact_payload_json="[]")
            self.assertTrue(len(prompt) > 0)
            self.assertIn("synthetic", prompt.lower())
        except Exception as e:
            self.fail(f"Prompt rendering failed: {e}")

    def test_get_prompt_extension_auto_add(self):
        # Should work without .j2 extension
        prompt = self.pm.get_prompt("generation_final", 
                                    instructions="Test Instructions", 
                                    all_columns=["A", "B"], 
                                    is_empty=False,
                                    num_rows_to_generate=1,
                                    compact_payload_json="[]")
        self.assertTrue(len(prompt) > 0)

    def test_invalid_template(self):
        from jinja2 import TemplateNotFound
        with self.assertRaises(TemplateNotFound):
            self.pm.get_prompt("non_existent_template_xyz")

if __name__ == "__main__":
    unittest.main()
