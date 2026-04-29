import unittest
from services.rule_validator import RuleValidator

class TestRuleValidator(unittest.TestCase):
    def test_dynamic_model_numeric_range(self):
        rules = [
            {"name": "age", "type": "number", "level": 1},
            {"name": "age", "type": "number", "level": 2, "min": 18, "max": 100}
        ]
        
        Model = RuleValidator.create_pydantic_model(rules)
        self.assertIsNotNone(Model)
        
        # Valid
        Model(age=25)
        
        # Invalid min
        with self.assertRaises(ValueError):
            Model(age=10)
            
        # Invalid max
        with self.assertRaises(ValueError):
            Model(age=150)

    def test_dynamic_model_categorical_enum(self):
        rules = [
            {"name": "color", "type": "category", "level": 1},
            {"name": "color", "type": "category", "level": 2, "text": "red, green, blue"}
        ]
        
        Model = RuleValidator.create_pydantic_model(rules)
        
        # Valid
        Model(color="red")
        
        # Invalid
        with self.assertRaises(ValueError):
            Model(color="yellow")

    def test_validate_batch_with_violations(self):
        rules = [{"name": "score", "type": "number", "level": 2, "min": 0, "max": 10}]
        Model = RuleValidator.create_pydantic_model(rules)
        
        data = [
            {"score": 5},
            {"score": 15}, # Violation
            {"score": -1}  # Violation
        ]
        
        violations = RuleValidator.validate_batch(data, Model)
        self.assertEqual(len(violations), 2)
        self.assertEqual(violations[0]["row_index"], 1)
        self.assertEqual(violations[1]["row_index"], 2)
        
        # Test formatting
        feedback = RuleValidator.format_violations_for_llm(violations)
        self.assertIn("RULE VIOLATIONS DETECTED", feedback)
        self.assertIn("Row 1", feedback)
        self.assertIn("Row 2", feedback)

if __name__ == "__main__":
    unittest.main()
