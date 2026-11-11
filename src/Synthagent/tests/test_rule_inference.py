import unittest
import pandas as pd
import numpy as np
from services.rule_inference import RuleInference

class TestRuleInference(unittest.TestCase):
    def test_infer_dataset_rules(self):
        df = pd.DataFrame({
            "age": [20, 30, 40],
            "gender": ["M", "F", "F"],
            "income": [20000.5, 30000.0, 40000.2]
        })
        
        rules = RuleInference.infer_dataset_rules(df)
        
        # 3 Level 1 rules (one for each col)
        # 3 Level 2 rules (ranges for age/income, categories for gender)
        self.assertEqual(len(rules), 6)
        
        # Check Integrity Rules
        integrity_rules = [r for r in rules if r["level"] == 1]
        self.assertEqual(len(integrity_rules), 3)
        self.assertIn("age", [r["name"] for r in integrity_rules])
        
        # Check Boundary Rules
        boundary_rules = [r for r in rules if r["level"] == 2]
        self.assertEqual(len(boundary_rules), 3)
        
        # Check age range
        age_rule = [r for r in boundary_rules if r["name"] == "age"][0]
        self.assertEqual(age_rule["min"], 20)
        self.assertEqual(age_rule["max"], 40)
        
        # Check gender categories
        gender_rule = [r for r in boundary_rules if r["name"] == "gender"][0]
        self.assertEqual(gender_rule["text"], "F, M")

    def test_infer_rules_with_nans(self):
        df = pd.DataFrame({
            "val": [1, np.nan, 10]
        })
        rules = RuleInference.infer_dataset_rules(df)
        boundary_rule = [r for r in rules if r["level"] == 2][0]
        # Should ignore NaN
        self.assertEqual(boundary_rule["min"], 1)
        self.assertEqual(boundary_rule["max"], 10)

if __name__ == "__main__":
    unittest.main()
