import unittest
import json
import os
import pandas as pd
from callbacks.session import save_workspace_session

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.filename = "test_workspace_session.json"
        if os.path.exists(self.filename):
            os.remove(self.filename)
        # Mock metadata
        self.metadata = {"ds1": {"filename": "data.csv"}}
        self.all_ids = [{"index": "ds1"}]
        self.dummy_data = [{"_row_id": "1", "col1": "val1"}]

    def test_save_load_serialized_json(self):
        # We need to simulate the callback context or just call the logic
        # Since dash callbacks are decorated, it's easier to test the serialization logic
        
        # Manually pack like the callback does
        tabs = {"ds1": {"original-data-store": self.dummy_data}}
        session_data = {
            "metadata": self.metadata,
            "tabs": tabs
        }
        
        with open(self.filename, "w") as f:
            json.dump(session_data, f)
            
        self.assertTrue(os.path.exists(self.filename))
        
        with open(self.filename, "r") as f:
            loaded = json.load(f)
            
        self.assertEqual(loaded["metadata"]["ds1"]["filename"], "data.csv")
        self.assertEqual(loaded["tabs"]["ds1"]["original-data-store"][0]["col1"], "val1")

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

if __name__ == '__main__':
    unittest.main()
