import unittest
import uuid
import pandas as pd
from callbacks.table import handle_multiple_uploads_pm, confirm_delete_dataset_pm

class TestCallbackFlow(unittest.TestCase):
    def setUp(self):
        self.dataset_id = str(uuid.uuid4())
        self.metadata = {self.dataset_id: {"filename": "original.csv"}}

    def test_handle_uploads_logic(self):
        # We need a base64 string for a CSV. 
        # For simplicity, we just check if it calls parse_uploaded_file correctly
        with unittest.mock.patch('callbacks.table.parse_uploaded_file') as p_mock:
            p_mock.return_value = pd.DataFrame({"a": [1], "b": [2]})
            
            # Simulated Dash Input
            contents = ["data:text/csv;base64,YWIsY2QKMSwyCg=="]
            filenames = ["test.csv"]
            current_tabs = []
            metadata = {}
            
            new_tabs, new_meta, last_id = handle_multiple_uploads_pm(contents, filenames, current_tabs, metadata)
            
            self.assertEqual(len(new_tabs), 1)
            self.assertIn(last_id, new_meta)
            self.assertEqual(new_meta[last_id]["filename"], "test.csv")

    def test_confirm_delete_logic(self):
        # Setup metadata and tabs
        metadata = {self.dataset_id: {"filename": "tobedeleted.csv"}}
        current_tabs = [{"props": {"value": self.dataset_id}}]
        current_value = self.dataset_id
        
        # confirm_delete_dataset_pm(n_clicks, delete_id, current_tabs, metadata, current_value)
        new_tabs, new_meta, new_value = confirm_delete_dataset_pm(1, self.dataset_id, current_tabs, metadata, current_value)
        
        self.assertEqual(len(new_tabs), 0)
        self.assertNotIn(self.dataset_id, new_meta)
        self.assertIsNone(new_value)

if __name__ == '__main__':
    unittest.main()
