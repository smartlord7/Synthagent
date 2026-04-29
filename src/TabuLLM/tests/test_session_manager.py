import unittest
import os
import shutil
import json
import base64
from services.session_manager import SessionManager, SESSION_ROOT

class TestSessionManager(unittest.TestCase):
    
    def setUp(self):
        # Use a temporary session root for testing
        self.original_root = SESSION_ROOT
        self.test_root = "test_sessions"
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
        os.makedirs(self.test_root)
        
        # Patch SESSION_ROOT in the class (it's a module level constant though)
        # We'll just change the class methods to use our test root if we can,
        # but since they are static, let's just use the real one and clean up?
        # Actually, let's just temporarily override the module constant.
        import services.session_manager
        services.session_manager.SESSION_ROOT = self.test_root

    def tearDown(self):
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
        import services.session_manager
        services.session_manager.SESSION_ROOT = self.original_root

    def test_get_dataset_hash(self):
        content = "a,b,c\n1,2,3"
        encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        contents = f"data:text/csv;base64,{encoded}"
        
        h1 = SessionManager.get_dataset_hash(contents)
        h2 = SessionManager.get_dataset_hash(contents)
        
        self.assertEqual(h1, h2)
        self.assertTrue(len(h1) > 0)

    def test_save_and_load_session(self):
        hash_id = "test_hash"
        session_id = "session_1"
        state = {"data": [1, 2, 3], "session_name": "Test Session"}
        
        SessionManager.save_session(hash_id, session_id, state)
        
        # Check if file exists
        session_path = os.path.join(self.test_root, hash_id, f"{session_id}.json")
        self.assertTrue(os.path.exists(session_path))
        
        # Load back
        loaded = SessionManager.load_session(hash_id, session_id)
        self.assertEqual(loaded["data"], [1, 2, 3])
        self.assertIn("modified_at", loaded)

    def test_list_sessions(self):
        hash_id = "list_hash"
        SessionManager.save_session(hash_id, "s1", {"session_name": "First"})
        SessionManager.save_session(hash_id, "s2", {"session_name": "Second"})
        
        sessions = SessionManager.list_sessions(hash_id)
        self.assertEqual(len(sessions), 2)
        
        names = [s["name"] for s in sessions]
        self.assertIn("First", names)
        self.assertIn("Second", names)

    def test_sync_global_session(self):
        metadata = {"ds1": {"filename": "f1.csv"}}
        SessionManager.sync_global_session(metadata)
        
        self.assertTrue(os.path.exists("workspace_session.json"))
        with open("workspace_session.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["ds1"]["filename"], "f1.csv")
            
        # Cleanup global session file
        if os.path.exists("workspace_session.json"):
            os.remove("workspace_session.json")

if __name__ == "__main__":
    unittest.main()
