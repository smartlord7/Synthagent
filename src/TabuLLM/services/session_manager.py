import os
import json
import hashlib
import base64
from datetime import datetime
from utils.logging_config import logger

SESSION_ROOT = "dataset_sessions"

class SessionManager:
    @staticmethod
    def get_dataset_hash(contents: str) -> str:
        """
        Generates a SHA-256 hash of the uploaded file content.
        The content is expected to be a base64 encoded string from dcc.Upload.
        """
        if not contents:
            return ""
        try:
            # dcc.Upload content format: "data:text/csv;base64,..."
            # We want the part after the comma
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            return hashlib.sha256(decoded).hexdigest()
        except Exception as e:
            logger.error(f"[SESSION] Failed to compute dataset hash: {e}")
            return ""

    @staticmethod
    def get_session_dir(hash_id: str) -> str:
        path = os.path.join(SESSION_ROOT, hash_id)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def list_sessions(hash_id: str):
        """Returns all sessions found for this dataset hash."""
        session_dir = SessionManager.get_session_dir(hash_id)
        sessions = []
        for f in os.listdir(session_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(session_dir, f), 'r') as file:
                        data = json.load(file)
                        sessions.append({
                            "id": f.replace(".json", ""),
                            "name": data.get("session_name", "Unnamed Session"),
                            "created_at": data.get("created_at", "N/A"),
                            "modified_at": data.get("modified_at", "N/A"),
                            "row_count": data.get("row_count", 0),
                            "synthetic_count": data.get("synthetic_count", 0)
                        })
                except:
                    pass
        # Sort by modified_at desc
        sessions.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
        return sessions

    @staticmethod
    def save_session(hash_id: str, session_id: str, state: dict):
        """Saves a specific session state to the hash-grouped directory."""
        session_dir = SessionManager.get_session_dir(hash_id)
        session_path = os.path.join(session_dir, f"{session_id}.json")
        
        # Inject metadata if not present
        if "created_at" not in state:
            state["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["modified_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(session_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"[SESSION] Session {session_id} saved for dataset {hash_id[:8]}...")
        except Exception as e:
            logger.error(f"[SESSION] Failed to save session {session_id}: {e}")

    @staticmethod
    def load_session(hash_id: str, session_id: str):
        """Loads a specific session state."""
        session_dir = SessionManager.get_session_dir(hash_id)
        session_path = os.path.join(session_dir, f"{session_id}.json")
        if not os.path.exists(session_path):
            return None
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[SESSION] Failed to load session {session_id}: {e}")
            return None

    @staticmethod
    def sync_global_session(metadata, tabs_data=None, pipeline_results=None, pipeline_config=None, report_config=None):
        """
        Manually syncs the workspace_session.json file.
        Used for immediate persistence when tabs are closed.
        """
        session_file = "workspace_session.json"
        
        # Load existing if not provided
        current_data = {}
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    current_data = json.load(f)
            except:
                pass
        
        session_data = {
            "metadata": metadata if metadata is not None else current_data.get("metadata", {}),
            "pipeline_results": pipeline_results if pipeline_results is not None else current_data.get("pipeline_results", {}),
            "pipeline_config": pipeline_config if pipeline_config is not None else current_data.get("pipeline_config", {}),
            "report_config": report_config if report_config is not None else current_data.get("report_config", {}),
            "tabs": tabs_data if tabs_data is not None else current_data.get("tabs", {})
        }
        
        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f)
            logger.info(f"[SESSION] Global workspace synced ({len(session_data['metadata'])} active datasets).")
        except Exception as e:
            logger.error(f"[SESSION] Manual global sync failed: {e}")
