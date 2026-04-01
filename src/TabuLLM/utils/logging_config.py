import logging
import os
import tempfile

# Configure logging to both Console and a File (outside project root to avoid reloader)
def setup_logger():
    log_path = os.path.join(tempfile.gettempdir(), "debug_ai_workspace.log")
    print(f"DEBUG: TabuLLM logging to {log_path}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TabuLLM")

logger = setup_logger()
