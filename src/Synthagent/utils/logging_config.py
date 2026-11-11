import logging
import os
import tempfile

# Configure logging to both Console and a File (outside project root to avoid reloader)
def setup_logger():
    log_path = os.path.join(tempfile.gettempdir(), "debug_synthagent.log")
    print(f"DEBUG: Synthagent logging to {log_path}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Synthagent")

logger = setup_logger()
