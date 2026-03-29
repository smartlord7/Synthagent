import logging
import os

# Configure logging to both Console and a File
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("debug_ai_workspace.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TabuLLM")

logger = setup_logger()
