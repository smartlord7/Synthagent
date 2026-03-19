import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import app, logger
from components.layout import create_layout
import callbacks

# Assign layout to the app
app.layout = create_layout()

if __name__ == '__main__':
    logger.info("Starting TabuLLM Application...")
    app.run(debug=True)