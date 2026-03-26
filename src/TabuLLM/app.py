import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import app, logger
from components.layout import create_layout
import callbacks

# Assign layout to the app
app.layout = create_layout()
server = app.server  # Expose for Gunicorn

if __name__ == '__main__':
    logger.info("Starting TabuLLM Application...")
    # Bind to 0.0.0.0 and enable debug in prod/docker generally or use env variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8050))
    
    # Auto-detect if running inside Docker
    is_docker = os.path.exists('/.dockerenv')
    default_debug = "True" if not is_docker else "False"
    debug_mode = os.environ.get("DEBUG", default_debug).lower() == "true"
    
    app.run(host=host, port=port, debug=debug_mode)