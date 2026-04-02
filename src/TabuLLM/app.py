import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import app
from utils.logging_config import logger
from components.layout import create_layout
import callbacks

# Assign layout to the app
app.layout = create_layout
server = app.server  # Expose for Gunicorn

if __name__ == '__main__':
    # Bind to 0.0.0.0 and enable debug in prod/docker generally or use env variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8050))
    
    # Auto-detect if running inside Docker
    is_docker = os.path.exists('/.dockerenv')
    default_debug = "True" if not is_docker else "False"
    debug_mode = os.environ.get("DEBUG", default_debug).lower() == "true"
    
    logger.info(f"[SYS] Booting TabuLLM Server Instance...")
    logger.info(f"[SYS] Environment: {'Docker' if is_docker else 'Local'} | Host: {host} | Port: {port} | Dash Debug: {debug_mode}")
    
    try:
        app.run(host=host, port=port, debug=debug_mode, reloader_type="stat")
    except Exception as e:
        logger.error(f"[SYS] Fatal exception during application launch: {e}")
        sys.exit(1)
