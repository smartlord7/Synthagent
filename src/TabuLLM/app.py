from core import app, logger
from components.layout import create_layout
import callbacks

# Assign layout to the app
app.layout = create_layout()

if __name__ == '__main__':
    logger.info("Starting TabuLLM Application...")
    app.run(debug=True)