import dash
import dash_bootstrap_components as dbc
import logging

# Configure logging to both Console and a File
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debug_ai_workspace.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import diskcache
from dash import DiskcacheManager

# Diskcache to enable background callbacks for live progress bars
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    background_callback_manager=background_callback_manager,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP]
)
app.title = "TabuLLM"

