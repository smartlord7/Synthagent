import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import DiskcacheManager
from utils.logging_config import logger

# Diskcache to enable background callbacks for live progress bars
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    background_callback_manager=background_callback_manager,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "TabuLLM"
