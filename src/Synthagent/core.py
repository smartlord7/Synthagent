import dash
import dash_bootstrap_components as dbc
import diskcache
import tempfile
import os
from dash import DiskcacheManager
from utils.logging_config import logger

# Diskcache to enable background callbacks (outside project root to avoid reloader)
cache_path = os.path.join(tempfile.gettempdir(), "synthagent_cache")
cache = diskcache.Cache(cache_path)
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    background_callback_manager=background_callback_manager,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Synthagent"
