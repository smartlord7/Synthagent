import uuid
import dash
from dash import Input, Output, State, MATCH
from core import app
from utils.logging_config import logger
from components.dataset_tab import create_dataset_tab
from services.data_manager import parse_uploaded_file, df_to_aggrid_format
from utils.constants import ai_highlight_style

@app.callback(
    Output("tabs-datasets", "children"),
    Output("datasets-metadata-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("tabs-datasets", "children"),
    State("datasets-metadata-store", "data"),
    prevent_initial_call=True
)
def handle_multiple_uploads_pm(list_of_contents, list_of_filenames, current_tabs, metadata):
    if not list_of_contents:
        return dash.no_update, dash.no_update

    metadata = metadata or {}
    new_tabs = list(current_tabs or [])
    
    for contents, filename in zip(list_of_contents, list_of_filenames):
        dataset_id = str(uuid.uuid4())
        try:
            df = parse_uploaded_file(contents, filename)
            if df is not None and not df.empty:
                new_tab = create_dataset_tab(dataset_id, filename, df=df)
                new_tabs.append(new_tab)
                
                metadata[dataset_id] = {
                    "filename": filename,
                    "status": "Ready",
                    "rows": df.shape[0],
                    "cols": df.shape[1]
                }
            else:
                logger.error(f"Failed to parse {filename}: Empty or invalid.")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    return new_tabs, metadata


@app.callback(
    Output({"type": "data-grid", "index": MATCH}, "rowData", allow_duplicate=True),
    Input({"type": "btn-add-row", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    prevent_initial_call=True
)
def add_row_pm(n_clicks, row_data):
    if not n_clicks: return dash.no_update
    new_row = {"_row_id": str(uuid.uuid4()), "is_context": False, "is_target": False, "ai_audit": ""}
    return row_data + [new_row]


@app.callback(
    Output({"type": "data-grid", "index": MATCH}, "columnDefs", allow_duplicate=True),
    Input({"type": "btn-add-col", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    prevent_initial_call=True
)
def add_column_pm(n_clicks, col_defs):
    if not n_clicks: return dash.no_update
    new_col_id = f"new_col_{len(col_defs)}"
    new_col = {"field": new_col_id, "headerName": "New Column", "editable": True, "cellStyle": ai_highlight_style}
    return col_defs + [new_col]
