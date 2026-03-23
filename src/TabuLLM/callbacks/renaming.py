import dash
from dash import Input, Output, State
from core import app

# --- IN-SITU RENAMING LOGIC ---
@app.callback(
    Output("quick-rename-input", "value"),
    Output("active-col-store", "data"),
    Input("data-grid", "cellClicked"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def select_column_to_rename(cell_clicked, col_defs):
    if not cell_clicked: return dash.no_update, dash.no_update
    col_id = cell_clicked["colId"]
    current_name = next((col.get("headerName", col_id) for col in col_defs if col["field"] == col_id), col_id)
    return current_name, col_id


@app.callback(
    Output("data-grid", "columnDefs", allow_duplicate=True),
    Input("quick-rename-input", "value"),
    State("active-col-store", "data"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def apply_live_rename(new_name, active_col_id, col_defs):
    if not new_name or not active_col_id: return dash.no_update
    updated_cols = list(col_defs)
    for col in updated_cols:
        if col["field"] == active_col_id:
            col["headerName"] = new_name
            break
    return updated_cols


@app.callback(
    Output("data-grid", "columnDefs", allow_duplicate=True),
    Input("btn-remove-col", "n_clicks"),
    State("active-col-store", "data"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def remove_live_column(n_clicks, active_col_id, col_defs):
    if not n_clicks or not active_col_id: return dash.no_update
    # Skip deleting internal management columns if any
    if active_col_id in ['ai_audit', '_row_id']:
        return dash.no_update
        
    updated_cols = [col for col in col_defs if col["field"] != active_col_id]
    return updated_cols

