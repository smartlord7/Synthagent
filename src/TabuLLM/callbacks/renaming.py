import dash
from dash import Input, Output, State, MATCH
from core import app

# --- IN-SITU RENAMING LOGIC ---
@app.callback(
    Output({"type": "quick-rename-input", "index": MATCH}, "value"),
    Output({"type": "active-col-store", "index": MATCH}, "data"),
    Input({"type": "data-grid", "index": MATCH}, "cellClicked"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    prevent_initial_call=True
)
def select_column_to_rename_pm(cell_clicked, col_defs):
    if not cell_clicked: return dash.no_update, dash.no_update
    col_id = cell_clicked["colId"]
    current_name = next((col.get("headerName", col_id) for col in col_defs if col["field"] == col_id), col_id)
    return current_name, col_id


@app.callback(
    Output({"type": "data-grid", "index": MATCH}, "columnDefs", allow_duplicate=True),
    Input({"type": "quick-rename-input", "index": MATCH}, "value"),
    State({"type": "active-col-store", "index": MATCH}, "data"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    prevent_initial_call=True
)
def apply_live_rename_pm(new_name, active_col_id, col_defs):
    if not new_name or not active_col_id: return dash.no_update
    updated_cols = list(col_defs)
    for col in updated_cols:
        if col["field"] == active_col_id:
            col["headerName"] = new_name
            break
    return updated_cols


@app.callback(
    Output({"type": "data-grid", "index": MATCH}, "columnDefs", allow_duplicate=True),
    Input({"type": "btn-remove-col", "index": MATCH}, "n_clicks"),
    State({"type": "active-col-store", "index": MATCH}, "data"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    prevent_initial_call=True
)
def remove_live_column_pm(n_clicks, active_col_id, col_defs):
    if not n_clicks or not active_col_id: return dash.no_update
    if active_col_id in ['ai_audit', '_row_id']:
        return dash.no_update
        
    updated_cols = [col for col in col_defs if col["field"] != active_col_id]
    return updated_cols
