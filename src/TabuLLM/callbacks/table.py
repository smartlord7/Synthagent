import uuid
import json
import dash
from dash import Input, Output, State, ALL, MATCH, html
import dash_bootstrap_components as dbc
from core import app
from utils.logging_config import logger
from components.dataset_tab import create_dataset_tab
from services.data_manager import parse_uploaded_file, df_to_aggrid_format
from utils.constants import ai_highlight_style

@app.callback(
    Output("tabs-datasets", "children"),
    Output("datasets-metadata-store", "data"),
    Output("tabs-datasets", "value", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("tabs-datasets", "children"),
    State("datasets-metadata-store", "data"),
    prevent_initial_call=True
)
def handle_multiple_uploads_pm(list_of_contents, list_of_filenames, current_tabs, metadata):
    if not list_of_contents:
        return dash.no_update, dash.no_update, dash.no_update

    metadata = metadata or {}
    new_tabs = list(current_tabs or [])
    last_id = None
    
    logger.info(f"[IO] Processing {len(list_of_contents)} upload(s)...")
    for contents, filename in zip(list_of_contents, list_of_filenames):
        dataset_id = str(uuid.uuid4())
        try:
            df = parse_uploaded_file(contents, filename)
            if df is not None and not df.empty:
                logger.info(f"[IO] Successfully parsed '{filename}' -> {df.shape[0]} rows, {df.shape[1]} cols.")
                new_tab = create_dataset_tab(dataset_id, filename, df=df)
                new_tabs.append(new_tab)
                
                metadata[dataset_id] = {
                    "filename": filename,
                    "status": "Ready",
                    "rows": df.shape[0],
                    "cols": df.shape[1]
                }
                last_id = dataset_id
            else:
                logger.warning(f"[IO] Failed to parse '{filename}': Empty or invalid format.")
        except Exception as e:
            logger.error(f"[IO] Error processing upload '{filename}': {e}")

    return new_tabs, metadata, last_id if last_id else dash.no_update


@app.callback(
    Output("delete-dataset-modal", "is_open"),
    Output("delete-dataset-name", "children"),
    Output("pending-delete-id-store", "data"),
    Input({"type": "btn-delete-dataset", "index": ALL}, "n_clicks"),
    Input("btn-cancel-delete", "n_clicks"),
    Input("btn-confirm-delete", "n_clicks"),
    State("datasets-metadata-store", "data"),
    State("delete-dataset-modal", "is_open"),
    prevent_initial_call=True
)
def manage_delete_modal_pm(n_delete, n_cancel, n_confirm, metadata, is_open):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update
    
    trig_id = ctx.triggered[0]["prop_id"]
    trig_value = ctx.triggered[0]["value"]
    
    if "btn-delete-dataset" in trig_id:
        if not trig_value: return dash.no_update
        # Pattern-matched ID is a JSON string in prop_id
        try:
            btn_id_json = trig_id.split(".")[0]
            btn_id = json.loads(btn_id_json)
            idx = btn_id["index"]
            fname = metadata.get(idx, {}).get("filename", "Unknown")
            logger.info(f"[UI] User initiated deletion of dataset '{fname}' ({idx}).")
            return True, f"Dataset: {fname}", idx
        except Exception as e:
            logger.error(f"[UI] Error opening delete modal: {e}")
            return dash.no_update
        
    if "btn-cancel-delete" in trig_id or "btn-confirm-delete" in trig_id:
        return False, dash.no_update, dash.no_update
        
    return dash.no_update


@app.callback(
    Output("tabs-datasets", "children", allow_duplicate=True),
    Output("datasets-metadata-store", "data", allow_duplicate=True),
    Output("tabs-datasets", "value", allow_duplicate=True),
    Input("btn-confirm-delete", "n_clicks"),
    State("pending-delete-id-store", "data"),
    State("tabs-datasets", "children"),
    State("datasets-metadata-store", "data"),
    State("tabs-datasets", "value"),
    prevent_initial_call=True
)
def confirm_delete_dataset_pm(n_clicks, delete_id, current_tabs, metadata, current_value):
    if not n_clicks or not delete_id:
        return dash.no_update, dash.no_update, dash.no_update
    
    # 1. Update metadata
    if delete_id in metadata:
        fname = metadata[delete_id].get("filename", "Unknown")
        del metadata[delete_id]
        logger.info(f"[UI] Dataset '{fname}' confirmed and removed from workspace.")
        
    # 2. Update tabs list
    new_tabs = []
    for tab in current_tabs:
        if isinstance(tab, dict):
            if tab.get("props", {}).get("value") != delete_id:
                new_tabs.append(tab)
        elif hasattr(tab, "value"):
            if tab.value != delete_id:
                new_tabs.append(tab)
        else:
            new_tabs.append(tab)
            
    # 3. Switch active tab if we just deleted the current one
    new_value = current_value
    if delete_id == current_value:
        if new_tabs:
            first_tab = new_tabs[0]
            if isinstance(first_tab, dict):
                new_value = first_tab.get("props", {}).get("value")
            else:
                new_value = getattr(first_tab, "value", None)
            logger.info(f"[UI] Switched active view to {new_value}.")
        else:
            new_value = None
            logger.info(f"[UI] All datasets cleared. Workbench empty.")

    return new_tabs, metadata, new_value


@app.callback(
    Output({"type": "data-grid", "index": MATCH}, "rowData", allow_duplicate=True),
    Input({"type": "btn-add-row", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    prevent_initial_call=True
)
def add_row_pm(n_clicks, row_data):
    if not n_clicks: return dash.no_update
    new_row = {"_row_id": str(uuid.uuid4()), "is_context": False, "is_target": False, "ai_audit": ""}
    logger.info(f"[UI] Added new manual row to current grid (Total: {len(row_data) + 1}).")
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


@app.callback(
    Output({"type": "inspect-modal", "index": MATCH}, "is_open"),
    Output({"type": "inspect-modal-body", "index": MATCH}, "children"),
    Input({"type": "btn-inspect-row", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "selectedRows"),
    State({"type": "algo-selector", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def inspect_selected_rows_pm(n_clicks, selected_rows, algo):
    if not n_clicks or not selected_rows:
        return dash.no_update, dash.no_update
    
    batches = {}
    for row in selected_rows:
        audit = row.get('ai_audit', '')
        if not audit: 
            continue
        if audit not in batches:
            batches[audit] = []
        batches[audit].append(row)
        
    if not batches:
        return True, dbc.Alert("Selected rows are original (real) data and have no generative audit.", color="info", className="m-0")
        
    children = []
    
    for audit_str, rows in batches.items():
        parts = audit_str.split(" | ")
        
        # Determine contents based on legacy format or new format
        if len(parts) >= 4:
            model = parts[1]
            tokens = parts[2]
            ts = parts[3]
        elif len(parts) == 3:
            model = algo or "Unknown"
            tokens = parts[1]
            ts = parts[2]
        else:
            model = algo or "Unknown"
            tokens = "N/A"
            ts = audit_str.replace("Gen AI", "").strip(" |") if audit_str != "Gen AI" else "N/A"
            
        details = [
            html.Div([
                html.Strong("Timestamp: ", className="text-muted"), html.Span(ts, className="ms-1"), html.Br(),
                html.Strong("Tokens Used: ", className="text-muted"), html.Span(tokens, className="ms-1"), html.Br(),
                html.Strong("Model: ", className="text-muted"), html.Span(model, className="ms-1"), html.Br(),
                html.Strong("Rows count: ", className="text-muted"), html.Span(str(len(rows)), className="ms-1 fw-bold text-info"),
            ], className="small mb-3"),
            html.Pre(json.dumps([r for r in rows], indent=2), 
                     className="bg-dark text-light p-2 rounded border border-secondary",
                     style={"maxHeight": "250px", "overflowY": "auto", "fontSize": "0.75rem"})
        ]
        
        children.append(dbc.Card([
            dbc.CardHeader([html.I(className="bi bi-box me-2"), f"Batch: {ts}"], className="bg-transparent border-secondary fw-bold small"),
            dbc.CardBody(details, className="p-3")
        ], className="mb-3 bg-transparent border-secondary shadow-sm"))
        
    return True, html.Div(children)
