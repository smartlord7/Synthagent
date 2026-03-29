import dash
from dash import Input, Output, State, MATCH
from core import app

@app.callback(
    Output({"type": "context-row-ids", "index": MATCH}, "data"),
    Output({"type": "target-row-ids", "index": MATCH}, "data"),
    Output({"type": "data-grid", "index": MATCH}, "rowData", allow_duplicate=True),
    Input({"type": "btn-mode-context", "index": MATCH}, "n_clicks"),
    Input({"type": "btn-mode-target", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "selectedRows"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "context-row-ids", "index": MATCH}, "data"),
    State({"type": "target-row-ids", "index": MATCH}, "data"),
    State({"type": "check-full-context", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def manage_selections_pm(btn_ctx, btn_tgt, selected_rows, current_data, ctx_ids, tgt_ids, full_ctx_val):
    ctx_ids = set(ctx_ids or [])
    tgt_ids = set(tgt_ids or [])
    is_full_ctx = len(full_ctx_val or []) > 0

    # Determine which button was clicked using dash.callback_context
    triggered_id = dash.callback_context.triggered[0]["prop_id"]
    is_context_btn = '"type":"btn-mode-context"' in triggered_id
    
    selected_uuids = [r["_row_id"] for r in selected_rows] if selected_rows else []

    # 1. Update the ID stores (Manual selections)
    for uuid in selected_uuids:
        if is_context_btn:
            if uuid in ctx_ids:
                ctx_ids.remove(uuid)
            else:
                ctx_ids.add(uuid)
                if uuid in tgt_ids: tgt_ids.remove(uuid)
        else:
            if uuid in tgt_ids:
                tgt_ids.remove(uuid)
            else:
                tgt_ids.add(uuid)
                if uuid in ctx_ids: ctx_ids.remove(uuid)

    # 2. Map visual flags back to rowData
    for row in current_data:
        row_id = row["_row_id"]
        row["is_target"] = row_id in tgt_ids
        if is_full_ctx:
            row["is_context"] = not row["is_target"]
        else:
            row["is_context"] = row_id in ctx_ids

    return list(ctx_ids), list(tgt_ids), current_data


@app.callback(
    Output({"type": "data-grid", "index": MATCH}, "rowData", allow_duplicate=True),
    Input({"type": "check-full-context", "index": MATCH}, "value"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "context-row-ids", "index": MATCH}, "data"),
    State({"type": "target-row-ids", "index": MATCH}, "data"),
    prevent_initial_call=True
)
def toggle_full_dataset_visual_pm(full_ctx_value, current_data, ctx_ids, tgt_ids):
    is_full_ctx = len(full_ctx_value or []) > 0
    ctx_ids_set = set(ctx_ids or [])

    for row in current_data:
        if is_full_ctx:
            row["is_context"] = not row.get("is_target", False)
        else:
            row["is_context"] = row.get("_row_id") in ctx_ids_set

    return current_data
