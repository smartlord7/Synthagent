import dash
from dash import Input, Output, State
from core import app

@app.callback(
    Output("context-row-ids", "data"),
    Output("target-row-ids", "data"),
    Output("data-grid", "rowData", allow_duplicate=True),
    Input("btn-mode-context", "n_clicks"),
    Input("btn-mode-target", "n_clicks"),
    State("data-grid", "selectedRows"),
    State("data-grid", "rowData"),
    State("context-row-ids", "data"),
    State("target-row-ids", "data"),
    State("check-full-context", "value"),  # Added this state
    prevent_initial_call=True
)
def manage_selections(btn_ctx, btn_tgt, selected_rows, current_data, ctx_ids, tgt_ids, full_ctx_val):
    ctx_ids = set(ctx_ids or [])
    tgt_ids = set(tgt_ids or [])
    is_full_ctx = len(full_ctx_val or []) > 0

    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    selected_uuids = [r["_row_id"] for r in selected_rows] if selected_rows else []

    # 1. Update the ID stores (Manual selections)
    for uuid in selected_uuids:
        if triggered_id == "btn-mode-context":
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

        # Target logic is always the same (Red)
        row["is_target"] = row_id in tgt_ids

        # Context logic (Blue) now respects the global switch
        if is_full_ctx:
            # If global switch is ON, show blue unless it's a target
            row["is_context"] = not row["is_target"]
        else:
            # If global switch is OFF, only show blue if manually selected
            row["is_context"] = row_id in ctx_ids

    return list(ctx_ids), list(tgt_ids), current_data


@app.callback(
    Output("data-grid", "rowData", allow_duplicate=True),
    Input("check-full-context", "value"),
    State("data-grid", "rowData"),
    State("context-row-ids", "data"),
    State("target-row-ids", "data"),
    prevent_initial_call=True
)
def toggle_full_dataset_visual(full_ctx_value, current_data, ctx_ids, tgt_ids):
    is_full_ctx = len(full_ctx_value) > 0
    ctx_ids_set = set(ctx_ids or [])

    for row in current_data:
        # If full context is ON, everyone is blue unless they are specifically a red target
        if is_full_ctx:
            row["is_context"] = not row.get("is_target", False)
        else:
            # Revert to only the manually selected IDs
            row["is_context"] = row["_row_id"] in ctx_ids_set

    return current_data
