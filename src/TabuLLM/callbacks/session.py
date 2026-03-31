import json
import dash
from dash import Input, Output, State, ALL
from core import app

@app.callback(
    Output("session-toast-container", "children"),
    Input("btn-save-session", "n_clicks"),
    State("datasets-metadata-store", "data"),
    State("pipeline-results-store", "data"),
    State({"type": "original-data-store", "index": ALL}, "data"),
    State({"type": "data-grid", "index": ALL}, "rowData"),
    State({"type": "dataset-description-store", "index": ALL}, "data"),
    State({"type": "target-cols-dropdown", "index": ALL}, "value"),
    State({"type": "algo-selector", "index": ALL}, "value"),
    State({"type": "param-tone", "index": ALL}, "value"),
    State({"type": "param-consistency", "index": ALL}, "value"),
    State({"type": "temperature-slider", "index": ALL}, "value"),
    State({"type": "gen-chunk-size", "index": ALL}, "value"),
    State({"type": "gen-max-context", "index": ALL}, "value"),
    State({"type": "gen-context-strategy", "index": ALL}, "value"),
    State({"type": "column-roles-store", "index": ALL}, "data"),
    State({"type": "ml-config-store", "index": ALL}, "data"),
    State({"type": "bench-state-store", "index": ALL}, "data"),
    State("pipeline-config-store", "data"),
    State({"type": "original-data-store", "index": ALL}, "id"),
    prevent_initial_call=True
)
def save_workspace_session(n_clicks, metadata, pipeline_results,
                           all_orig, all_grid, all_desc, all_targets, all_algo, all_tone, all_consistency,
                           all_temp, all_chunk, all_max_ctx, all_strategy,
                           all_roles, all_ml_config, all_bench, 
                           pipeline_config, all_ids):
    if not n_clicks: return dash.no_update
    
    # Pack tabs data
    tabs = {}
    for i, _id_obj in enumerate(all_ids):
        d_id = _id_obj["index"]
        tabs[d_id] = {
            "original-data-store": all_orig[i] if i < len(all_orig) else None,
            "data-grid": all_grid[i] if i < len(all_grid) else None,
            "dataset-description-store": all_desc[i] if i < len(all_desc) else "",
            "target-cols-dropdown": all_targets[i] if i < len(all_targets) else [],
            "algo-selector": all_algo[i] if i < len(all_algo) else "google/gemma-3-4b",
            "param-tone": all_tone[i] if i < len(all_tone) else "Concise / Standard",
            "param-consistency": all_consistency[i] if i < len(all_consistency) else "Strict (Perfectly uniform)",
            "temperature-slider": all_temp[i] if i < len(all_temp) else 0.3,
            "gen-chunk-size": all_chunk[i] if i < len(all_chunk) else 30,
            "gen-max-context": all_max_ctx[i] if i < len(all_max_ctx) else 60,
            "gen-context-strategy": all_strategy[i] if i < len(all_strategy) else "random",
            "column-roles-store": all_roles[i] if i < len(all_roles) else {"targets": []},
            "ml-config-store": all_ml_config[i] if i < len(all_ml_config) else {},
            "bench-state-store": all_bench[i] if i < len(all_bench) else {"active": False, "k_total": 0, "k_current": 0, "fraction": 0.2, "results": []}
        }
        
    session_data = {
        "metadata": metadata or {},
        "pipeline_results": pipeline_results or {},
        "pipeline_config": pipeline_config or {},
        "tabs": tabs
    }
    
    try:
        with open("workspace_session.json", "w") as f:
            json.dump(session_data, f)
        
        import dash_bootstrap_components as dbc
        from dash import html
        return dbc.Alert([html.I(className="bi bi-check-circle-fill me-2"), "Session completely synced!"], 
                         color="success", duration=3000, className="mb-0 py-1 px-3 small shadow-sm")
    except Exception as e:
        import dash_bootstrap_components as dbc
        return dbc.Alert(f"Failed to sync: {e}", color="danger", duration=4000)
