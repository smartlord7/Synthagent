import dash
from dash import Input, Output, State, ALL, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from core import app
from utils.logging_config import logger
from services.pipeline_manager import PipelineManager
from callbacks.eda import get_eda_layout

@app.callback(
    Output("pipeline-modal", "is_open"),
    Input("btn-open-pipeline", "n_clicks"),
    Input("btn-close-pipeline", "n_clicks"),
    State("pipeline-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_pipeline_modal(n_open, n_close, is_open):
    if n_open or n_close:
        return not is_open
    return is_open

@app.callback(
    Output("pipeline-modal-body", "children"),
    Input("btn-open-pipeline", "n_clicks"),
    State("datasets-metadata-store", "data"),
    prevent_initial_call=True
)
def render_pipeline_ui_in_modal(n_clicks, metadata):
    if not metadata:
        return dbc.Alert("No datasets uploaded. Upload files to start a pipeline.", color="info", className="m-0")
    
    rows = []
    # Sort by filename or ID for stable display
    for dataset_id in sorted(metadata.keys()):
        info = metadata[dataset_id]
        if dataset_id == "default-dataset" and info.get("rows", 0) <= 10:
             if info.get("rows", 0) <= 0 or info.get("filename") == "Workbench":
                 continue

        size_str = f"{info.get('rows', 0)}R × {info.get('cols', 0)}C"
        
        rows.append(dbc.ListGroupItem([
            dbc.Row([
                dbc.Col(html.Span(info["filename"], className="fw-bold text-info small"), width=5),
                dbc.Col(html.Small(size_str, className="text-muted d-block", style={"fontSize": "0.75rem"}), width=3),
                dbc.Col([
                    dbc.Input(id={"type": "pipeline-priority", "index": dataset_id}, type="number", value=1, size="sm", style={"width": "45px", "textAlign": "center"}),
                    dbc.Checkbox(id={"type": "pipeline-include", "index": dataset_id}, value=True, className="ms-2")
                ], width=4, className="d-flex align-items-center justify-content-end"),
            ], align="center", className="gx-2")
        ], className="bg-dark border-secondary text-light py-2"))

    if not rows:
        return dbc.Alert("No valid datasets found for pipeline.", color="warning", className="m-0")

    ui = html.Div([
        html.P("Execute parallel validation for selected datasets (Lower priority runs first).", className="text-muted small mb-3"),
        
        dbc.ListGroup(rows, className="mb-3 border-secondary"),
        
        dbc.Row([
            dbc.Col(dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run Pipeline"], 
                              id="btn-run-pipeline", color="success", size="sm", className="w-100 rounded-pill shadow-sm"), width=12),
        ], className="mb-3"),
        
        dbc.Spinner(html.Div(id="pipeline-status", className="text-center small text-info"), size="sm"),
        html.Hr(className="border-secondary my-2"),
        html.Div(id="pipeline-results-container", className="mt-2")
    ], className="px-1")
    return ui


@app.callback(
    Output("pipeline-results-container", "children"),
    Output("pipeline-status", "children"),
    Output({"type": "eda-dashboard-container", "index": ALL}, "children"),
    Input("btn-run-pipeline", "n_clicks"),
    State("datasets-metadata-store", "data"),
    # Modal-specific states (one per row in the modal list)
    State({"type": "pipeline-include", "index": ALL}, "value"),
    State({"type": "pipeline-include", "index": ALL}, "id"),
    State({"type": "pipeline-priority", "index": ALL}, "value"),
    # Tab-specific states (one per and all dataset tabs)
    State({"type": "original-data-store", "index": ALL}, "data"),
    State({"type": "original-data-store", "index": ALL}, "id"),
    State({"type": "data-grid", "index": ALL}, "columnDefs"),
    State({"type": "data-grid", "index": ALL}, "id"),
    State({"type": "bench-folds", "index": ALL}, "value"),
    State({"type": "bench-folds", "index": ALL}, "id"),
    State({"type": "gen-max-context", "index": ALL}, "value"),
    State({"type": "gen-max-context", "index": ALL}, "id"),
    State({"type": "gen-context-strategy", "index": ALL}, "value"),
    State({"type": "gen-context-strategy", "index": ALL}, "id"),
    State({"type": "algo-selector", "index": ALL}, "value"),
    State({"type": "algo-selector", "index": ALL}, "id"),
    State({"type": "param-tone", "index": ALL}, "value"),
    State({"type": "param-tone", "index": ALL}, "id"),
    State({"type": "param-consistency", "index": ALL}, "value"),
    State({"type": "param-consistency", "index": ALL}, "id"),
    State({"type": "param-uniqueness", "index": ALL}, "value"),
    State({"type": "param-uniqueness", "index": ALL}, "id"),
    State({"type": "temperature-slider", "index": ALL}, "value"),
    State({"type": "temperature-slider", "index": ALL}, "id"),
    State({"type": "gen-chunk-size", "index": ALL}, "value"),
    State({"type": "gen-chunk-size", "index": ALL}, "id"),
    State({"type": "dataset-description-store", "index": ALL}, "data"),
    State({"type": "dataset-description-store", "index": ALL}, "id"),
    # EDA containers (for all tabs)
    State({"type": "eda-dashboard-container", "index": ALL}, "id"),
    prevent_initial_call=True
)
def execute_pipeline_pm(n_clicks, metadata, 
                        inc_vals, inc_ids, prios,
                        orig_data_list, orig_ids,
                        col_defs_list, col_ids,
                        folds_list, folds_ids,
                        max_ctx_list, max_ctx_ids,
                        strategy_list, strategy_ids,
                        algo_list, algo_ids,
                        tone_list, tone_ids,
                        consist_list, consist_ids,
                        uniq_list, uniq_ids,
                        temp_list, temp_ids,
                        chunk_list, chunk_ids,
                        desc_list, desc_ids,
                        eda_ids):
    
    if not n_clicks: return dash.no_update, dash.no_update, [dash.no_update] * len(eda_ids)
    
    # helper to map ALL states back to dataset_id
    def map_states(values, ids):
        return {id_obj["index"]: values[i] for i, id_obj in enumerate(ids)}

    # Map all tab-specific config by dataset_id
    orig_map = map_states(orig_data_list, orig_ids)
    col_map = map_states(col_defs_list, col_ids)
    folds_map = map_states(folds_list, folds_ids)
    max_ctx_map = map_states(max_ctx_list, max_ctx_ids)
    strategy_map = map_states(strategy_list, strategy_ids)
    algo_map = map_states(algo_list, algo_ids)
    tone_map = map_states(tone_list, tone_ids)
    consist_map = map_states(consist_list, consist_ids)
    uniq_map = map_states(uniq_list, uniq_ids)
    temp_map = map_states(temp_list, temp_ids)
    chunk_map = map_states(chunk_list, chunk_ids)
    desc_map = map_states(desc_list, desc_ids)

    # 1. Build Task list from Modal selections
    tasks_raw = []
    for i, id_obj in enumerate(inc_ids):
        d_id = id_obj["index"]
        if not inc_vals[i]: continue
        
        priority = prios[i] if prios[i] is not None else 100
        filename = metadata.get(d_id, {}).get("filename", "Unknown")
        
        # Pull specific config for THIS dataset from the maps
        # Fallback values if for some reason tab component is missing
        config = {
            "folds": folds_map.get(d_id, 3),
            "max_context": max_ctx_map.get(d_id, 60),
            "strategy": strategy_map.get(d_id, "random"),
            "algorithm": algo_map.get(d_id, "bulk"),
            "temperature": temp_map.get(d_id, 0.3),
            "chunk_size": chunk_map.get(d_id, 30),
            "instructions": f"Style: {tone_map.get(d_id, 'Concise')}\nConsistency: {consist_map.get(d_id, 'Strict')}\nUniqueness: {uniq_map.get(d_id, 'Allow Duplicates')}",
            "desc_metadata": desc_map.get(d_id, "")
        }
        
        tasks_raw.append({
            "priority": priority,
            "payload": (d_id, filename, orig_map.get(d_id, []), col_map.get(d_id, []), config)
        })

    if not tasks_raw:
        return dbc.Alert("No datasets selected.", color="warning", className="m-0"), "", [dash.no_update] * len(eda_ids)

    # Sort tasks by priority
    tasks_raw.sort(key=lambda x: x["priority"])
    tasks = [t["payload"] for t in tasks_raw]
    
    # 2. Run parallel execution
    try:
        pipeline_results = PipelineManager.run_pipeline(tasks)
    except Exception as e:
        logger.error(f"High-level pipeline failure: {e}")
        return dbc.Alert(f"Pipeline failed: {e}", color="danger"), "", [dash.no_update] * len(eda_ids)
    
    # 3. Process Results & Prepare Updates
    result_cards = []
    eda_updates = [dash.no_update] * len(eda_ids)
    eda_index_map = {id_obj["index"]: i for i, id_obj in enumerate(eda_ids)}

    for res in pipeline_results:
        d_id = res['dataset_id']
        fidelity = res.get('avg_fidelity', 0)
        color = "success" if fidelity > 80 else "warning" if fidelity > 50 else "danger"
        
        # Add summary card to modal results
        card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Small(res['filename'], className="fw-bold truncate-text", style={"maxWidth": "120px"}),
                    html.Small(f"{fidelity:.1f}%", className=f"fw-bold text-{color} ms-auto")
                ], className="d-flex align-items-center mb-1"),
                dbc.Progress(value=fidelity, color=color, style={"height": "4px"})
            ], className="p-2")
        ], className="mb-2 border-secondary bg-dark text-light shadow-sm")
        result_cards.append(card)
        
        # Update individual EDA container in tab
        if d_id in eda_index_map:
            try:
                df_orig = pd.DataFrame(orig_map.get(d_id, []))
                df_aug = pd.DataFrame(res.get('synthetic_data', []))
                
                if '_row_id' in df_orig.columns:
                    df_orig.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
                
                c_defs = col_map.get(d_id, [])
                field_to_header = {c['field']: c.get('headerName', c['field']) for c in c_defs 
                                   if c['field'] not in ['ai_audit', '_row_id'] and not c['field'].startswith('is_ai_')}
                
                df_orig = df_orig.rename(columns=field_to_header)
                
                if not df_aug.empty:
                    eda_idx = eda_index_map[d_id]
                    eda_updates[eda_idx] = get_eda_layout(df_orig, df_aug, d_id)
            except Exception as e:
                logger.error(f"EDA Refresh failed for {res['filename']}: {e}")

    summary_ui = html.Div(result_cards, style={"maxHeight": "200px", "overflowY": "auto"})
    status_msg = f"Completed {len(pipeline_results)} datasets."
    
    return summary_ui, status_msg, eda_updates
