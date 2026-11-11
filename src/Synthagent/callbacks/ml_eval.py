import dash
from dash import Input, Output, State, ALL, MATCH, html
from core import app
from components.ml_components import get_model_config_ui
from services.ml_evaluator import MLEvaluator
from services.llm_client import detect_target_columns
import pandas as pd

@app.callback(
    Output({"type": "target-cols-dropdown", "index": MATCH}, "value", allow_duplicate=True),
    Input({"type": "btn-auto-suggest-targets", "index": MATCH}, "n_clicks"),
    State({"type": "original-data-store", "index": MATCH}, "data"),
    State({"type": "check-single-target", "index": MATCH}, "value"),
    State({"type": "algo-selector", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def auto_suggest_targets(n_clicks, orig_data, single_target, algorithm):
    if not n_clicks or not orig_data:
        return dash.no_update
    df = pd.DataFrame(orig_data)
    
    # Filter out internal and ID columns before sending to LLM
    cols_to_drop = [c for c in df.columns if c.lower() == 'id' or c.lower().endswith('_id') or c.lower().endswith('id')]
    cols_to_drop.extend(['_row_id', 'is_context', 'is_target', 'ai_audit'])
    
    df_filtered = df.drop(columns=cols_to_drop, errors='ignore')
    
    from utils.logging_config import logger
    suggested = detect_target_columns(df_filtered, single_target=single_target, algorithm=algorithm)
    logger.info(f"--- [CALLBACK] AUTO-SUGGEST TARGETS RECIEVED: {suggested} ---")
    
    # Filter to only include valid columns (double check against original df)
    valid_suggested = [c for c in suggested if c in df.columns]
    logger.info(f"--- [CALLBACK] AUTO-SUGGEST VALIDATED TARGETS: {valid_suggested} ---")
    
    # Defensive enforcement for single target
    if single_target and len(valid_suggested) > 1:
        valid_suggested = valid_suggested[:1]
        
    return valid_suggested

@app.callback(
    Output({"type": "column-roles-store", "index": MATCH}, "data", allow_duplicate=True),
    Input({"type": "target-cols-dropdown", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def sync_target_roles(targets):
    return {"targets": targets or []}

@app.callback(
    Output({"type": "ml-model-config-container", "index": MATCH}, "children"),
    Input({"type": "ml-model-checklist", "index": MATCH}, "value"),
    State({"type": "original-data-store", "index": MATCH}, "data"),
    State({"type": "column-roles-store", "index": MATCH}, "data"),
    State({"type": "ml-model-checklist", "index": MATCH}, "id"),
)
def render_ml_configs(selected_models, orig_data, roles, id_obj):
    if not selected_models:
        return html.Div("Select models to configure them.", className="text-muted small italic")
    
    dataset_id = id_obj["index"]
    df = pd.DataFrame(orig_data)
    task_type = MLEvaluator.detect_task_type(df, roles.get("targets", []))
    
    configs = []
    n_samples, n_features = df.shape
    for model in selected_models:
        ui = get_model_config_ui(model, dataset_id, task_type, n_samples=n_samples, n_features=n_features)
        if ui: configs.append(ui)
    
    return configs

@app.callback(
    Output("pipeline-ml-model-config-container", "children"),
    Input("pipeline-ml-model-checklist", "value"),
)
def render_global_ml_configs(selected_models):
    if not selected_models:
        return html.Div("Select models to configure them.", className="text-muted small italic")
        
    configs = []
    for model in selected_models:
        # Use a dummy dataframe or assume classification for UI defaults in the modal
        # since we don't know which dataset will be used yet (it's global)
        ui = get_model_config_ui(model, "pipeline-global", "classification")
        if ui: configs.append(ui)
    
    return configs

@app.callback(
    Output({"type": "ml-config-store", "index": MATCH}, "data"),
    Input({"type": "ml-model-checklist", "index": MATCH}, "value"),
    Input({"type": "ml-param", "index": ALL}, "value"),
    State({"type": "ml-param", "index": ALL}, "id"),
    State({"type": "ml-config-store", "index": MATCH}, "id"),
)
def collect_ml_params(selected_models, param_values, param_ids, id_obj):
    dataset_id = id_obj["index"]
    config = {"selected_models": selected_models or [], "params": {}}
    
    for val, pid in zip(param_values, param_ids):
        # pid["index"] is like "dataset_id-model-param_name"
        idx_str = pid["index"]
        if idx_str.startswith(dataset_id):
            # Parse model and param name
            parts = idx_str[len(dataset_id)+1:].split("-")
            if len(parts) >= 2:
                model = parts[0]
                param = "-".join(parts[1:])
                if model not in config["params"]:
                    config["params"][model] = {}
                config["params"][model][param] = val
                
    return config
