import dash
from dash import Input, Output, State, ALL, MATCH, html
from core import app
from components.ml_components import get_model_config_ui
from services.ml_evaluator import MLEvaluator
import pandas as pd

@app.callback(
    Output({"type": "column-roles-store", "index": MATCH}, "data"),
    Input({"type": "target-cols-dropdown", "index": MATCH}, "value"),
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
    for model in selected_models:
        ui = get_model_config_ui(model, dataset_id, task_type)
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
