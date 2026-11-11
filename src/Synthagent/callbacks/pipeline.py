import dash
from dash import Input, Output, State, ALL, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from core import app
from utils.logging_config import logger
from services.pipeline_manager import PipelineManager, run_single_dataset_benchmark
from services.llm_client import detect_target_columns
from callbacks.eda import get_eda_layout
import plotly.graph_objects as go
import json
import multiprocessing
import time

MODEL_NAMES = {
    "mlp": "Multi-Layer Perceptron (MLP)",
    "rf": "Random Forest (RF)",
    "xgb": "Extreme Gradient Boosting (XGB)",
    "gb": "Gradient Boosting (GB)",
    "ab": "AdaBoost (AB)",
    "svm": "Support Vector Machine (SVM)",
    "nb": "Naive Bayes (NB)",
    "knn": "K-Nearest Neighbors (KNN)",
    "logistic": "Logistic Regression (LogR)",
    "linear": "Linear Regression (LinR)",
    "dt": "Decision Tree (DT)"
}

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
    State("pipeline-config-store", "data"),
    State({"type": "column-roles-store", "index": ALL}, "data"),
    State({"type": "column-roles-store", "index": ALL}, "id"),
    prevent_initial_call=True
)
def render_pipeline_ui_in_modal(n_clicks, metadata, pipeline_config, roles_vals, roles_ids):
    if not metadata:
        return dbc.Alert("No datasets uploaded. Upload files to start a pipeline.", color="info", className="m-0")
    
    pipeline_config = pipeline_config or {}
    roles_map = {id_obj["index"]: val for id_obj, val in zip(roles_ids, roles_vals)}
    ds_settings = pipeline_config.get("datasets", {})

    rows = []
    for dataset_id in sorted(metadata.keys()):
        info = metadata[dataset_id]
        if dataset_id == "default-dataset" and info.get("rows", 0) <= 10:
             if info.get("rows", 0) <= 0 or info.get("filename") == "Workbench":
                  continue

        size_str = f"{info.get('rows', 0)}R × {info.get('cols', 0)}C"
        existing_targets = roles_map.get(dataset_id, {}).get("targets", [])
        target_val = ",".join(existing_targets) if existing_targets else ""
        
        # Header for the list
        if not rows:
             rows.append(dbc.ListGroupItem([
                 dbc.Row([
                     dbc.Col(html.Small("Incl", className="xx-small text-muted fw-bold text-center"), width=1),
                     dbc.Col(html.Small("Dataset Filename", className="xx-small text-muted fw-bold"), width=2),
                     dbc.Col(html.Small("Shape", className="xx-small text-muted fw-bold text-center"), width=1),
                     dbc.Col(html.Small("Target Columns", className="xx-small text-muted fw-bold"), width=2),
                     dbc.Col(html.Small("Join | Context & Gen size | Tokens", className="xx-small text-muted fw-bold"), width=5),
                     dbc.Col(html.Small("Prio", className="xx-small text-muted fw-bold text-end"), width=1),
                 ], align="center", className="gx-1")
             ], className="bg-transparent border-0 border-bottom border-secondary pb-1 mb-1"))

        p_val = ds_settings.get(dataset_id, {})
        priority = p_val.get("priority", 1)
        include = p_val.get("include", True)
        join_context = p_val.get("join_context", True)
        ctx_size_pct = p_val.get("ctx_size_pct", 20)
        gen_size_pct = p_val.get("gen_size_pct", 80)

        rows.append(dbc.ListGroupItem([
            dbc.Row([
                dbc.Col(dbc.Checkbox(id={"type": "pipeline-include", "index": dataset_id}, value=include, className="d-flex justify-content-center"), width=1),
                dbc.Col(html.Span(info["filename"], className="fw-bold text-info small truncate-text", style={"maxWidth": "120px"}), width=2),
                dbc.Col(html.Small(size_str, className="text-muted d-block text-center", style={"fontSize": "0.65rem"}), width=1),
                dbc.Col(dbc.Input(id={"type": "pipeline-target", "index": dataset_id}, placeholder="Target", value=target_val, size="sm", className="bg-dark text-light border-secondary"), width=2),
                dbc.Col([
                    # Controls: Join Toggle + Two Sliders
                    html.Div([
                        dbc.Checkbox(id={"type": "pipeline-join-context", "index": dataset_id}, value=join_context, label="Join", labelClassName="x-small text-muted mb-0", className="me-2"),
                        html.Div([
                            html.Small("Ctx:", className="xx-small text-muted me-1"),
                                dcc.Slider(
                                    id={"type": "pipeline-ctx-size", "index": dataset_id},
                                    min=5, max=95, step=5, value=ctx_size_pct,
                                    marks={5: {"label": "5%", "style": {"fontSize": "0.41rem"}}, 95: {"label": "95%", "style": {"fontSize": "0.41rem"}}},
                                    className="p-0 flex-grow-1"
                                ),
                        ], className="d-flex align-items-center flex-grow-1 me-2"),
                        html.Div([
                            html.Small("Gen:", className="xx-small text-muted me-1"),
                                dcc.Slider(
                                    id={"type": "pipeline-gen-size", "index": dataset_id},
                                    min=10, max=500, step=10, value=gen_size_pct,
                                    marks={10: {"label": "10%", "style": {"fontSize": "0.41rem"}}, 500: {"label": "500%", "style": {"fontSize": "0.41rem"}}},
                                    className="p-0 flex-grow-1"
                                ),
                        ], className="d-flex align-items-center flex-grow-1"),
                    ], className="d-flex align-items-center mb-1"),
                    html.Div(id={"type": "pipeline-token-est", "index": dataset_id}, className="text-info x-small text-end pt-1", style={"fontSize": "0.6rem"}),
                ], width=5, className="px-2 d-flex flex-column justify-content-center"),
                dbc.Col(dbc.Input(id={"type": "pipeline-priority", "index": dataset_id}, type="number", value=priority, size="sm", className="text-end"), width=1),
            ], align="center", className="gx-2")
        ], className="bg-dark border-secondary text-light py-2 px-1"))

    if not rows:
        return dbc.Alert("No valid datasets found for pipeline.", color="warning", className="m-0")

    kfolds = pipeline_config.get("global_folds", 3)
    auto_detect = pipeline_config.get("auto_detect", True)
    selected_models = pipeline_config.get("selected_models", ["logistic", "rf", "xgb", "mlp", "svm", "knn", "nb", "dt", "gb", "ab"])

    ui = html.Div([
        html.P("Benchmark validation pipeline configuration.", className="text-muted small mb-3"),
        
        dbc.Row([
            dbc.Col(html.Label("Parallel Execution:", className="small text-info fw-bold"), width=3),
            dbc.Col(dbc.Switch(id="pipeline-parallel-toggle", label="", value=pipeline_config.get("parallel_enabled", True), className="mt-1"), width=3),
        ], className="mb-3 align-items-center g-2 bg-dark border border-secondary rounded p-2 mx-0"),
        
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Checkbox(id="pipeline-auto-detect-targets", label="detect targets automatically", value=auto_detect, className="mb-3 text-info small"),
                dbc.ListGroup(rows, className="mb-3 border-secondary"),
                dbc.Row([
                    dbc.Col(html.Label("K-Folds (Global):", className="small text-muted fw-bold"), width=3),
                    dbc.Col(dbc.Input(id="pipeline-global-folds", type="number", value=kfolds, min=1, max=10, size="sm", className="bg-dark text-light border-secondary"), width=3),
                ], className="mb-3 align-items-center g-2"),
            ], title=[html.I(className="bi bi-database-fill me-2"), "Datasets Configurations"], item_id="datasets-settings", className="bg-transparent border-0"),
        
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        html.Label("NA Handling Strategy:", className="text-muted small fw-bold mb-2 d-block"),
                        dbc.Select(
                            id="pipeline-na-handling",
                            options=[
                                {"label": "Drop Rows with NAs", "value": "drop"},
                                {"label": "Impute (Mean/Mode)", "value": "impute"}
                            ],
                            value=pipeline_config.get("na_strategy", "impute"),
                            size="sm", className="bg-dark text-light border-secondary mb-3"
                        ),
                        html.Label("Select Models:", className="text-muted small fw-bold mb-2 d-block"),
                        dbc.Checklist(
                            id="pipeline-ml-model-checklist",
                            options=[{"label": v, "value": k} for k, v in MODEL_NAMES.items()],
                            value=selected_models,
                            switch=True, className="text-light small",
                            style={"fontSize": "0.8rem"}
                        )
                    ], md=4),
                    dbc.Col([
                        html.Div(id="pipeline-ml-model-config-container", className="ps-2 border-start border-secondary")
                    ], md=8)
                ], className="g-3")
            ], title=[html.I(className="bi bi-cpu-fill me-2"), "Downstream ML Evaluation Settings"], 
               item_id="ml-settings", 
               className="bg-transparent border-0")
        ], active_item=[], flush=True, className="mb-4 rounded border border-secondary"),

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
    Output("pipeline-config-store", "data"),
    Input("pipeline-global-folds", "value"),
    Input("pipeline-ml-model-checklist", "value"),
    Input("pipeline-auto-detect-targets", "value"),
    Input("pipeline-parallel-toggle", "value"),
    Input("pipeline-na-handling", "value"),
    Input({"type": "pipeline-priority", "index": ALL}, "value"),
    Input({"type": "pipeline-include", "index": ALL}, "value"),
    Input({"type": "pipeline-join-context", "index": ALL}, "value"),
    Input({"type": "pipeline-ctx-size", "index": ALL}, "value"),
    Input({"type": "pipeline-gen-size", "index": ALL}, "value"),
    State({"type": "pipeline-priority", "index": ALL}, "id"),
    State("pipeline-config-store", "data"),
    prevent_initial_call=True
)
def sync_pipeline_config_pm(folds, models, auto, parallel, na_strategy, priorities, includes, joins, ctx_sizes, gen_sizes, ids, current_config):
    current_config = current_config or {}
    current_config["global_folds"] = folds
    current_config["selected_models"] = models
    current_config["auto_detect"] = auto
    current_config["parallel_enabled"] = parallel
    current_config["na_strategy"] = na_strategy
    
    if "datasets" not in current_config:
        current_config["datasets"] = {}
        
    for i, id_obj in enumerate(ids):
        d_id = id_obj["index"]
        if d_id not in current_config["datasets"]:
            current_config["datasets"][d_id] = {}
        current_config["datasets"][d_id]["priority"] = priorities[i]
        current_config["datasets"][d_id]["include"] = includes[i]
        current_config["datasets"][d_id]["join_context"] = joins[i]
        current_config["datasets"][d_id]["ctx_size_pct"] = ctx_sizes[i]
        current_config["datasets"][d_id]["gen_size_pct"] = gen_sizes[i]
        
    return current_config

def get_delta_color_class(metric_name, delta_val):
    """
    Returns the Bootstrap color class (success, warning, danger) for a delta value.
    Higher is Better: Accuracy, F1, Precision, Recal, AUC, JSD, Stability...
    Lower is Better: RMSE, MSE, MAE, Drift, Risk, Shift...
    """
    m_lower = metric_name.lower()
    lower_is_better = any(x in m_lower for x in ["rmse", "mse", "mae", "drift", "risk", "memorization", "shift"])
    
    if lower_is_better:
        if delta_val <= 0: return "success" # Green (Better or Equal)
        if delta_val <= 0.05: return "warning" # Orange (Slightly Worse)
        return "danger" # Red (Worse)
    else:
        # Higher is Better
        if delta_val >= 0: return "success" # Green (Better or Equal)
        if delta_val >= -0.05: return "warning" # Orange (Slightly Worse)
        return "danger" # Red (Worse)

@app.callback(
    Output("pipeline-status", "children"),
    Output("pipeline-results-container", "children"),
    Output({"type": "eda-dashboard-container", "index": ALL}, "children"),
    Output({"type": "ml-evaluation-results-dashboard", "index": ALL}, "children"),
    Output({"type": "pipeline-target", "index": ALL}, "value"),
    Output({"type": "target-cols-dropdown", "index": ALL}, "value", allow_duplicate=True),
    Output({"type": "column-roles-store", "index": ALL}, "data", allow_duplicate=True),
    Output("pipeline-results-store", "data"),
    Input("btn-run-pipeline", "n_clicks"),
    State("pipeline-global-folds", "value"),
    State("pipeline-ml-model-checklist", "value"),
    State("pipeline-auto-detect-targets", "value"),
    State({"type": "ml-param", "index": ALL}, "value"),
    State({"type": "ml-param", "index": ALL}, "id"),
    State("datasets-metadata-store", "data"),
    State({"type": "pipeline-include", "index": ALL}, "value"),
    State({"type": "pipeline-include", "index": ALL}, "id"),
    State({"type": "pipeline-priority", "index": ALL}, "value"),
    State({"type": "original-data-store", "index": ALL}, "data"),
    State({"type": "original-data-store", "index": ALL}, "id"),
    State({"type": "data-grid", "index": ALL}, "columnDefs"),
    State({"type": "data-grid", "index": ALL}, "id"),
    State({"type": "dataset-description-store", "index": ALL}, "data"),
    State({"type": "dataset-description-store", "index": ALL}, "id"),
    State({"type": "column-roles-store", "index": ALL}, "data"),
    State({"type": "column-roles-store", "index": ALL}, "id"),
    State({"type": "eda-dashboard-container", "index": ALL}, "id"),
    State({"type": "ml-evaluation-results-dashboard", "index": ALL}, "id"),
    State({"type": "pipeline-target", "index": ALL}, "value"),
    State({"type": "pipeline-target", "index": ALL}, "id"),
    State({"type": "pipeline-join-context", "index": ALL}, "value"),
    State({"type": "pipeline-ctx-size", "index": ALL}, "value"),
    State({"type": "pipeline-gen-size", "index": ALL}, "value"),
    State({"type": "target-cols-dropdown", "index": ALL}, "id"),
    State({"type": "algo-selector", "index": ALL}, "value"),
    State({"type": "algo-selector", "index": ALL}, "id"),
    State("pipeline-parallel-toggle", "value"),
    background=True,
    progress=[
        Output("pipeline-status", "children"),
        Output("pipeline-results-container", "children")
    ],
    prevent_initial_call=True
)
def execute_pipeline_pm(set_progress, n_clicks, global_folds, global_ml_models, auto_detect,
                        all_ml_param_values, all_ml_param_ids, metadata, 
                        inc_vals, inc_ids, prios,
                        orig_data_list, orig_ids,
                        col_defs_list, col_ids,
                        desc_list, desc_ids,
                        roles_list, roles_ids,
                        eda_ids, ml_res_ids,
                        dataset_targets_vals, dataset_targets_ids,
                        join_vals, ctx_size_vals, gen_size_vals,
                        target_dropdown_ids,
                        all_algo_vals, all_algo_ids,
                        parallel_enabled):
    
    if not n_clicks: 
        return (dash.no_update, dash.no_update, [dash.no_update]*len(eda_ids), 
                [dash.no_update]*len(ml_res_ids), [dash.no_update]*len(dataset_targets_ids), 
                [dash.no_update]*len(target_dropdown_ids), [dash.no_update]*len(roles_ids), dash.no_update)
    
    logger.info(f"[UI] Benchmark Pipeline Triggered. Folds: {global_folds} | Parallel: {parallel_enabled}")
    
    def map_states(values, ids):
        return {id_obj["index"]: values[i] for i, id_obj in enumerate(ids)}

    orig_map = map_states(orig_data_list, orig_ids)
    col_map = map_states(col_defs_list, col_ids)
    desc_map = map_states(desc_list, desc_ids)
    roles_map = map_states(roles_list, roles_ids)
    target_map = map_states(dataset_targets_vals, dataset_targets_ids)
    algo_map = map_states(all_algo_vals, all_algo_ids)
    join_map = map_states(join_vals, dataset_targets_ids)
    ctx_size_map = map_states(ctx_size_vals, dataset_targets_ids) # Using targets_ids for index mapping consistency
    gen_size_map = map_states(gen_size_vals, dataset_targets_ids)
    
    global_ml_config = {"selected_models": global_ml_models or [], "params": {}}
    for val, pid in zip(all_ml_param_values, all_ml_param_ids):
        if pid["index"].startswith("pipeline-global"):
            parts = pid["index"][len("pipeline-global")+1:].split("-")
            if len(parts) >= 2:
                model = parts[0]
                param = "-".join(parts[1:])
                if model not in global_ml_config["params"]:
                    global_ml_config["params"][model] = {}
                global_ml_config["params"][model][param] = val

    tasks_raw = []
    for i, id_obj in enumerate(inc_ids):
        d_id = id_obj["index"]
        if not inc_vals[i]: continue
        
        priority = prios[i] if prios[i] is not None else 100
        filename = metadata.get(d_id, {}).get("filename", "Unknown")
        raw_target = target_map.get(d_id, "")
        target_cols = [t.strip() for t in raw_target.split(",")] if raw_target else roles_map.get(d_id, {}).get("targets", [])
        
        if auto_detect and not raw_target:
            set_progress((f"Detecting target for {filename}...", html.Div(dbc.Spinner(size="sm"))))
            try:
                df_temp = pd.DataFrame(orig_map.get(d_id, []))
                if not df_temp.empty:
                    detected = detect_target_columns(df_temp, single_target=True, algorithm=algo_map.get(d_id))
                    if detected:
                        roles_map[d_id] = {"targets": detected}
                        target_cols = detected
                        target_map[d_id] = ",".join(detected)
            except Exception as e:
                logger.error(f"Auto-detection failed for {filename}: {e}")
        
        config = {
            "folds": int(global_folds) if global_folds else 3,
            "ml_config": global_ml_config,
            "targets": target_cols,
            "desc_metadata": desc_map.get(d_id, ""),
            "algorithm": algo_map.get(d_id, "google/gemma-3-4b"),
            "join_context": join_map.get(d_id, True),
            "ctx_size_pct": ctx_size_map.get(d_id, 20),
            "gen_size_pct": gen_size_map.get(d_id, 80)
        }
        tasks_raw.append({"priority": priority, "payload": (d_id, filename, orig_map.get(d_id, []), col_map.get(d_id, []), config)})

    if not tasks_raw:
        logger.warning("[UI] Pipeline triggered but no datasets were selected for processing.")
        return (dbc.Alert("No datasets selected.", color="warning", className="m-0"), "", 
                [dash.no_update]*len(eda_ids), [dash.no_update]*len(ml_res_ids), 
                [dash.no_update]*len(dataset_targets_ids), [dash.no_update]*len(target_dropdown_ids),
                [dash.no_update]*len(roles_ids), dash.no_update)
    
    logger.info(f"[UI] Pipeline Tasks Queued: {len(tasks_raw)} dataset(s). Models: {global_ml_models}")

    tasks_raw.sort(key=lambda x: x["priority"])
    tasks = [t["payload"] for t in tasks_raw]
    
    try:
        from multiprocessing import Manager
        manager = Manager()
        progress_dict = manager.dict()
        tasks_with_progress = []
        for t in tasks:
            d_id, filename, d, c, cfg = t
            cfg["_progress_dict"] = progress_dict
            cfg["_dataset_id"] = d_id
            tasks_with_progress.append((d_id, filename, d, c, cfg))
            progress_dict[d_id] = {"filename": filename, "done": 0, "total": cfg.get("folds", 3), "status": "Queued"}

        # Define Progress Polling Function
        def update_ui_progress(is_done=False):
            prog_bars = []
            for d_id, p_info in progress_dict.items():
                pct = (p_info["done"] / p_info["total"]) * 100 if p_info["total"] > 0 else 0
                prog_bars.append(html.Div([
                    dbc.Row([
                        dbc.Col(html.Small(p_info["filename"], className="fw-bold text-info truncate-text", style={"maxWidth": "150px"}), width=4),
                        dbc.Col(dbc.Progress(value=pct, color="info", striped=True, animated=True, style={"height": "8px"}), width=6),
                        dbc.Col(html.Small(f"{int(pct)}%", className="text-light x-small"), width=2),
                    ], align="center", className="mb-1"),
                    html.Div(html.Small(p_info.get("status", "Queued"), className="text-muted xx-small fst-italic ms-1"), className="mb-2")
                ]))
            msg = f"Completed {sum(1 for d_id, p in progress_dict.items() if p['done'] == p['total'])}/{len(tasks)} datasets..." if not is_done else "Finalizing results..."
            set_progress((msg, html.Div(prog_bars, className="mt-3 border-top border-secondary pt-3")))

        pipeline_results = []
        if parallel_enabled:
            cpus = min(len(tasks), multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=cpus) as pool:
                async_results = [pool.apply_async(run_single_dataset_benchmark, (t,)) for t in tasks_with_progress]
                while any(not r.ready() for r in async_results):
                    update_ui_progress()
                    time.sleep(1)
                pipeline_results = [r.get() for r in async_results]
            # Sequential Execution
            for t in tasks_with_progress:
                # In sequential mode, we can't poll easily, so we just show start status
                update_ui_progress()
                res = run_single_dataset_benchmark(t)
                pipeline_results.append(res)
                # Force an update after each dataset in sequential mode
                update_ui_progress()
            update_ui_progress(is_done=True)
    except Exception as e:
        logger.error(f"[UI] Pipeline Execution Engine Failure: {e}")
        return (dbc.Alert(f"Pipeline failed: {e}", color="danger"), dash.no_update, [dash.no_update]*len(eda_ids), [dash.no_update]*len(ml_res_ids), [dash.no_update]*len(dataset_targets_ids), [dash.no_update]*len(target_dropdown_ids), [dash.no_update]*len(roles_ids), dash.no_update)
    
    logger.info(f"[UI] Pipeline processing complete. Processing results for {len(pipeline_results)} datasets.")
    
    result_cards = [html.H5("Results", className="text-light fw-bold mb-3 border-bottom border-secondary pb-2")]
    eda_updates = [dash.no_update] * len(eda_ids)
    ml_updates = [dash.no_update] * len(ml_res_ids)
    target_updates = [dash.no_update] * len(dataset_targets_ids)
    dropdown_updates = [dash.no_update] * len(target_dropdown_ids)
    roles_updates = [dash.no_update] * len(roles_ids)
    
    target_index_map = {id_obj["index"]: i for i, id_obj in enumerate(dataset_targets_ids)}
    dropdown_index_map = {id_obj["index"]: i for i, id_obj in enumerate(target_dropdown_ids)}
    roles_index_map = {id_obj["index"]: i for i, id_obj in enumerate(roles_ids)}
    eda_index_map = {id_obj["index"]: i for i, id_obj in enumerate(eda_ids)}
    ml_index_map = {id_obj["index"]: i for i, id_obj in enumerate(ml_res_ids)}

    for d_id, target in target_map.items():
        if d_id in target_index_map: target_updates[target_index_map[d_id]] = target
        if d_id in dropdown_index_map: dropdown_updates[dropdown_index_map[d_id]] = [t.strip() for t in target.split(",")] if target else []
        if d_id in roles_index_map: roles_updates[roles_index_map[d_id]] = roles_map.get(d_id, {"targets": []})

    for res in pipeline_results:
        d_id = res['dataset_id']
        results_list_m = res.get('results', [])
        if not results_list_m: continue
        
        agg_metrics = {"fidelity": [], "correlation": [], "privacy": [], "coverage": []}
        for r in results_list_m:
            agg_metrics["fidelity"].append(r.get("Synthetic Fidelity (JSD)", 0))
            agg_metrics["correlation"].append(r.get("Correlation Stability", 0))
            agg_metrics["privacy"].append(r.get("Privacy Risk", 0))
            agg_metrics["coverage"].append(r.get("Coverage Score", 0))
        
        ml_agg = res.get("overall_ml", {})
        task_type = "classification"
        # Determine task type for header purposes
        for m_type, m_data in ml_agg.items():
             if "folds_metadata" in m_data and m_data["folds_metadata"]:
                  # Heuristic or just take from the first fold
                  pass

        def get_stats(vals):
            if not vals: return 0.0, 0.0
            return float(np.mean(vals)), float(np.std(vals))

        f_m, f_s = get_stats(agg_metrics["fidelity"])
        c_m, c_s = get_stats(agg_metrics["correlation"])
        p_m, p_s = get_stats(agg_metrics["privacy"])
        cov_m, cov_s = get_stats(agg_metrics["coverage"])

        syn_col = html.Div([
            html.Small("SYNTHETIC DATA QUALITY", className="text-muted xx-small fw-bold border-bottom border-secondary d-block mb-2"),
            html.Div([
                dbc.Row([dbc.Col(html.Small("Fidelity", className="x-small text-light"), width=4), dbc.Col(html.Small(f"{f_m:.1f}±{f_s:.1f}", className="x-small text-info fw-bold"), width=3), dbc.Col(dbc.Progress(value=f_m, color="info", style={"height": "4px"}), width=5)], align="center", className="g-0 mb-1"),
                dbc.Row([dbc.Col(html.Small("Correlation", className="x-small text-light"), width=4), dbc.Col(html.Small(f"{c_m:.1f}±{c_s:.1f}", className="x-small text-light"), width=3), dbc.Col(dbc.Progress(value=c_m, color="success", style={"height": "4px"}), width=5)], align="center", className="g-0 mb-1"),
                dbc.Row([dbc.Col(html.Small("Privacy (Risk)", className="x-small text-light"), width=4), dbc.Col(html.Small(f"{p_m:.1f}±{p_s:.1f}", className="x-small text-light"), width=3), dbc.Col(dbc.Progress(value=p_m, color="danger", style={"height": "4px"}), width=5)], align="center", className="g-0 mb-1"),
            ])
        ])
        
        ml_rows = []
        # Find task type from first valid entry
        for m_t, m_d in ml_agg.items():
            if "accuracy" in m_d:
                task_type = "classification"
                break
            if "rmse" in m_d:
                task_type = "regression"
                break

        primary_metric = "accuracy" if task_type == "classification" else "r2"
        # Add Header Row for Predictive Utility
        ml_rows.append(dbc.Row([
            dbc.Col(html.Small("Model", className="xx-small text-muted fw-bold"), width=5),
            dbc.Col(html.Small("Sync (±SD)", className="xx-small text-muted fw-bold"), width=3),
            dbc.Col(html.Small("Real", className="xx-small text-muted fw-bold"), width=2),
            dbc.Col(html.Small("Delta", className="xx-small text-muted fw-bold"), width=2),
        ], className="g-0 opacity-75 mb-2 border-bottom border-secondary pb-1"))

        for m_type, m_data in ml_agg.items():
            m_label = MODEL_NAMES.get(m_type, m_type.upper())
            
            if "error" in m_data:
                err_text = m_data["error"]
                ml_rows.append(dbc.Row([
                    dbc.Col(html.Small(m_label, className="x-small text-muted"), width=5),
                    dbc.Col(html.Span([
                        html.I(className="bi bi-exclamation-triangle-fill me-1 text-warning"),
                        html.Small(err_text, className="xx-small text-warning")
                    ], className="d-flex align-items-center"), width=7)
                ], className="g-0 mb-1 opacity-75"))
                continue

            metrics = m_data.get(primary_metric, {})
            if metrics:
                s_aug = metrics.get("display_aug", "N/A")
                s_real = metrics.get("display_real", "0.000")
                s_delta = metrics.get("display_delta", "+0.000")
                
                # Use 3-state coloring helper
                d_val = metrics.get("delta", 0)
                d_color = get_delta_color_class(primary_metric, d_val)
                
                ml_rows.append(html.Div([
                    dbc.Row([
                        dbc.Col(html.Small(m_label, className="x-small text-info fw-bold"), width=5),
                        dbc.Col(html.Small(s_aug.split()[0], className="x-small text-light"), width=3),
                        dbc.Col(html.Small(s_real.split()[0], className="x-small text-muted"), width=2),
                        dbc.Col(html.Small(s_delta.split()[0], className=f"x-small text-{d_color} fw-bold text-end"), width=2)
                    ], className="g-0 align-items-center mb-1")
                ]))

        pred_col = html.Div([
            html.Small("PREDICTIVE UTILITY", className="text-muted xx-small fw-bold border-bottom border-secondary d-block mb-2"),
            html.Div(ml_rows if ml_rows else html.Small("No models evaluated", className="text-muted italic x-small"))
        ], className="ps-3 border-start border-secondary")

        card = dbc.Card([dbc.CardBody([html.Div(res['filename'], className="fw-bold text-info small mb-3"), dbc.Row([dbc.Col(syn_col, width=5), dbc.Col(pred_col, width=7)])], className="py-2 px-3")], className="mb-3 border-secondary bg-dark shadow-sm")
        result_cards.append(card)
        
        if d_id in eda_index_map:
            try:
                df_orig, df_aug = pd.DataFrame(orig_map.get(d_id, [])), pd.DataFrame(res.get('augmented_data', []))
                field_to_header = {c['field']: c.get('headerName', c['field']) for c in col_defs_list[eda_index_map[d_id]] if c['field'] not in ['ai_audit', '_row_id']}
                df_orig.rename(columns=field_to_header, inplace=True)
                if not df_aug.empty: eda_updates[eda_index_map[d_id]] = get_eda_layout(df_orig, df_aug, d_id, results_list=results_list_m)
            except Exception as e: logger.error(f"EDA build failed for {d_id}: {e}")

        if d_id in ml_index_map:
            ml_cards = []
            for m_type, metrics in ml_agg.items():
                m_rows = []
                if "error" in metrics:
                     m_rows.append(html.Small(f"Error: {metrics['error']}", className="text-warning xx-small italic"))
                else:
                    for m_name, data in metrics.items():
                        if not isinstance(data, dict) or "display_aug" not in data: continue
                        orig_d, aug_d, d_d = data["display_real"], data["display_aug"], data["display_delta"]
                        d_val = data.get("delta", 0)
                        m_rows.append(dbc.Row([
                             dbc.Col(html.Small(m_name.upper(), className="x-small text-info fw-bold"), width=3),
                             dbc.Col(html.Span(orig_d.split()[0], className="x-small text-muted"), width=3),
                             dbc.Col(html.Span(aug_d, className="x-small text-light fw-bold"), width=3),
                             dbc.Col(html.Span(d_d.split()[0], className=f"text-{get_delta_color_class(m_name, d_val)} x-small fw-bold"), width=3)
                        ], className="g-0 border-bottom border-secondary-subtle py-1"))
                ml_cards.append(dbc.Col(dbc.Card([dbc.CardBody([html.Div([html.Div([html.I(className="bi bi-cpu me-2 text-info"), html.Small(MODEL_NAMES.get(m_type, m_type.upper()), className="fw-bold")], className="d-flex align-items-center"), dbc.Button("Inspect", id={"type": "btn-inspect-ml", "dataset": d_id, "model": m_type}, color="info", size="sm", outline=True, className="rounded-pill p-0 px-2", style={"fontSize": "0.6rem"}, n_clicks=0)], className="d-flex justify-content-between mb-2"), html.Div(m_rows)], className="p-2")], className="h-100 border-secondary bg-dark shadow-sm"), lg=3, md=6, className="mb-3"))
            if ml_cards: ml_updates[ml_index_map[d_id]] = html.Div([html.H6("ML Results & Diagnostics", className="text-light mb-3 small fw-bold"), dbc.Row(ml_cards, className="g-3")])

    results_dict = {res['dataset_id']: res for res in pipeline_results}
    status_msg = f"Pipeline completed: {len(pipeline_results)} datasets processed."
    return status_msg, html.Div(result_cards), eda_updates, ml_updates, target_updates, dropdown_updates, roles_updates, results_dict

@app.callback(
    Output("ml-diag-modal", "is_open"),
    Output("ml-diag-active-model", "data"),
    Output("ml-diag-current-fold", "data", allow_duplicate=True),
    Input({"type": "btn-inspect-ml", "dataset": ALL, "model": ALL}, "n_clicks"),
    Input("btn-diag-close", "n_clicks"),
    State({"type": "btn-inspect-ml", "dataset": ALL, "model": ALL}, "id"),
    prevent_initial_call=True
)
def toggle_ml_diag_modal(n_clicks_list, n_close, ids):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update, dash.no_update
    trig_id = ctx.triggered_id
    if trig_id == "btn-diag-close": return False, dash.no_update, 0
    if isinstance(trig_id, dict) and trig_id.get("type") == "btn-inspect-ml":
        for n, b_id in zip(n_clicks_list, ids):
            if b_id == trig_id and n and n > 0: return True, {"dataset": trig_id["dataset"], "model": trig_id["model"]}, 0
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("ml-diag-current-fold", "data"),
    Input("btn-diag-prev", "n_clicks"),
    Input("btn-diag-next", "n_clicks"),
    State("ml-diag-current-fold", "data"),
    State("ml-diag-active-model", "data"),
    State("pipeline-results-store", "data"),
    prevent_initial_call=True
)
def navigate_ml_diag_fold(n_prev, n_next, current_idx, active_model, results_all):
    if not active_model or not results_all: return 0
    res = results_all.get(active_model["dataset"], {})
    folds = res.get("overall_ml", {}).get(active_model["model"], {}).get("folds_metadata", [])
    if not folds: return 0
    trig = dash.callback_context.triggered[0]["prop_id"]
    if "btn-diag-prev" in trig: return (current_idx - 1) % len(folds)
    if "btn-diag-next" in trig: return (current_idx + 1) % len(folds)
    return current_idx

@app.callback(
    Output("ml-diag-modal-title", "children"),
    Output("ml-diag-params-content", "children"),
    Output("ml-diag-aggregate-content", "children"),
    Input("ml-diag-modal", "is_open"),
    State("ml-diag-active-model", "data"),
    State("pipeline-results-store", "data")
)
def render_ml_diag_static_content(is_open, active_model, results_all):
    if not is_open or not active_model or not results_all:
        return dash.no_update, dash.no_update, dash.no_update
    
    d_id, m_type = active_model["dataset"], active_model["model"]
    res = results_all.get(d_id, {})
    model_res = res.get("overall_ml", {}).get(m_type, {})
    folds_meta = model_res.get("folds_metadata", [])
    
    title = [html.I(className="bi bi-graph-up me-2"), f"{MODEL_NAMES.get(m_type, m_type.upper())}: {res.get('filename', 'Dataset')}"]
    if not folds_meta:
        return title, "", dbc.Alert("No diagnostic data available.", color="warning")
    
    # Calculate aggregate
    agg_ui = []
    has_loss, has_fi, has_coef = any("loss_curve" in m for m in folds_meta), any("feature_importances" in m for m in folds_meta), any("coef" in m for m in folds_meta)
    
    if m_type == "mlp" and has_loss:
        all_c = [m.get("loss_curve", []) for m in folds_meta if "loss_curve" in m]
        if all_c:
            mlen = max(len(c) for c in all_c); aligned = [c + [np.nan]*(mlen-len(c)) for c in all_c]
            mn, st = np.nanmean(aligned, axis=0), np.nanstd(aligned, axis=0); steps = list(range(mlen))
            fig = go.Figure([go.Scatter(x=steps+steps[::-1], y=(mn+st).tolist()+(mn-st).tolist()[::-1], fill='toself', fillcolor='rgba(0,209,255,0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False), go.Scatter(x=steps, y=mn.tolist(), mode='lines', line=dict(color='#00d1ff', width=3), name='Mean Loss')])
            fig.update_layout(title="Mean Loss \u00b1 STD", template="plotly_dark", height=350, margin=dict(l=40, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            agg_ui.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    elif has_fi:
        all_i = [m.get("feature_importances", []) for m in folds_meta if "feature_importances" in m]
        names = next((m.get("feature_names") for m in folds_meta if "feature_names" in m), None)
        if all_i and names:
            minl = min(len(i) for i in all_i); all_i = [i[:minl] for i in all_i]; names = names[:minl]
            mn, st = np.mean(all_i, axis=0), np.std(all_i, axis=0); idx = np.argsort(mn)[-15:]
            fig = go.Figure(go.Bar(x=[mn[i] for i in idx], y=[names[i] for i in idx], orientation='h', marker_color='#00d1ff', error_x=dict(type='data', array=[st[i] for i in idx], visible=True)))
            fig.update_layout(title="Mean Feature Importance \u00b1 STD", template="plotly_dark", height=350, margin=dict(l=150, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            agg_ui.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    elif has_coef:
        def flt(c): arr = np.array(c); return np.mean(np.abs(arr), axis=0) if arr.ndim > 1 else arr
        all_c = [flt(m.get("coef", [])) for m in folds_meta if "coef" in m]
        names = next((m.get("feature_names") for m in folds_meta if "feature_names" in m), None)
        if all_c and names:
            minl = min(len(c) for c in all_c); all_c = [c[:minl] for c in all_c]; names = names[:minl]
            mn, st = np.mean(all_c, axis=0), np.std(all_c, axis=0); idx = np.argsort(np.abs(mn))[-15:]
            fig = go.Figure(go.Bar(x=[mn[i] for i in idx], y=[names[i] for i in idx], orientation='h', marker_color='#ff8c00', error_x=dict(type='data', array=[st[i] for i in idx], visible=True)))
            fig.update_layout(title="Mean Coefficient Effect \u00b1 STD", template="plotly_dark", height=350, margin=dict(l=150, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            agg_ui.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    else:
        agg_ui.append(html.Div([html.I(className="bi bi-info-circle-fill me-2 text-info"), html.Span(f"Summary for {MODEL_NAMES.get(m_type, m_type.upper())}", className="fw-bold"), html.P("This model uses probabilistic density or neighbors rather than static weights. Interpretability is based on CV stability.", className="text-muted small mt-2")], className="p-3 bg-dark border border-secondary rounded"))
    
    params_ui = format_params(folds_meta[0].get("params", {}))
    return title, params_ui, agg_ui

@app.callback(
    Output("ml-diag-fold-label", "children"),
    Output("ml-diag-modal-content", "children"),
    Input("ml-diag-current-fold", "data"),
    Input("ml-diag-modal", "is_open"),
    State("ml-diag-active-model", "data"),
    State("pipeline-results-store", "data")
)
def render_ml_diag_fold_content(current_idx, is_open, active_model, results_all):
    if not is_open or not active_model or not results_all:
        return "", ""
    
    d_id, m_type = active_model["dataset"], active_model["model"]
    res = results_all.get(d_id, {})
    model_res = res.get("overall_ml", {}).get(m_type, {})
    folds_meta = model_res.get("folds_metadata", [])
    if not folds_meta: return "N/A", ""

    fold_label = f"Fold {current_idx + 1} / {len(folds_meta)}"
    meta = folds_meta[current_idx]
    fold_ui = []
    
    if m_type == "mlp" and "loss_curve" in meta:
        fig = go.Figure(go.Scatter(y=meta["loss_curve"], mode='lines', line=dict(color='#00d1ff')))
        fig.update_layout(title=f"Fold {current_idx+1} Loss Curve", template="plotly_dark", height=300, margin=dict(l=40, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fold_ui.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    elif "feature_importances" in meta:
        imps = meta["feature_importances"]; names = meta.get("feature_names", [f"F{i}" for i in range(len(imps))])
        idx = np.argsort(imps)[-15:]; fig = go.Figure(go.Bar(x=[imps[i] for i in idx], y=[names[i] for i in idx], orientation='h', marker_color='#00d1ff'))
        fig.update_layout(title=f"Fold {current_idx+1} Feature Importance", template="plotly_dark", height=300, margin=dict(l=100, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fold_ui.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    elif "coef" in meta:
        coefs = np.array(meta["coef"]).flatten(); names = meta.get("feature_names", [f"F{i}" for i in range(len(coefs))])
        idx = np.argsort(np.abs(coefs))[-15:]; fig = go.Figure(go.Bar(x=[coefs[i] for i in idx], y=[names[i] for i in idx], orientation='h', marker_color='#ff8c00'))
        fig.update_layout(title=f"Fold {current_idx+1} Coefficients", template="plotly_dark", height=300, margin=dict(l=100, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fold_ui.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    else: fold_ui.append(html.Pre(json.dumps(meta, indent=2), className="text-muted xx-small"))

    return fold_label, fold_ui

def format_params(params):
    if not params: return html.Small("No parameter metadata captured.", className="text-muted italic")
    # Group into badges
    badges = []
    for k, v in sorted(params.items()):
        # Prettify value
        if isinstance(v, float): v_str = f"{v:.4g}"
        elif isinstance(v, list): v_str = f"[{len(v)} items]" if len(v) > 5 else str(v)
        else: v_str = str(v)
        
        badges.append(html.Div([
            html.Small(f"{k}:", className="xx-small text-muted fw-bold d-block text-uppercase"),
            html.Div(v_str, className="small text-info text-truncate", style={"maxWidth": "150px"})
        ], className="m-1 p-2 border border-secondary rounded bg-dark d-inline-block", style={"minWidth": "160px"}))
    
    return html.Div(badges, className="d-flex flex-wrap")

@app.callback(
    Output({"type": "pipeline-token-est", "index": ALL}, "children"),
    Input({"type": "pipeline-ctx-size", "index": ALL}, "value"),
    Input({"type": "pipeline-gen-size", "index": ALL}, "value"),
    Input("pipeline-global-folds", "value"),
    State("datasets-metadata-store", "data"),
    State({"type": "pipeline-ctx-size", "index": ALL}, "id"),
    prevent_initial_call=False
)
def update_pipeline_token_estimates(ctx_sizes, gen_sizes, folds, metadata, ctx_ids):
    if not metadata or not ctx_ids: 
        return [dash.no_update] * len(ctx_ids)
    
    results = []
    TOKEN_RATIO = 2.8 # Heuristic for dense JSON
    TRAIN_FRAC = 0.8
    
    for i, id_obj in enumerate(ctx_ids):
        d_id = id_obj["index"]
        info = metadata.get(d_id, {})
        rows, cols = info.get("rows", 0), info.get("cols", 0)
        
        ctx_pct = ctx_sizes[i] if i < len(ctx_sizes) else 20
        gen_pct = gen_sizes[i] if i < len(gen_sizes) else 80
        
        avg_row_chars = cols * 20 
        n_ctx = int(rows * TRAIN_FRAC * (ctx_pct / 100))
        n_gen = int(rows * TRAIN_FRAC * (gen_pct / 100))
        
        # In benchmark mode, each fold generates n_gen rows.
        # We estimate per fold.
        in_tok = 600 + int((n_ctx * avg_row_chars) / TOKEN_RATIO)
        out_tok = int((n_gen * avg_row_chars) / TOKEN_RATIO)
        
        results.append(html.Div([
            html.Span([html.I(className="bi bi-box-arrow-in-right me-1"), f"~{in_tok}"], className="me-2"),
            html.Span([html.I(className="bi bi-box-arrow-right me-1"), f"~{out_tok}"])
        ], className="d-flex justify-content-end align-items-center"))
    
    return results
