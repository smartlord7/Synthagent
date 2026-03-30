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
    State({"type": "column-roles-store", "index": ALL}, "data"),
    State({"type": "column-roles-store", "index": ALL}, "id"),
    prevent_initial_call=True
)
def render_pipeline_ui_in_modal(n_clicks, metadata, roles_vals, roles_ids):
    if not metadata:
        return dbc.Alert("No datasets uploaded. Upload files to start a pipeline.", color="info", className="m-0")
    
    # Map roles by dataset_id
    roles_map = {id_obj["index"]: val for id_obj, val in zip(roles_ids, roles_vals)}
    
    rows = []
    # Sort by filename or ID for stable display
    for dataset_id in sorted(metadata.keys()):
        info = metadata[dataset_id]
        if dataset_id == "default-dataset" and info.get("rows", 0) <= 10:
             if info.get("rows", 0) <= 0 or info.get("filename") == "Workbench":
                 continue

        size_str = f"{info.get('rows', 0)}R × {info.get('cols', 0)}C"
        
        # Pre-fill target from roles_map
        existing_targets = roles_map.get(dataset_id, {}).get("targets", [])
        target_val = ",".join(existing_targets) if existing_targets else ""

        rows.append(dbc.ListGroupItem([
            dbc.Row([
                dbc.Col(html.Span(info["filename"], className="fw-bold text-info small truncate-text", style={"maxWidth": "120px"}), width=3),
                dbc.Col(html.Small(size_str, className="text-muted d-block", style={"fontSize": "0.7rem"}), width=2),
                dbc.Col(dbc.Input(id={"type": "pipeline-target", "index": dataset_id}, placeholder="Target", value=target_val, size="sm", className="bg-dark text-light border-secondary"), width=4),
                dbc.Col([
                    dbc.Input(id={"type": "pipeline-priority", "index": dataset_id}, type="number", value=1, size="sm", style={"width": "40px"}),
                    dbc.Checkbox(id={"type": "pipeline-include", "index": dataset_id}, value=True, className="ms-2")
                ], width=3, className="d-flex align-items-center justify-content-end"),
            ], align="center", className="gx-1")
        ], className="bg-dark border-secondary text-light py-1 px-2"))

    if not rows:
        return dbc.Alert("No valid datasets found for pipeline.", color="warning", className="m-0")

    ui = html.Div([
        html.P("Execute parallel validation for selected datasets (Lower priority runs first).", className="text-muted small mb-3"),
        
        dbc.Checkbox(id="pipeline-auto-detect-targets", label="Auto-detect missing targets via LLM", value=True, className="mb-3 text-info small"),

        dbc.ListGroup(rows, className="mb-3 border-secondary"),
        
        dbc.Row([
            dbc.Col(html.Label("K-Folds (Global):", className="small text-muted fw-bold"), width=3),
            dbc.Col(dbc.Input(id="pipeline-global-folds", type="number", value=3, min=1, max=10, size="sm", className="bg-dark text-light border-secondary"), width=3),
        ], className="mb-3 align-items-center g-2"),

        html.Hr(className="border-secondary"),
        
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Models:", className="text-muted small fw-bold mb-2 d-block"),
                        dbc.Checklist(
                            id="pipeline-ml-model-checklist",
                            options=[
                                {"label": "LogReg/Linear", "value": "logistic"},
                                {"label": "Random Forest", "value": "rf"},
                                {"label": "XGBoost", "value": "xgb"},
                                {"label": "Neural Net", "value": "mlp"}
                            ],
                            value=[], switch=True, className="text-light small",
                            style={"fontSize": "0.8rem"}
                        )
                    ], md=3),
                    dbc.Col([
                        html.Div(id="pipeline-ml-model-config-container", className="ps-2 border-start border-secondary")
                    ], md=9)
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
    # Access to the model configuration inputs in the modal
    State({"type": "ml-param", "index": ALL}, "value"),
    State({"type": "ml-param", "index": ALL}, "id"),
    State("datasets-metadata-store", "data"),
    # Modal-specific states
    State({"type": "pipeline-include", "index": ALL}, "value"),
    State({"type": "pipeline-include", "index": ALL}, "id"),
    State({"type": "pipeline-priority", "index": ALL}, "value"),
    # Tab-specific states
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
    # ML Configs
    State({"type": "ml-config-store", "index": ALL}, "data"),
    State({"type": "ml-config-store", "index": ALL}, "id"),
    State({"type": "column-roles-store", "index": ALL}, "data"),
    State({"type": "column-roles-store", "index": ALL}, "id"),
    # Containers for ALL tabs (must match Output ALL order)
    State({"type": "eda-dashboard-container", "index": ALL}, "id"),
    State({"type": "ml-evaluation-results-dashboard", "index": ALL}, "id"),
    State({"type": "pipeline-target", "index": ALL}, "value"),
    State({"type": "pipeline-target", "index": ALL}, "id"),
    State({"type": "target-cols-dropdown", "index": ALL}, "id"),
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
                        ml_config_list, ml_ids,
                        roles_list, roles_ids,
                        eda_ids, ml_res_ids,
                        dataset_targets_vals, dataset_targets_ids,
                        target_dropdown_ids):
    
    if not n_clicks: 
        return (dash.no_update, dash.no_update, [dash.no_update]*len(eda_ids), 
                [dash.no_update]*len(ml_res_ids), [dash.no_update]*len(dataset_targets_ids), 
                [dash.no_update]*len(target_dropdown_ids), [dash.no_update]*len(roles_ids), dash.no_update)
    
    # helper to map ALL states back to dataset_id
    def map_states(values, ids):
        return {id_obj["index"]: values[i] for i, id_obj in enumerate(ids)}

    # Map all tab-specific config by dataset_id
    orig_map = map_states(orig_data_list, orig_ids)
    col_map = map_states(col_defs_list, col_ids)
    max_ctx_map = map_states(max_ctx_list, max_ctx_ids)
    strategy_map = map_states(strategy_list, strategy_ids)
    algo_map = map_states(algo_list, algo_ids)
    tone_map = map_states(tone_list, tone_ids)
    consist_map = map_states(consist_list, consist_ids)
    uniq_map = map_states(uniq_list, uniq_ids)
    temp_map = map_states(temp_list, temp_ids)
    chunk_map = map_states(chunk_list, chunk_ids)
    desc_map = map_states(desc_list, desc_ids)
    ml_map = map_states(ml_config_list, ml_ids)
    roles_map = map_states(roles_list, roles_ids)
    
    # 1. Parse Global ML Config from Modal
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


    # 2. Build Task list from Modal selections
    tasks_raw = []
    
    # Map for target variables per dataset
    target_map = map_states(dataset_targets_vals, dataset_targets_ids)
    
    for i, id_obj in enumerate(inc_ids):
        d_id = id_obj["index"]
        if not inc_vals[i]: continue
        
        priority = prios[i] if prios[i] is not None else 100
        filename = metadata.get(d_id, {}).get("filename", "Unknown")
        
        # Determine target(s) for this dataset
        raw_target = target_map.get(d_id, "")
        target_cols = [t.strip() for t in raw_target.split(",")] if raw_target else roles_map.get(d_id, {}).get("targets", [])
        
        # LLM Auto-detection if requested and input is empty
        if auto_detect and not raw_target:
            set_progress((f"Detecting target for {filename}...", html.Div(dbc.Spinner(size="sm"))))
            logger.info(f"Triggering auto-detection for {filename} (current input empty)")
            try:
                df_temp = pd.DataFrame(orig_map.get(d_id, []))
                if not df_temp.empty:
                    detected = detect_target_columns(df_temp, single_target=True)
                    logger.info(f"Detected targets for {filename}: {detected}")
                    if detected:
                        roles_map[d_id] = {"targets": detected}
                        target_cols = detected
                        # Update maps for UI feedback
                        target_map[d_id] = ",".join(detected)
                        if d_id in roles_map:
                            roles_map[d_id] = {"targets": detected}
                        else:
                            roles_map[d_id] = {"targets": detected}
            except Exception as e:
                logger.error(f"Auto-detection failed for {filename}: {e}")
        
        config = {
            "folds": int(global_folds) if global_folds else 3,
            "max_context": max_ctx_map.get(d_id, 60),
            "strategy": strategy_map.get(d_id, "random"),
            "algorithm": algo_map.get(d_id, "bulk"),
            "temperature": temp_map.get(d_id, 0.3),
            "chunk_size": chunk_map.get(d_id, 30),
            "instructions": f"Style: {tone_map.get(d_id, 'Concise')}\nConsistency: {consist_map.get(d_id, 'Strict')}\nUniqueness: {uniq_map.get(d_id, 'Allow Duplicates')}",
            "desc_metadata": desc_map.get(d_id, ""),
            "ml_config": global_ml_config,
            "targets": target_cols
        }
        
        tasks_raw.append({
            "priority": priority,
            "payload": (d_id, filename, orig_map.get(d_id, []), col_map.get(d_id, []), config)
        })

    if not tasks_raw:
        return (dbc.Alert("No datasets selected.", color="warning", className="m-0"), "", 
                [dash.no_update]*len(eda_ids), [dash.no_update]*len(ml_res_ids), 
                [dash.no_update]*len(dataset_targets_ids), [dash.no_update]*len(target_dropdown_ids),
                [dash.no_update]*len(roles_ids), dash.no_update)

    tasks_raw.sort(key=lambda x: x["priority"])
    tasks = [t["payload"] for t in tasks_raw]
    
    # 2. Run parallel execution with progress tracking
    try:
        import multiprocessing
        import time
        from multiprocessing import Manager
        
        manager = Manager()
        progress_dict = manager.dict() # key: dataset_id, value: {filename, folds_done, total_folds}
        
        # Add progress_dict to each task's config
        tasks_with_progress = []
        for t in tasks:
            d_id, filename, d, c, cfg = t
            cfg["_progress_dict"] = progress_dict
            cfg["_dataset_id"] = d_id # for worker to know where to write
            tasks_with_progress.append((d_id, filename, d, c, cfg))
            progress_dict[d_id] = {"filename": filename, "done": 0, "total": cfg.get("folds", 3), "status": "Queued"}

        cpus = min(len(tasks), multiprocessing.cpu_count())
        pipeline_results = []
        
        with multiprocessing.Pool(processes=cpus) as pool:
            async_results = [pool.apply_async(run_single_dataset_benchmark, (t,)) for t in tasks_with_progress]
            
            while any(not r.ready() for r in async_results):
                # Build Progress UI
                prog_bars = []
                for d_id, p_info in progress_dict.items():
                    pct = (p_info["done"] / p_info["total"]) * 100 if p_info["total"] > 0 else 0
                    color = "info" if pct < 100 else "success"
                    
                    prog_bars.append(html.Div([
                        dbc.Row([
                            dbc.Col(html.Small(p_info["filename"], className="fw-bold text-info truncate-text", style={"maxWidth": "150px"}), width=4),
                            dbc.Col(dbc.Progress(value=pct, color=color, striped=True, animated=True, style={"height": "8px"}), width=6),
                            dbc.Col(html.Small(f"{int(pct)}%", className="text-light x-small"), width=2),
                        ], align="center", className="mb-2")
                    ]))
                
                set_progress((
                    f"Processing {sum(1 for r in async_results if r.ready())}/{len(tasks)} datasets...",
                    html.Div(prog_bars, className="mt-3 border-top border-secondary pt-3")
                ))
                time.sleep(1)
            
            pipeline_results = [r.get() for r in async_results]
            
            # Explicit cleanup for Windows/multiprocessing
            pool.close()
            pool.join()

    except Exception as e:
        logger.error(f"High-level pipeline failure: {e}")
        return (dbc.Alert(f"Pipeline failed: {e}", color="danger"), dash.no_update, 
                [dash.no_update]*len(eda_ids), [dash.no_update]*len(ml_res_ids),
                [dash.no_update]*len(dataset_targets_ids), [dash.no_update]*len(target_dropdown_ids),
                [dash.no_update]*len(roles_ids), dash.no_update)
    
    # 3. Process Results & Prepare Updates
    result_cards = [html.H5("Results", className="text-light fw-bold mb-3 border-bottom border-secondary pb-2")]
    eda_updates = [dash.no_update] * len(eda_ids)
    ml_updates = [dash.no_update] * len(ml_res_ids)
    
    eda_index_map = {id_obj["index"]: i for i, id_obj in enumerate(eda_ids)}
    ml_index_map = {id_obj["index"]: i for i, id_obj in enumerate(ml_res_ids)}

    for res in pipeline_results:
        d_id = res['dataset_id']
        results_list_m = res.get('results', [])
        if not results_list_m: continue
        
        # --- AGGREGATE ALL METRICS (Fidelity & ML) ---
        agg_metrics = {
            "fidelity": [], "correlation": [], "privacy": [], "coverage": [],
            "jsd": [], "z_drift": [], "entropy": [], "density": [], "accuracy": [], "novelty": []
        }
        
        for r in results_list_m:
            agg_metrics["fidelity"].append(r.get("Synthetic Fidelity (JSD)", 0))
            agg_metrics["correlation"].append(r.get("Correlation Stability", 0))
            agg_metrics["privacy"].append(r.get("Privacy Risk", 0))
            agg_metrics["coverage"].append(r.get("Coverage Score", 0))
        
        ml_agg = {} # {model_type: {metric_name: {"orig": [], "aug": [], "delta": []}}}
        task_type = "classification"
        
        for r in results_list_m:
            ml_eval = r.get("ml_eval", {})
            for m_type, m_perf in ml_eval.items():
                if not m_perf: continue
                task_type = m_perf.get("task_type", "classification")
                if m_type not in ml_agg:
                    ml_agg[m_type] = {}
                
                # Iterate over all metrics returned
                for m_name in ["accuracy", "f1", "rmse", "r2"]:
                    v_orig = (m_perf.get("original") or {}).get(m_name)
                    v_aug = (m_perf.get("augmented") or {}).get(m_name)
                    
                    if v_orig is not None and v_aug is not None:
                        if m_name not in ml_agg[m_type]:
                            ml_agg[m_type][m_name] = {"orig": [], "aug": [], "delta": []}
                        ml_agg[m_type][m_name]["orig"].append(float(v_orig))
                        ml_agg[m_type][m_name]["aug"].append(float(v_aug))
                        ml_agg[m_type][m_name]["delta"].append(float(v_aug - v_orig))

        # --- 1. SYNTHETIC DATA QUALITY COLUMN ---
        def get_stats(vals):
            if not vals: return 0.0, 0.0
            m, s = float(np.mean(vals)), float(np.std(vals))
            return (m if np.isfinite(m) else 0.0), (s if np.isfinite(s) else 0.0)

        f_m, f_s = get_stats(agg_metrics["fidelity"])
        c_m, c_s = get_stats(agg_metrics["correlation"])
        p_m, p_s = get_stats(agg_metrics["privacy"])
        cov_m, cov_s = get_stats(agg_metrics["coverage"])
        
        # Helper for color mapping
        def get_color(val, reverse=False):
            if reverse: # Lower is better (Risk)
                return "success" if val < 20 else "warning" if val < 50 else "danger"
            return "success" if val > 80 else "warning" if val > 50 else "danger"

        f_color = get_color(f_m)
        c_color = get_color(c_m)
        p_color = get_color(p_m, reverse=True)
        cov_color = get_color(cov_m)

        syn_col = html.Div([
            html.Small("SYNTHETIC DATA QUALITY", className="text-muted xx-small fw-bold border-bottom border-secondary d-block mb-2"),
            html.Div([
                # Fidelity
                dbc.Row([
                    dbc.Col(html.Small("Fidelity", className="x-small text-light"), width=4),
                    dbc.Col(html.Small(f"{f_m:.1f}±{f_s:.1f}", className=f"x-small text-{f_color} fw-bold"), width=3),
                    dbc.Col(dbc.Progress(value=f_m, color=f_color, style={"height": "4px", "marginTop": "6px"}), width=5),
                ], align="center", className="g-0 mb-1"),
                # Correlation
                dbc.Row([
                    dbc.Col(html.Small("Correlation", className="x-small text-muted"), width=4),
                    dbc.Col(html.Small(f"{c_m:.1f}±{c_s:.1f}", className=f"x-small text-{c_color}"), width=3),
                    dbc.Col(dbc.Progress(value=c_m, color=c_color, style={"height": "4px", "marginTop": "6px"}), width=5),
                ], align="center", className="g-0 mb-1"),
                # Privacy (Risk)
                dbc.Row([
                    dbc.Col(html.Small("Privacy (Risk)", className="x-small text-muted"), width=4),
                    dbc.Col(html.Small(f"{p_m:.1f}±{p_s:.1f}", className=f"x-small text-{p_color}"), width=3),
                    dbc.Col(dbc.Progress(value=p_m, color=p_color, style={"height": "4px", "marginTop": "6px"}), width=5),
                ], align="center", className="g-0 mb-1"),
                # Coverage
                dbc.Row([
                    dbc.Col(html.Small("Coverage", className="x-small text-muted"), width=4),
                    dbc.Col(html.Small(f"{cov_m:.1f}±{cov_s:.1f}", className=f"x-small text-{cov_color}"), width=3),
                    dbc.Col(dbc.Progress(value=cov_m, color=cov_color, style={"height": "4px", "marginTop": "6px"}), width=5),
                ], align="center", className="g-0"),
            ])
        ])
        
        # --- 2. PREDICTIVE UTILITY COLUMN (ALL MODELS) ---
        ml_rows = []
        primary_metric = "accuracy" if task_type == "classification" else "r2"
        
        for m_type, metrics in ml_agg.items():
            if primary_metric in metrics:
                m_data = metrics[primary_metric]
                orig_m, orig_s = get_stats(m_data["orig"])
                aug_m, aug_s = get_stats(m_data["aug"])
                delta = aug_m - orig_m
                d_color = "success" if (delta >= -0.05 if primary_metric != "rmse" else delta <= 0.05) else "danger"
                
                ml_rows.append(html.Div([
                    dbc.Row([
                        dbc.Col(html.Small(m_type.upper(), className="x-small text-info fw-bold"), width=3),
                        dbc.Col(html.Small(f"{aug_m:.3f}±{aug_s:.2f}", className="x-small text-light"), width=4),
                        dbc.Col(html.Small(f"{orig_m:.3f}", className="x-small text-muted italic"), width=2),
                        dbc.Col(html.Small(f"{delta:+.3f}", className=f"x-small text-{d_color} fw-bold"), width=3),
                    ], className="g-0 border-bottom border-secondary-subtle py-1 align-items-center")
                ]))

        pred_col = html.Div([
            html.Small("PREDICTIVE UTILITY (ALL MODELS)", className="text-muted xx-small fw-bold border-bottom border-secondary d-block mb-2"),
            html.Div([
                dbc.Row([
                    dbc.Col(html.Small("Model", className="xx-small text-muted"), width=3),
                    dbc.Col([
                        html.Small("Aug Score", className="xx-small text-muted", id=f"head-aug-{d_id}"),
                        dbc.Tooltip("Score on Augmented Data (Real Context + Synthetic)", target=f"head-aug-{d_id}")
                    ], width=4),
                    dbc.Col([
                        html.Small("Real", className="xx-small text-muted", id=f"head-real-{d_id}"),
                        dbc.Tooltip("Baseline score on original Real data", target=f"head-real-{d_id}")
                    ], width=2),
                    dbc.Col(html.Small("Delta", className="xx-small text-muted"), width=3),
                ], className="g-0 opacity-75 mb-1"),
                html.Div(ml_rows if ml_rows else html.Small("No models evaluated", className="text-muted italic x-small"))
            ])
        ], className="ps-3 border-start border-secondary")

        # --- COMBINE INTO CARD ---
        card = dbc.Card([
            dbc.CardBody([
                html.Div(res['filename'], className="fw-bold text-info small mb-3"),
                dbc.Row([
                    dbc.Col(syn_col, width=5),
                    dbc.Col(pred_col, width=7)
                ])
            ], className="py-3 px-3")
        ], className="mb-3 border-secondary bg-dark shadow-sm")
        result_cards.append(card)
        
        # --- EDA & ML TABS REFRESH (Keep existing logic) ---
        if d_id in eda_index_map:
            try:
                df_orig = pd.DataFrame(orig_map.get(d_id, []))
                # Ensure df_orig columns match the Header Names used in df_aug
                field_to_header = {c['field']: c.get('headerName', c['field']) for c in col_defs_list[eda_index_map[d_id]] if c['field'] not in ['ai_audit', '_row_id']}
                df_orig.rename(columns=field_to_header, inplace=True)
                
                df_aug = pd.DataFrame(res.get('synthetic_data', []))
                if not df_aug.empty:
                    # 3. Build Detailed EDA Dashboard for this dataset (includes aggregation if results_list provided)
                    eda_updates[eda_index_map[d_id]] = get_eda_layout(df_orig, df_aug, d_id, results_list=results_list_m)
            except Exception as e:
                logger.error(f"Failed to build EDA results for {d_id}: {e}")

        if d_id in ml_index_map:
            ml_cards = []
            for m_type, metrics in ml_agg.items():
                metric_rows = []
                for m_name, data in metrics.items():
                    orig_m, orig_s = get_stats(data["orig"])
                    aug_m, aug_s = get_stats(data["aug"])
                    d_m, d_s = get_stats(data["delta"])
                    diff_color = "success" if (d_m >= -0.05 if m_name != "rmse" else d_m <= 0.05) else "danger"
                    metric_rows.append(dbc.Row([
                        dbc.Col(html.Small(m_name.upper(), className="x-small text-info fw-bold"), width=3),
                        dbc.Col(html.Span(f"{orig_m:.3f}±{orig_s:.2f}", className="x-small text-muted"), width=3),
                        dbc.Col(html.Span(f"{aug_m:.3f}±{aug_s:.2f}", className="x-small text-light fw-bold"), width=3),
                        dbc.Col(html.Span(f"{d_m:+.3f}±{d_s:.2f}", className=f"text-{diff_color} x-small fw-bold"), width=3),
                    ], className="g-0 border-bottom border-secondary-subtle py-1 align-items-center"))
                ml_cards.append(dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="bi bi-cpu me-2 text-info"), html.Span(m_type.upper(), className="fw-bold small")], className="d-flex align-items-center mb-2 border-bottom border-secondary pb-1"),
                        html.Div([
                            dbc.Row([dbc.Col(html.Small("Metric", className="xx-small text-muted"), width=3), dbc.Col(html.Small("Real", className="xx-small text-muted"), width=3), dbc.Col(html.Small("Syn", className="xx-small text-muted"), width=3), dbc.Col(html.Small("Delta", className="xx-small text-muted"), width=3)], className="g-0 opacity-75 mb-1"),
                            html.Div(metric_rows)
                        ])
                    ], className="p-2")
                ], className="mb-2 border-secondary bg-dark shadow-sm", style={"minWidth": "280px", "flex": "1"}))
            if ml_cards:
                ml_updates[ml_index_map[d_id]] = html.Div([
                    html.H6([html.I(className="bi bi-cpu-fill me-2 text-warning"), "ML Evaluation"], className="text-light mb-3 small fw-bold"),
                    html.Div(ml_cards, className="d-flex flex-wrap gap-2")
                ], className="mt-2")

    summary_ui = html.Div(result_cards, style={"maxHeight": "400px", "overflowY": "auto"})
    status_msg = f"Completed {len(pipeline_results)} datasets."
    
    # Prepare sync updates for UI
    target_updates = []
    for i, id_obj in enumerate(dataset_targets_ids):
        d_id = id_obj["index"]
        val = target_map.get(d_id, dataset_targets_vals[i])
        target_updates.append(val)
        
    dropdown_updates = []
    for i, id_obj in enumerate(target_dropdown_ids):
        d_id = id_obj["index"]
        # Match target from target_map if it exists, else keep current or empty
        val = target_map.get(d_id, "") 
        # Dropdown expects a list if multi=True
        dropdown_updates.append(val.split(",") if isinstance(val, str) and "," in val else [val] if val else [])
        
    roles_updates = []
    for i, id_obj in enumerate(roles_ids):
        d_id = id_obj["index"]
        roles_updates.append(roles_map.get(d_id, roles_list[i]))

    # 4. Prepare Global Results Store
    results_dict = {res['dataset_id']: res for res in pipeline_results}

    return status_msg, summary_ui, eda_updates, ml_updates, target_updates, dropdown_updates, roles_updates, results_dict
