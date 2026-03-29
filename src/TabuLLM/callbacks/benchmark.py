import pandas as pd
import json
import dash
from dash import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
from core import app
from utils.logging_config import logger
from services.llm_client import generate_bulk_data
from utils.metrics import calculate_metrics

@app.callback(
    Output({"type": "bench-state-store", "index": MATCH}, "data"),
    Output({"type": "bench-progress-collapse", "index": MATCH}, "is_open"),
    Input({"type": "btn-run-benchmark", "index": MATCH}, "n_clicks"),
    State({"type": "gen-max-context", "index": MATCH}, "value"),
    State({"type": "bench-folds", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def start_benchmark_pm(n_clicks, max_context, folds):
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    return {
        "active": True,
        "k_total": int(folds) if folds is not None else 3,
        "k_current": 0,
        "max_context": int(max_context) if max_context is not None else 60,
        "results": []
    }, True


@app.callback(
    Output({"type": "bench-state-store", "index": MATCH}, "data", allow_duplicate=True),
    Output({"type": "eda-dashboard-container", "index": MATCH}, "children", allow_duplicate=True),
    Output({"type": "bench-progress-bar", "index": MATCH}, "value"),
    Output({"type": "bench-progress-bar", "index": MATCH}, "label"),
    Input({"type": "bench-state-store", "index": MATCH}, "data"),
    State({"type": "original-data-store", "index": MATCH}, "data"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    State({"type": "algo-selector", "index": MATCH}, "value"),
    State({"type": "param-tone", "index": MATCH}, "value"),
    State({"type": "param-consistency", "index": MATCH}, "value"),
    State({"type": "param-uniqueness", "index": MATCH}, "value"),
    State({"type": "temperature-slider", "index": MATCH}, "value"),
    State({"type": "dataset-description-store", "index": MATCH}, "data"),
    State({"type": "gen-chunk-size", "index": MATCH}, "value"),
    State({"type": "gen-max-context", "index": MATCH}, "value"),
    State({"type": "gen-context-strategy", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def run_benchmark_fold_pm(state, orig_data, col_defs, algorithm, 
                           param_tone, param_consistency, param_uniqueness, temperature, 
                           desc_metadata, chunk_size, max_context, strategy):

    if not state or not state.get("active") or state.get("k_current", 0) >= state.get("k_total"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        current_fold_idx = int(state.get("k_current", 0))
        total_folds = int(state.get("k_total", 3))
        max_context_val = int(state.get("max_context", 60))
        df_orig = pd.DataFrame(orig_data)
        fraction = min(0.8, max_context_val / len(df_orig)) if not df_orig.empty else 0.2
    except (ValueError, TypeError):
        current_fold_idx = 0
        total_folds = 3
        fraction = 0.2

    all_synthetic_rows = []
    df_orig = pd.DataFrame(orig_data)

    # We remove background=True so we loop through all folds in one request
    # NOTE: This might be slow for many folds, but it bypasses the WildcardInLongCallback error.
    for fold_num in range(current_fold_idx + 1, total_folds + 1):
        logger.info(f"Running Fold {fold_num}/{total_folds}...")
        
        try:
            df_orig_c = df_orig.copy()
            if '_row_id' in df_orig_c.columns:
                df_orig_c.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
                
            field_to_header = {
                col['field']: col.get('headerName', col['field'])
                for col in col_defs
                if col['field'] not in ['ai_audit', '_row_id'] and not col['field'].startswith('is_ai_')
            }
            
            df_orig_c = df_orig_c.rename(columns=field_to_header)
            
            from services.data_manager import select_context_rows
            n_target = int(len(df_orig_c) * fraction)
            n_context_rows = min(n_target, max_context if max_context else n_target)
            
            context_data_list = select_context_rows(df_orig_c, strategy=strategy, n_rows=n_context_rows)
            context_df = pd.DataFrame(context_data_list)
            
            num_empty = len(df_orig_c) - len(context_df)
            
            context_dict = {
                "metadata": desc_metadata if desc_metadata else "Not provided",
                "manual_ctx": "[]",
                "global_ctx": context_data_list
            }

            instructions = f"Style: {param_tone}\nConsistency: {param_consistency}\nUniqueness: {param_uniqueness}"
            
            result_bundle = generate_bulk_data(
                [], context_dict, list(field_to_header.values()),
                algorithm, instructions, temperature, num_empty=num_empty,
                chunk_size=chunk_size
            )

            if result_bundle:
                filled_data = result_bundle["data"]
                all_synthetic_rows.extend(filled_data)
                
                generated_df = pd.DataFrame(filled_data)
                aug_df = pd.concat([context_df, generated_df], ignore_index=True)
                
                m = calculate_metrics(df_orig_c, aug_df)
                state["results"].append(m)
            
            state["k_current"] = fold_num

        except Exception as e:
            logger.error(f"Benchmark fold {fold_num} error: {e}")
            break

    state["active"] = False
    
    eda_layout = []
    if all_synthetic_rows:
        try:
            super_aug_df = pd.DataFrame(all_synthetic_rows)
            from callbacks.eda import get_eda_layout
            eda_layout = get_eda_layout(df_orig_c, super_aug_df, "bench")
        except Exception as e:
            logger.error(f"EDA layout error: {e}")
            eda_layout = [dbc.Alert(f"Completed Benchmark metrics saved, but layout failed: {e}", color="danger")]

    return state, eda_layout, 100, f"{state['k_current']} / {total_folds} Folds Complete"
