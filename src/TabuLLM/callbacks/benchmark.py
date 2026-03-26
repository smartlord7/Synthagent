import pandas as pd
import json
import dash
from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
from core import app, logger
from services.llm_client import generate_bulk_data
from utils.metrics import calculate_metrics

@app.callback(
    Output("bench-state-store", "data"),
    Output("bench-progress-collapse", "is_open"),
    Input("btn-run-benchmark", "n_clicks"),
    State("gen-max-context", "value"),
    State("bench-folds", "value"),
    prevent_initial_call=True
)
def start_benchmark(n_clicks, max_context, folds):
    logger.info(f"BENCHMARK START: n_clicks={n_clicks}, max_context={max_context}, folds={folds}")
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
    Output("bench-state-store", "data", allow_duplicate=True),
    Output("eda-dashboard-container", "children", allow_duplicate=True),
    Input("bench-state-store", "data"),
    State("original-data-store", "data"),
    State("data-grid", "columnDefs"),
    State('algo-selector', 'value'),
    State('param-tone', 'value'),
    State('param-consistency', 'value'),
    State('param-uniqueness', 'value'),
    State('temperature-slider', 'value'),
    State('dataset-description-store', 'data'),
    State('gen-chunk-size', 'value'),
    State('gen-max-context', 'value'),
    State('gen-context-strategy', 'value'),
    background=True,
    progress=[Output("bench-progress-bar", "value"), Output("bench-progress-bar", "label")],
    prevent_initial_call=True
)
def run_benchmark_fold(set_progress, state, orig_data, col_defs, algorithm, 
                       param_tone, param_consistency, param_uniqueness, temperature, 
                       desc_metadata, chunk_size, max_context, strategy):

    print(f"DEBUG: run_benchmark_fold triggered. state={state}", flush=True)
    if not state or not state.get("active") or state.get("k_total") is None or state.get("k_current", 0) >= state.get("k_total"):
        return dash.no_update, dash.no_update
    
    try:
        current_fold_idx = int(state.get("k_current", 0))
        total_folds = int(state.get("k_total", 3))
        # Derive fraction from max_context and dataset size
        # Cap at 0.8 to ensure at least 20% is reserved for generation in Benchmark
        max_context_val = int(state.get("max_context", 60))
        df_orig = pd.DataFrame(orig_data)
        fraction = min(0.8, max_context_val / len(df_orig)) if not df_orig.empty else 0.2
    except (ValueError, TypeError):
        current_fold_idx = 0
        total_folds = 3
        fraction = 0.2

    all_synthetic_rows = []
    df_orig = pd.DataFrame(orig_data)

    for fold_num in range(current_fold_idx + 1, total_folds + 1):
        logger.info(f"--- Running Benchmark Fold {fold_num}/{total_folds} (Strategy: {strategy}) ---")
        set_progress((int(((fold_num - 1) / total_folds) * 100), f"Running Fold {fold_num}/{total_folds}..."))
        
        try:
            df_orig_c = df_orig.copy()
            if '_row_id' in df_orig_c.columns:
                df_orig_c.drop(columns=['_row_id'], inplace=True, errors='ignore')
                
            field_to_header = {
                col['field']: col.get('headerName', col['field'])
                for col in col_defs
                if col['field'] not in ['ai_audit', '_row_id'] and not col['field'].startswith('is_ai_')
            }
            
            cols_to_keep = [c for c in df_orig_c.columns if c in field_to_header]
            df_orig_c = df_orig_c[cols_to_keep].rename(columns=field_to_header)
            
            # --- CONTEXT STRATEGY APPLICATION ---
            from services.data_manager import select_context_rows
            n_target = int(len(df_orig_c) * fraction)
            n_context_rows = min(n_target, max_context if max_context else n_target)
            
            context_data_list = select_context_rows(df_orig_c, strategy=strategy, n_rows=n_context_rows)
            context_df = pd.DataFrame(context_data_list)
            
            num_empty = len(df_orig_c) - len(context_df)
            
            # Pack context as dictionary structure for bulk downstreams
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
                all_synthetic_rows.extend(filled_data) # Accumulate!
                
                generated_df = pd.DataFrame(filled_data)
                aug_df = pd.concat([context_df, generated_df], ignore_index=True)
                
                from utils.metrics import calculate_metrics
                m = calculate_metrics(df_orig_c, aug_df)
                state["results"].append(m)
            else:
                logger.error(f"Fold {fold_num} generation failed.")
                
            state["k_current"] = fold_num

        except Exception as e:
            import traceback
            logger.error(f"CRITICAL BENCHMARK CRASH ON FOLD {fold_num}: {e}\n{traceback.format_exc()}")
            break

    state["active"] = False
    set_progress((100, f"{state['k_current']} / {total_folds} Folds Complete"))

    # BUILD EDA LAYOUT OVER AGGREGATE IF WE HAVE DATA
    eda_layout = []
    if all_synthetic_rows:
        try:
            super_aug_df = pd.DataFrame(all_synthetic_rows)
            from callbacks.eda import get_eda_layout
            # Prepare standard context df columns aligned with original for get_eda_layout
            df_orig_p = df_orig_c.copy() if 'df_orig_c' in locals() else df_orig
            eda_layout = get_eda_layout(df_orig_p, super_aug_df)
        except Exception as e:
            logger.error(f"Failed to build EDA layout from benchmark aggregate: {e}")
            eda_layout = [dbc.Alert(f"Completed Benchmark metrics saved, but layout rendering failed: {e}", color="danger")]

    return state, eda_layout








# Deprecated display_benchmark_results removed. Layout now pipes to eda-dashboard-container.


