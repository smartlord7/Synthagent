import pandas as pd
import json
import dash
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from core import app, logger
from services.llm_client import generate_bulk_data
from utils.metrics import calculate_metrics

@app.callback(
    Output("bench-state-store", "data"),
    Output("bench-progress-collapse", "is_open"),
    Input("btn-run-benchmark", "n_clicks"),
    State("bench-fraction", "value"),
    State("bench-folds", "value"),
    prevent_initial_call=True
)
def start_benchmark(n_clicks, fraction, folds):
    print(f"DEBUG: start_benchmark triggered. n_clicks={n_clicks}, active fraction={fraction}, folds={folds}", flush=True)
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    return {
        "active": True,
        "k_total": folds,
        "k_current": 0,
        "fraction": fraction,
        "results": []
    }, True


@app.callback(
    Output("bench-state-store", "data", allow_duplicate=True),
    Output("bench-progress-bar", "value"),
    Output("bench-progress-bar", "label"),
    Input("bench-state-store", "data"),
    State("original-data-store", "data"),
    State("data-grid", "columnDefs"),
    State('algo-selector', 'value'),
    State('param-tone', 'value'),
    State('param-consistency', 'value'),
    State('param-uniqueness', 'value'),
    State('temperature-slider', 'value'),
    prevent_initial_call=True
)
def run_benchmark_fold(state, orig_data, col_defs, algorithm, param_tone, param_consistency, param_uniqueness, temperature):
    print(f"DEBUG: run_benchmark_fold triggered. state={state}", flush=True)
    if not state or not state.get("active") or state["k_current"] >= state["k_total"]:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        current_fold_idx = int(state.get("k_current", 0))
        total_folds = int(state.get("k_total", 3))
        fraction = float(state.get("fraction", 0.2))
    except (ValueError, TypeError):
        current_fold_idx = 0
        total_folds = 3
        fraction = 0.2

    # WE ITERATE ALL REMAINING FOLDS IN MEMORY 
    # This prevents brittle Circular Callbacks that Dash 2.0+ drops for slow REST requests
    for fold_num in range(current_fold_idx + 1, total_folds + 1):
        logger.info(f"--- Running Benchmark Fold {fold_num}/{total_folds} ---")
        
        try:
            df_orig = pd.DataFrame(orig_data)
            if '_row_id' in df_orig.columns:
                df_orig.drop(columns=['_row_id'], inplace=True, errors='ignore')
                
            field_to_header = {
                col['field']: col.get('headerName', col['field'])
                for col in col_defs
                if col['field'] not in ['ai_audit', '_row_id']
                   and not col['field'].startswith('is_ai_')
            }
            
            cols_to_keep = [c for c in df_orig.columns if c in field_to_header]
            df_orig = df_orig[cols_to_keep].rename(columns=field_to_header)
            
            context_df = df_orig.sample(frac=fraction)
            num_empty = len(df_orig) - len(context_df)
            
            context_str = f"General Dataset Patterns: {context_df.to_json(orient='records')}"
            instructions = f"Style: {param_tone}\nConsistency: {param_consistency}\nUniqueness: {param_uniqueness}"
            
            result_bundle = generate_bulk_data(
                [], context_str, list(field_to_header.values()),
                algorithm, instructions, temperature, num_empty=num_empty
            )
            
            if result_bundle:
                filled_data = result_bundle["data"]
                generated_df = pd.DataFrame(filled_data)
                aug_df = pd.concat([context_df, generated_df], ignore_index=True)
                
                logger.info(f"Fold {fold_num} Augmented Size: {len(aug_df)}. Calculating metrics...")
                m = calculate_metrics(df_orig, aug_df)
                state["results"].append(m)
            else:
                logger.error(f"Fold {fold_num} generation failed.")
                
            state["k_current"] = fold_num
            logger.info(f"Fold {fold_num} completed.")

        except Exception as e:
            import traceback
            logger.error(f"CRITICAL BENCHMARK CRASH ON FOLD {fold_num}: {e}\n{traceback.format_exc()}")
            break # Exit loop on hard crash

    # After iterating all folds or breaking
    state["active"] = False
    progress_val = 100
    progress_label = f"{state['k_current']} / {total_folds} Folds Complete"
    
    return state, progress_val, progress_label






@app.callback(
    Output("bench-results-container", "children"),
    Input("bench-state-store", "data"),
    prevent_initial_call=True
)
def display_benchmark_results(state):
    if not state or state.get("active") or state["k_current"] == 0:
        return dash.no_update
        
    results = state["results"]
    if not results:
        return dbc.Alert("Benchmark failed to produce results.", color="danger")
        
    # Average the results
    avg_metrics = {}
    for key in results[0].keys():
        avg_metrics[key] = sum(r[key] for r in results) / len(results)
        
    score_color = "success" if avg_metrics["fidelity_score"] > 80 else "warning" if avg_metrics["fidelity_score"] > 50 else "danger"

    return dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-award me-2"), f"Benchmark Results Average ({state['k_total']} Folds at {state['fraction']*100}% Context)"], className="fw-bold bg-light"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{int(avg_metrics['fidelity_score'])}%", className=f"text-{score_color} text-center mb-0"),
                    html.P("Avg. Fidelity (JSD)", className="small text-muted text-center")
                ], width=3),
                dbc.Col([
                    html.H4(f"{avg_metrics['corr_stability']:.1f}%", className="text-info text-center mb-0"),
                    html.P("Avg. Correlation Stability", className="small text-muted text-center")
                ], width=3),
                dbc.Col([
                    html.H4(f"{avg_metrics['avg_z_drift']:.2f}", className="text-warning text-center mb-0"),
                    html.P("Avg. Z-Score Drift", className="small text-muted text-center")
                ], width=3),
                dbc.Col([
                    html.H4(f"{avg_metrics['nndr']:.2f}", className="text-secondary text-center mb-0"),
                    html.P("Avg. NNDR (Novelty)", className="small text-muted text-center")
                ], width=3),
            ])
        ])
    ], className="mt-3 border-primary")
