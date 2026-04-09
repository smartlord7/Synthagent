import pandas as pd
import json
import dash
from dash import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from core import app
from utils.logging_config import logger
from services.llm_client import generate_bulk_data
from utils.metrics import calculate_metrics

@app.callback(
    Output({"type": "bench-state-store", "index": MATCH}, "data"),
    Output({"type": "bench-progress-collapse", "index": MATCH}, "is_open"),
    Input({"type": "btn-run-benchmark", "index": MATCH}, "n_clicks"),
    State({"type": "bench-folds", "index": MATCH}, "value"),
    State({"type": "bench-join-context", "index": MATCH}, "value"),
    State({"type": "bench-ctx-size", "index": MATCH}, "value"),
    State({"type": "bench-gen-size", "index": MATCH}, "value"),
    State({"type": "check-enforce-rules", "index": MATCH}, "value"),
    State({"type": "check-enforce-boundaries", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def start_benchmark_pm(n_clicks, folds, joins, ctx_size, gen_size, enforce_rules, enforce_boundaries):
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    return {
        "active": True,
        "k_total": int(folds) if folds is not None else 3,
        "k_current": 0,
        "join_context": bool(joins and 1 in joins),
        "ctx_size_pct": ctx_size if ctx_size is not None else 20,
        "gen_size_pct": gen_size if gen_size is not None else 80,
        "enforce_rules": bool(enforce_rules and len(enforce_rules) > 0),
        "enforce_boundaries": bool(enforce_boundaries and len(enforce_boundaries) > 0),
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
    State({"type": "temperature-slider", "index": MATCH}, "value"),
    State({"type": "dataset-description-store", "index": MATCH}, "data"),
    State({"type": "gen-chunk-size", "index": MATCH}, "value"),
    State({"type": "gen-max-context", "index": MATCH}, "value"),
    State({"type": "gen-context-strategy", "index": MATCH}, "value"),
    State({'type': 'cp-name', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-type', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-val-text', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-val-min', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-val-max', 'index': MATCH, 'row': ALL}, 'value'),
    prevent_initial_call=True
)
def run_benchmark_fold_pm(state, orig_data, col_defs, algorithm, 
                           param_tone, param_consistency, temperature, 
                           desc_metadata, chunk_size, max_context, strategy,
                           cp_names, cp_types, cp_texts, cp_mins, cp_maxs):

    if not state or not state.get("active") or state.get("k_current", 0) >= state.get("k_total"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        current_fold_idx = int(state.get("k_current", 0))
        total_folds = int(state.get("k_total", 3))
        join_context = state.get("join_context", True)
        ctx_size_pct = state.get("ctx_size_pct", 20)
        gen_size_pct = state.get("gen_size_pct", 80)
        
        df_orig = pd.DataFrame(orig_data)
    except (ValueError, TypeError):
        current_fold_idx = 0
        total_folds = 3
        join_context = True
        ctx_size_pct = 20
        gen_size_pct = 80
        df_orig = pd.DataFrame()

    all_synthetic_rows = []

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
            
            # Divide training set (simulating fold split - here we just use 80% as base for context)
            df_train_base = df_orig_c.sample(frac=0.8, random_state=42 + fold_num)
            
            n_context_rows = max(1, int(len(df_train_base) * (ctx_size_pct / 100)))
            num_empty = max(1, int(len(df_train_base) * (gen_size_pct / 100)))
            
            context_data_list = select_context_rows(df_train_base, strategy=strategy, n_rows=n_context_rows)
            context_df = pd.DataFrame(context_data_list)
            
            context_dict = {
                "metadata": desc_metadata if desc_metadata else "Not provided",
                "manual_ctx": "[]",
                "global_ctx": context_data_list
            }

            from callbacks.generation import _build_instructions
            instructions, rules_meta = _build_instructions(param_tone, param_consistency, cp_names, cp_types, cp_texts, cp_mins, cp_maxs)
            
            from services.rule_inference import RuleInference
            inferred_rules = RuleInference.infer_dataset_rules(df_orig_c)
            
            rules_hierarchy = {
                "integrity": [r for r in inferred_rules if r['level'] == 1],
                "boundaries": [r for r in inferred_rules if r['level'] == 2],
                "user": rules_meta,
                "enforce_user": state.get("enforce_rules", True),
                "enforce_boundaries": state.get("enforce_boundaries", False)
            }

            result_bundle = generate_bulk_data(
                [], context_dict, list(field_to_header.values()),
                algorithm, instructions, temperature, num_empty=num_empty,
                chunk_size=chunk_size, rules_metadata=rules_hierarchy, enforce_rules=True
            )

            if result_bundle:
                filled_data = result_bundle["data"]
                all_synthetic_rows.extend(filled_data)
                
                generated_df = pd.DataFrame(filled_data)
                
                if join_context:
                    aug_df = pd.concat([context_df, generated_df], ignore_index=True)
                else:
                    aug_df = generated_df # Pure Synthetic
                
                # Fidelity metrics - compare against whole real dataset if relevant, but here we keep calculate_metrics logic
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
