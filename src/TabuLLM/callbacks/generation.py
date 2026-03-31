import json
from datetime import datetime
import pandas as pd
import dash
from dash import Input, Output, State, ALL, MATCH
from core import app
from utils.logging_config import logger
from services.llm_client import generate_bulk_data, provider
from dash import callback_context

# --- Sync Local Models ---
@app.callback(
    Output({"type": "algo-selector", "index": MATCH}, "options"),
    Output({"type": "algo-selector", "index": MATCH}, "value"),
    Output({"type": "gen-max-context", "index": MATCH}, "max"),
    Output({"type": "gen-max-context", "index": MATCH}, "value"),
    Input({"type": "btn-refresh-models", "index": MATCH}, "n_clicks"),
    Input({"type": "algo-selector", "index": MATCH}, "value"),
    State({"type": "algo-selector", "index": MATCH}, "options"),
    prevent_initial_call=False
)
def refresh_local_models(n_clicks, current_value, existing_options):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
    
    # 1. Fetch info if Refresh clicked or initial load
    if "btn-refresh-models" in triggered or not triggered:
        models_info = provider.get_available_models()
        options = [{"label": m["id"].split("/")[-1].title(), "value": m["id"]} for m in models_info]
        ids = [m["id"] for m in models_info]
        new_value = current_value if current_value in ids else (ids[0] if ids else None)
    else:
        # Just use existing options if selection changed
        options = existing_options
        new_value = current_value
        # Re-fetch models info to get context limit for the new current_value
        models_info = provider.get_available_models()

    # Find context length
    ctx_limit = 4096
    for m in models_info:
        if m["id"] == new_value:
            ctx_limit = m["max_context"]
            break
            
    safe_default = int(ctx_limit * 0.85) if ctx_limit else 1000
    return options, new_value, ctx_limit, safe_default

# --- THE MASTER AI GENERATION ENGINE ---
@app.callback(
    Output({'type': 'data-grid', 'index': MATCH}, 'rowData', allow_duplicate=True),
    Output({'type': 'target-row-ids', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'btn-generate', 'index': MATCH}, 'n_clicks'),
    State({'type': 'data-grid', 'index': MATCH}, 'rowData'),
    State({'type': 'target-row-ids', 'index': MATCH}, 'data'),  # Target Row IDs (Red)
    State({'type': 'context-row-ids', 'index': MATCH}, 'data'),  # Context Row IDs (Blue)
    State({'type': 'check-full-context', 'index': MATCH}, 'value'),  # Full Dataset Context Toggle
    State({'type': 'data-grid', 'index': MATCH}, 'columnDefs'),
    State({'type': 'algo-selector', 'index': MATCH}, 'value'),
    State({'type': 'param-tone', 'index': MATCH}, 'value'),
    State({'type': 'param-consistency', 'index': MATCH}, 'value'),
    State({'type': 'temperature-slider', 'index': MATCH}, 'value'),
    State({'type': 'cp-name', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-type', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-val-text', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-val-min', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'cp-val-max', 'index': MATCH, 'row': ALL}, 'value'),
    State({'type': 'dataset-description-store', 'index': MATCH}, 'data'),
    State({'type': 'gen-chunk-size', 'index': MATCH}, 'value'),
    State({'type': 'gen-max-context', 'index': MATCH}, 'value'),
    State({'type': 'gen-context-strategy', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def run_ai_generation_pm(n_clicks, row_data, target_ids, context_ids, full_ctx_toggle,
                         col_defs, algorithm, param_tone, param_consistency,
                         temperature, cp_names, cp_types,
                         cp_texts, cp_mins, cp_maxs, desc_metadata, chunk_size, max_context, strategy):

    if not n_clicks or not target_ids:
        logger.warning("Generation aborted: No target rows (Red) defined.")
        return dash.no_update, dash.no_update

    logger.info(f"Starting AI Generation for {len(target_ids)} target rows.")

    # 1. Build Instructions
    compiled_instructions = (
        f"Style: {param_tone}\n"
        f"Consistency: {param_consistency}\n"
    )
    for i in range(len(cp_names)):
        if cp_names[i]:
            detail = cp_texts[i] if cp_types[i] != 'number' else f"Min:{cp_mins[i]}, Max:{cp_maxs[i]}"
            compiled_instructions += f"- Rule [{cp_names[i]}]: {detail}\n"

    # LOG: Instructions
    logger.info(f"--- PROMPT COMPONENT: INSTRUCTIONS ---\n{compiled_instructions}")

    # 2. Context Prep
    df_full = pd.DataFrame(row_data)

    # A. Manual Context (Blue Rows)
    manual_ctx_df = df_full[df_full['_row_id'].isin(context_ids or [])].drop(
        columns=['_row_id', 'is_context', 'is_target'], errors='ignore')
    
    from utils.json_utils import to_compact_json
    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs
                       if col['field'] not in ['ai_audit', '_row_id'] and not col['field'].startswith('is_ai_')}
    header_to_field = {v: k for k, v in field_to_header.items()}
    
    manual_ctx_records = manual_ctx_df.to_dict('records')
    # Use headers for the compact json instead of internal fields
    for r in manual_ctx_records:
        for k, v in list(r.items()):
            if k in field_to_header:
                r[field_to_header[k]] = v
                del r[k]
    manual_ctx_json = json.dumps(manual_ctx_records)

    # B. Global Patterns (Full Dataset Toggle)
    global_ctx_sample = []
    if full_ctx_toggle:
        from services.data_manager import select_context_rows
        max_c = max_context if max_context else 100
        
        logger.info(f"Retrieving context sample using Strategy: {strategy} for {max_c} rows.")
        # Filter df_full to only relevant columns and rename them
        df_full_cleaned = df_full.copy()
        for k in ['_row_id', 'is_context', 'is_target', 'ai_audit']:
            if k in df_full_cleaned.columns: df_full_cleaned.drop(columns=[k], inplace=True)
        
        df_full_cleaned = df_full_cleaned.rename(columns=field_to_header)
        global_ctx_sample = select_context_rows(df_full_cleaned, strategy=strategy, n_rows=max_c)

    # Pack context as Dictionary
    context_data = {
        "metadata": desc_metadata if desc_metadata else "Not provided",
        "manual_ctx": manual_ctx_json,
        "global_ctx": global_ctx_sample
    }

    # 3. Target Mapping
    updated_rows = list(row_data)
    partial_payload = []
    empty_row_count = 0
    target_indices = []

    for t_id in target_ids:
        idx = next((i for i, r in enumerate(updated_rows) if r.get('_row_id') == t_id), -1)
        if idx != -1:
            row_obj = updated_rows[idx]
            guidance = {field_to_header[f]: v for f, v in row_obj.items()
                        if f in field_to_header and v not in [None, "", "NaN"]}
            target_indices.append(idx)
            if len(guidance) > 0:
                partial_payload.append(guidance)
            else:
                empty_row_count += 1

    # 4. Call AI
    result_bundle = generate_bulk_data(
        partial_payload,
        context_data,
        list(field_to_header.values()),
        algorithm,
        compiled_instructions,
        temperature,
        num_empty=empty_row_count,
        chunk_size=chunk_size
    )

    remaining_targets = list(target_ids)

    # 5. Process Results
    if result_bundle:
        filled_data = result_bundle["data"]
        stats = result_bundle["stats"]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        for i, gen_row in enumerate(filled_data):
            if i >= len(target_indices): break
            master_idx = target_indices[i]
            row_uuid = updated_rows[master_idx]['_row_id']
            updated_rows[master_idx]['ai_audit'] = f"Gen AI | {algorithm} | {stats['total_tokens']} tokens | {current_time}"

            for header, val in gen_row.items():
                if header in header_to_field:
                    field = header_to_field[header]
                    sanitized_val = val
                    if isinstance(val, str):
                        clean_str = val.replace('$', '').replace(',', '').strip()
                        try:
                            sanitized_val = float(clean_str) if '.' in clean_str else int(clean_str)
                        except:
                            sanitized_val = val

                    if str(updated_rows[master_idx].get(field)) in [None, "", "NaN", "None"]:
                        updated_rows[master_idx][field] = sanitized_val
                        updated_rows[master_idx][f"is_ai_{field}"] = True

            updated_rows[master_idx]['is_target'] = False
            if row_uuid in remaining_targets:
                remaining_targets.remove(row_uuid)
    
    return updated_rows, remaining_targets


@app.callback(
    Output({'type': 'estimated-tokens-lbl', 'index': MATCH}, 'children'),
    Input({'type': 'gen-max-context', 'index': MATCH}, 'value'),
    Input({'type': 'check-full-context', 'index': MATCH}, 'value'),
    State({'type': 'data-grid', 'index': MATCH}, 'rowData'),
    State({'type': 'context-row-ids', 'index': MATCH}, 'data'),
    State({'type': 'dataset-description-store', 'index': MATCH}, 'data')
)
def estimate_context_tokens_pm(max_rows, full_ctx_toggle, row_data, context_ids, desc_metadata):
    if not row_data:
        return "~0 Tok"
    try:
        import json
        import pandas as pd
        df = pd.DataFrame(row_data)
        
        from utils.json_utils import to_compact_json
        
        manual_ctx_df = df[df['_row_id'].isin(context_ids or [])].drop(
            columns=['_row_id', 'is_context', 'is_target'], errors='ignore')
        all_cols = manual_ctx_df.columns.tolist()
        manual_ctx_rows = manual_ctx_df.to_dict('records')
        manual_ctx_compact = to_compact_json(manual_ctx_rows, all_cols)
        manual_ctx_str = json.dumps(manual_ctx_compact)
        
        global_ctx_len = 0
        clean_df = df.dropna(how='all') if not df.empty else df
        n_rows = min(max_rows if max_rows else 0, len(df))
        
        if not clean_df.empty:
            sample_count = min(5, len(clean_df))
            sample_rows_df = clean_df.sample(sample_count)
            for k in ['_row_id', 'is_context', 'is_target']: 
                if k in sample_rows_df.columns: sample_rows_df = sample_rows_df.drop(columns=[k], errors='ignore')
            
            all_cols_g = sample_rows_df.columns.tolist()
            sample_records = sample_rows_df.to_dict('records')
            compact_sample = to_compact_json(sample_records, all_cols_g)
            
            header_len = len(json.dumps(compact_sample['cols']))
            rows_only_len = len(json.dumps(compact_sample['data']))
            avg_row_char_len = rows_only_len / sample_count
            global_ctx_len = header_len + (avg_row_char_len * n_rows)

        # Compact JSON Heuristic:
        # Compact format {"cols": [...], "data": [[...]]} is much more efficient than 
        # standard list-of-dicts, but has high symbol density (brackets, commas).
        # We use 1:2.7 for compact JSON and 1:3.5 for descriptive metadata.
        text_tokens = (len(desc_metadata) / 3.5) if desc_metadata else 0
        
        # Structure tokens for 'manual_ctx_str' and 'global_ctx_len'
        # Compact JSON is dense with structural tokens (brackets, commas, quotes).
        json_tokens_est = (len(manual_ctx_str) + global_ctx_len) / 2.7
        
        # Prompt overhead: System instructions, template framing, column definitions.
        # This is roughly constant regardless of dataset size.
        prompt_overhead = 480 
        
        est_tokens = int(text_tokens + json_tokens_est + prompt_overhead)
        fraction = (n_rows / len(df)) * 100 if len(df) > 0 else 0
        suffix = " (Off)" if not full_ctx_toggle else ""
        
        return f"~{est_tokens} Tok ({fraction:.0f}%){suffix}"
    except Exception as e:
        logger.error(f"Token estimation error: {e}")
        return "~Err Tok"
