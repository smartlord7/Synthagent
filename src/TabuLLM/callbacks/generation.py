import json
from datetime import datetime
import pandas as pd
import dash
from dash import Input, Output, State, ALL, MATCH
from core import app
from utils.logging_config import logger
from services.llm_client import generate_bulk_data, provider
from dash import callback_context
from utils.json_utils import to_compact_json
from services.data_manager import select_context_rows

# --- Sync Local Models ---
@app.callback(
    Output({"type": "algo-selector", "index": MATCH}, "options"),
    Output({"type": "algo-selector", "index": MATCH}, "value"),
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
        
    return options, new_value

# --- Sync Slider Range with Dataset Size ---
@app.callback(
    Output({"type": "gen-max-context", "index": MATCH}, "max"),
    Output({"type": "gen-max-context", "index": MATCH}, "value"),
    Input({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "gen-max-context", "index": MATCH}, "value"),
    prevent_initial_call=False
)
def sync_max_context_range(row_data, current_val):
    if not row_data:
        return 100, 20
    
    n_rows = len(row_data)
    new_max = max(1, n_rows)
    
    # If it's a new load or the value was at the previous 'default', 
    # snap to the actual number of rows.
    # Otherwise, cap the current selection to the new range.
    if current_val is None or current_val == 60 or current_val > new_max:
        return new_max, new_max
    
    return new_max, current_val

def _build_instructions(tone, consistency, names, types, texts, mins, maxs):
    """Compile rules and parameters into a string."""
    header = f"Style: {tone}\nConsistency: {consistency}\n"
    rules = []
    for i in range(len(names)):
        if names[i]:
            detail = texts[i] if types[i] != 'number' else f"Min:{mins[i]}, Max:{maxs[i]}"
            rules.append(f"- Rule [{names[i]}]: {detail}")
    return header + "\n".join(rules)

def _prepare_context(row_data, context_ids, full_ctx_toggle, strategy, max_rows, col_defs, metadata):
    """Extract and sample reference context data."""
    df = pd.DataFrame(row_data)
    fields = {c['field']: c.get('headerName', c['field']) for c in col_defs
             if c['field'] not in ['ai_audit', '_row_id'] and not c['field'].startswith('is_ai_')}
    
    # Manual Context
    manual_df = df[df['_row_id'].isin(context_ids or [])].drop(columns=['_row_id', 'is_context', 'is_target'], errors='ignore')
    manual_records = manual_df.to_dict('records')
    for r in manual_records:
        for k, v in list(r.items()):
            if k in fields:
                r[fields[k]] = v
                del r[k]
    
    logger.info(f"Context Prep: Found {len(manual_records)} manual (Blue) reference rows.")
    
    # Global Context
    global_sample = []
    if full_ctx_toggle:
        logger.info(f"Context Prep: Sampling {max_rows} rows using strategy '{strategy}'.")
        df_clean = df.copy()
        for k in ['_row_id', 'is_context', 'is_target', 'ai_audit']:
            if k in df_clean.columns: df_clean.drop(columns=[k], inplace=True)
        df_clean = df_clean.rename(columns=fields)
        global_sample = select_context_rows(df_clean, strategy=strategy, n_rows=max_rows or 100)
    else:
        logger.info("Context Prep: Global context strategy is OFF.")

    return {
        "metadata": metadata or "Not provided",
        "manual_ctx": json.dumps(manual_records),
        "global_ctx": global_sample,
        "column_headers": list(fields.values()),
        "header_to_field": {v: k for k, v in fields.items()}
    }

def _map_targets(row_data, target_ids, headers):
    """Identify which rows need completion vs generation."""
    payload = []
    indices = []
    empty_count = 0
    header_to_field = {h: f for f, h in headers.items()} # headers is map from prepare_context
    
    for t_id in target_ids:
        idx = next((i for i, r in enumerate(row_data) if r.get('_row_id') == t_id), -1)
        if idx == -1: continue
        
        row = row_data[idx]
        guidance = {h: row.get(f) for h, f in header_to_field.items() 
                    if row.get(f) not in [None, "", "NaN"]}
        
        indices.append(idx)
        if guidance: payload.append(guidance)
        else: empty_count += 1
    
    logger.info(f"Target Mapping: {len(payload)} partial rows, {empty_count} entirely new rows.")
    return payload, indices, empty_count

def _process_results(rows, indices, bundle, algo, headers_map):
    """Integrate LLM tokens back into raw data strings/numbers. Handles partial successes."""
    if not bundle: 
        logger.error("Result Processing: No result bundle received from LLM engine.")
        return rows
    
    data = bundle["data"]
    stats = bundle["stats"]
    field_map = {h: f for f, h in headers_map.items()}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    logger.info(f"Result Processing: Integrating {len(data)} generated rows. (Expected {len(indices)})")

    # Keep track of which indices were actually filled
    filled_indices = set()

    for i, gen_row in enumerate(data):
        if i >= len(indices): break
        idx = indices[i]
        filled_indices.add(idx)
        
        rows[idx]['ai_audit'] = f"Gen AI | {algo} | {stats['total_tokens']} tok | {ts}"
        
        for header, val in gen_row.items():
            if header in field_map:
                f = field_map[header]
                # Light sanitization
                clean_val = val
                if isinstance(val, str):
                    s = val.replace('$', '').replace(',', '').strip()
                    try: 
                        clean_val = float(s) if '.' in s else int(s)
                    except: pass
                
                # Update only if currently empty
                if str(rows[idx].get(f)) in [None, "", "NaN", "None"]:
                    rows[idx][f] = clean_val
                    rows[idx][f"is_ai_{f}"] = True
        
        rows[idx]['is_target'] = False

    # Mark failed rows in the audit
    for i, idx in enumerate(indices):
        if idx not in filled_indices:
            rows[idx]['ai_audit'] = f"FAILED | {algo} | Turn timed out or socket reset | {ts}"
            # We keep is_target=True so they stay red and visible

    return rows

# --- THE MASTER AI GENERATION ENGINE ---
@app.callback(
    Output({'type': 'data-grid', 'index': MATCH}, 'rowData', allow_duplicate=True),
    Output({'type': 'target-row-ids', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'btn-generate', 'index': MATCH}, 'n_clicks'),
    State({'type': 'data-grid', 'index': MATCH}, 'rowData'),
    State({'type': 'target-row-ids', 'index': MATCH}, 'data'),
    State({'type': 'context-row-ids', 'index': MATCH}, 'data'),
    State({'type': 'check-full-context', 'index': MATCH}, 'value'),
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
def run_ai_generation_pm(n_clicks, row_data, target_ids, context_ids, full_ctx,
                         col_defs, algo, tone, consistency,
                         temp, cp_names, cp_types,
                         cp_texts, cp_mins, cp_maxs, desc, chunk_sz, max_c, strat):

    if not n_clicks or not target_ids:
        return dash.no_update, dash.no_update

    logger.info(f"--- STARTING TABULLM GENERATION PIPELINE ({len(target_ids)} targets) ---")

    # 1. Pipeline preparation
    inst = _build_instructions(tone, consistency, cp_names, cp_types, cp_texts, cp_mins, cp_maxs)
    logger.debug(f"Compiled Instructions:\n{inst}")
    
    ctx = _prepare_context(row_data, context_ids, full_ctx, strat, max_c, col_defs, desc)
    payload, target_indices, empty_count = _map_targets(row_data, target_ids, ctx['header_to_field'])

    # 2. Execution
    bundle = generate_bulk_data(
        payload, ctx, ctx['column_headers'], algo, inst, temp,
        num_empty=empty_count, chunk_size=chunk_sz
    )

    # 3. Post-processing
    updated_rows = _process_results(list(row_data), target_indices, bundle, algo, ctx['header_to_field'])
    
    # Identify remaining targets (those that are still marked as targets)
    remaining = [r['_row_id'] for r in updated_rows if r.get('is_target') and r['_row_id'] in target_ids]
    
    logger.info(f"--- GENERATION PIPELINE FINISHED. Successfully processed {len(target_ids) - len(remaining)}/{len(target_ids)} rows. ---")
    return updated_rows, remaining


@app.callback(
    Output({'type': 'estimated-tokens-lbl', 'index': MATCH}, 'children'),
    Input({'type': 'gen-max-context', 'index': MATCH}, 'value'),
    Input({'type': 'check-full-context', 'index': MATCH}, 'value'),
    Input({'type': 'gen-chunk-size', 'index': MATCH}, 'value'),
    Input({'type': 'temperature-slider', 'index': MATCH}, 'value'),
    Input({'type': 'algo-selector', 'index': MATCH}, 'value'),
    State({'type': 'data-grid', 'index': MATCH}, 'rowData'),
    State({'type': 'context-row-ids', 'index': MATCH}, 'data'),
    State({'type': 'dataset-description-store', 'index': MATCH}, 'data')
)
def estimate_context_tokens_pm(max_rows, full_ctx_toggle, chunk_size, temp, algo, row_data, context_ids, desc_metadata):
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
        json_tokens_est = (len(manual_ctx_str) + global_ctx_len) / 2.8
        
        # Prompt overhead: System instructions, template framing, column definitions.
        # This is roughly constant regardless of dataset size.
        # Based on generation_final.j2 and generation_init.j2 (~1300 chars)
        prompt_overhead = 550 
        
        # Chunk factor: smaller chunks lead to more overhead in multi-turn logic, 
        # though for a single estimation we focus on the context window.
        est_tokens = int(text_tokens + json_tokens_est + prompt_overhead)
        fraction = (n_rows / len(df)) * 100 if len(df) > 0 else 0
        suffix = " (Off)" if not full_ctx_toggle else ""
        
        return f"~{est_tokens} Tok ({fraction:.0f}%){suffix}"
    except Exception as e:
        logger.error(f"Token estimation error: {e}")
        return "~Err Tok"
