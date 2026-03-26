import json
from datetime import datetime
import pandas as pd
import dash
from dash import Input, Output, State, ALL
from core import app, logger
from services.llm_client import generate_bulk_data

# --- THE MASTER AI GENERATION ENGINE ---
@app.callback(
    Output('data-grid', 'rowData', allow_duplicate=True),
    Output('target-row-ids', 'data', allow_duplicate=True),
    Input('btn-generate', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('target-row-ids', 'data'),  # Target Row IDs (Red)
    State('context-row-ids', 'data'),  # Context Row IDs (Blue)
    State('check-full-context', 'value'),  # Full Dataset Context Toggle
    State('data-grid', 'columnDefs'),
    State('algo-selector', 'value'),
    State('param-tone', 'value'),
    State('param-consistency', 'value'),
    State('param-uniqueness', 'value'),
    State('temperature-slider', 'value'),
    State({'type': 'cp-name', 'index': ALL}, 'value'),
    State({'type': 'cp-type', 'index': ALL}, 'value'),
    State({'type': 'cp-val-text', 'index': ALL}, 'value'),
    State({'type': 'cp-val-min', 'index': ALL}, 'value'),
    State({'type': 'cp-val-max', 'index': ALL}, 'value'),
    State('dataset-description-store', 'data'),
    State('gen-chunk-size', 'value'),
    State('gen-max-context', 'value'),
    State('gen-context-strategy', 'value'),
    prevent_initial_call=True
)
def run_ai_generation(n_clicks, row_data, target_ids, context_ids, full_ctx_toggle,
                      col_defs, algorithm, param_tone, param_consistency,
                      param_uniqueness, temperature, cp_names, cp_types,
                      cp_texts, cp_mins, cp_maxs, desc_metadata, chunk_size, max_context, strategy):



    if not n_clicks or not target_ids:
        logger.warning("Generation aborted: No target rows (Red) defined.")
        return dash.no_update, dash.no_update

    logger.info(f"Starting AI Generation for {len(target_ids)} target rows.")

    # 1. Build Instructions
    compiled_instructions = (
        f"Style: {param_tone}\n"
        f"Consistency: {param_consistency}\n"
        f"Uniqueness: {param_uniqueness}\n"
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
    manual_ctx_json = manual_ctx_df.to_json(orient='records')

    # LOG: Manual Context
    logger.info(
        f"--- PROMPT COMPONENT: MANUAL CONTEXT (Blue Rows) ---\nRows identified: {len(manual_ctx_df)}\nData: {manual_ctx_json}")

    # B. Global Patterns (Full Dataset Toggle)
    global_ctx_sample = []
    if full_ctx_toggle:
        from services.data_manager import select_context_rows
        max_c = max_context if max_context else 100
        
        logger.info(f"Retrieving context sample using Strategy: {strategy} for {max_c} rows.")
        global_ctx_sample = select_context_rows(df_full, strategy=strategy, n_rows=max_c)

        # Clean technical flags from sample
        for r in global_ctx_sample:
            for k in ['_row_id', 'is_context', 'is_target']: r.pop(k, None)


    # LOG: Global Sample
    logger.info(
        f"--- PROMPT COMPONENT: GLOBAL PATTERNS (Toggle) ---\nSample Size: {len(global_ctx_sample)}\nData: {json.dumps(global_ctx_sample)}")

    # Pack context as Dictionary instead of string blob for downstream chunking engines
    context_data = {
        "metadata": desc_metadata if desc_metadata else "Not provided",
        "manual_ctx": manual_ctx_json,
        "global_ctx": global_ctx_sample
    }


    # 3. Target Mapping
    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs
                       if col['field'] not in ['ai_audit', '_row_id'] and not col['field'].startswith('is_ai_')}
    header_to_field = {v: k for k, v in field_to_header.items()}

    updated_rows = list(row_data)
    partial_payload = []  # Rows that have some guidance
    empty_row_count = 0  # Just a counter for the rest
    target_indices = []

    # This loop ONLY collects the data
    for t_id in target_ids:
        idx = next((i for i, r in enumerate(updated_rows) if r.get('_row_id') == t_id), -1)
        if idx != -1:
            row_obj = updated_rows[idx]
            # Create a dict of existing data (filtering out technical/audit fields)
            guidance = {field_to_header[f]: v for f, v in row_obj.items()
                        if f in field_to_header and v not in [None, "", "NaN"]}

            target_indices.append(idx)

            if len(guidance) > 0:
                partial_payload.append(guidance)
            else:
                empty_row_count += 1

    # 4. Call AI with the new optimization
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


    logger.info(
        f"--- PROMPT COMPONENT: TARGET PAYLOAD (Red Rows) ---\nRows to fill: {len(partial_payload)}\nPayload: {json.dumps(partial_payload)} + {empty_row_count} empty rows")

    remaining_targets = list(target_ids)

    # 5. Process Results
    if result_bundle:
        filled_data = result_bundle["data"]
        stats = result_bundle["stats"]

        logger.info(f"Input Tokens: {stats['prompt_tokens']} | Output Tokens: {stats['candidates_tokens']}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        for i, gen_row in enumerate(filled_data):
            if i >= len(target_indices): break
            master_idx = target_indices[i]
            row_uuid = updated_rows[master_idx]['_row_id']

            # Set Audit Info
            updated_rows[master_idx]['ai_audit'] = f"Gemini 1.5 Flash | {stats['total_tokens']} tokens | {current_time}"

            # Process individual columns
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

            # Clear Target Visuals
            updated_rows[master_idx]['is_target'] = False
            if row_uuid in remaining_targets:
                remaining_targets.remove(row_uuid)
    else:
        logger.error("AI Generation failed to return data.")

    return updated_rows, remaining_targets


@app.callback(
    Output('estimated-tokens-lbl', 'children'),
    Input('gen-max-context', 'value'),
    Input('check-full-context', 'value'),
    State('data-grid', 'rowData'),
    State('context-row-ids', 'data'),
    State('dataset-description-store', 'data')
)
def estimate_context_tokens(max_rows, full_ctx_toggle, row_data, context_ids, desc_metadata):
    if not row_data:
        return "~0 Tok"
    try:
        import json
        import pandas as pd
        
        df = pd.DataFrame(row_data)
        
        # 1. Manual Context String Size (Compact Format)
        from utils.json_utils import to_compact_json
        
        manual_ctx_df = df[df['_row_id'].isin(context_ids or [])].drop(
            columns=['_row_id', 'is_context', 'is_target'], errors='ignore')
        all_cols = manual_ctx_df.columns.tolist()
        manual_ctx_rows = manual_ctx_df.to_dict('records')
        manual_ctx_compact = to_compact_json(manual_ctx_rows, all_cols)
        manual_ctx_str = json.dumps(manual_ctx_compact)
        
        # 2. Global Context Fast Approximation (avoid slider-move clustering lag)
        global_ctx_len = 0
        clean_df = df.dropna(how='all') if not df.empty else df
        n_rows = min(max_rows if max_rows else 0, len(df))
        
        if not clean_df.empty:
            # Sample to estimate avg row size in list-of-lists format
            sample_count = min(5, len(clean_df))
            sample_rows_df = clean_df.sample(sample_count)
            for k in ['_row_id', 'is_context', 'is_target']: 
                if k in sample_rows_df.columns: sample_rows_df = sample_rows_df.drop(columns=[k], errors='ignore')
            
            all_cols_g = sample_rows_df.columns.tolist()
            sample_records = sample_rows_df.to_dict('records')
            compact_sample = to_compact_json(sample_records, all_cols_g)
            
            # header overhead (columns names) + N * average row length
            header_len = len(json.dumps(compact_sample['cols']))
            rows_only_len = len(json.dumps(compact_sample['data']))
            avg_row_char_len = rows_only_len / sample_count
            
            global_ctx_len = header_len + (avg_row_char_len * n_rows)

        meta_len = len(desc_metadata) if desc_metadata else 0
        
        # 3. Accumulated length
        estimated_char_len = len(manual_ctx_str) + global_ctx_len + meta_len + 300 # 300 instructions fluff
        
        # Standard assumption: ~4 characters per token
        est_tokens = int(estimated_char_len / 4)
        
        # Calculate dataset fraction
        fraction = (n_rows / len(df)) * 100 if len(df) > 0 else 0
        
        suffix = " (Off)" if not full_ctx_toggle else ""
        if est_tokens > 20000:
             return f"High: ~{est_tokens} Tok ({fraction:.0f}%){suffix}"
        return f"~{est_tokens} Tok ({fraction:.0f}%){suffix}"

    except Exception:
        return "~Err Tok"


