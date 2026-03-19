import json
from datetime import datetime
import pandas as pd
import dash
from dash import Input, Output, State, ALL
from core import app, logger
from services.gemini_client import generate_bulk_data

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
    prevent_initial_call=True
)
def run_ai_generation(n_clicks, row_data, target_ids, context_ids, full_ctx_toggle,
                      col_defs, algorithm, param_tone, param_consistency,
                      param_uniqueness, temperature, cp_names, cp_types,
                      cp_texts, cp_mins, cp_maxs):
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
        # Sample up to 15 non-empty rows
        global_ctx_sample = df_full.dropna(how='all').sample(min(15, len(df_full))).to_dict('records')
        # Clean technical flags from sample
        for r in global_ctx_sample:
            for k in ['_row_id', 'is_context', 'is_target']: r.pop(k, None)

    # LOG: Global Sample
    logger.info(
        f"--- PROMPT COMPONENT: GLOBAL PATTERNS (Toggle) ---\nSample Size: {len(global_ctx_sample)}\nData: {json.dumps(global_ctx_sample)}")

    full_context_str = f"""
    Manual Reference Examples: {manual_ctx_json}
    General Dataset Patterns: {json.dumps(global_ctx_sample)}
    """

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
        full_context_str,
        list(field_to_header.values()),
        algorithm,
        compiled_instructions,
        temperature,
        num_empty=empty_row_count
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
