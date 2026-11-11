import json
from datetime import datetime
import pandas as pd
import dash
from dash import Input, Output, State, ALL, MATCH
from core import app
from utils.logging_config import logger
from services.llm_client import generate_bulk_data, get_available_models
from dash import callback_context
from utils.json_utils import to_compact_json
from services.data_manager import select_context_rows
from services.rule_inference import RuleInference

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
        models_info = get_available_models()
        options = [{"label": m["id"].split("/")[-1].replace("-", " ").title(), "value": m["id"]} for m in models_info]
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

def _build_instructions(tone, consistency, names, types, texts, mins, maxs, all_headers):
    """Compile rules and parameters into a string and a structured metadata list."""
    rules_text = []
    rules_metadata = []
    
    # Identify Generic Columns for Active Target Tagging
    generic_targets = []
    for h in all_headers:
        low_h = str(h).lower().strip()
        if low_h == "new column" or low_h.startswith("column_") or low_h.startswith("col_"):
            generic_targets.append(h)

    header = f"Style: {tone}\nConsistency: {consistency}\n"
    if generic_targets:
        targets_str = ", ".join([f"'{t}'" for t in generic_targets])
        header += f"MANDATORY SCHEMA EVOLUTION: The columns [{targets_str}] are temporary placeholders and MUST be given specific, semantically relevant names.\n"
    
    for i in range(len(names)):
        if names[i]:
            detail = texts[i] if types[i] != 'number' else f"Min:{mins[i]}, Max:{maxs[i]}"
            rules_text.append(f"- Rule [{names[i]}]: {detail}")
            
            rules_metadata.append({
                "name": names[i],
                "type": types[i],
                "text": texts[i],
                "min": mins[i],
                "max": maxs[i]
            })
            
    full_inst = header + "\n".join(rules_text)
    return full_inst, rules_metadata

def _prepare_context(row_data, context_ids, full_ctx_toggle, strategy, max_rows, col_defs, metadata):
    """Extract and sample reference context data."""
    df = pd.DataFrame(row_data)
    fields = {c['field']: c.get('headerName', c['field']) for c in col_defs
             if c['field'] not in ['ai_audit', '_row_id'] and not c['field'].startswith('is_ai_')}
    
    # Manual Context extraction
    manual_rows = [r for r in row_data if r.get('_row_id') in (context_ids or [])]
    manual_records = []
    
    for row in manual_rows:
        record = {}
        for f, h in fields.items():
            # Get value, default to None if column is brand new and missing from dict
            record[h] = row.get(f)
        manual_records.append(record)
    
    logger.info(f"Context Prep: Found {len(manual_records)} manual (Blue) reference rows.")
    
    # Global Context
    global_sample = []
    if full_ctx_toggle:
        logger.info(f"Context Prep: Sampling {max_rows} rows using strategy '{strategy}'.")
        # Pre-process row_data to ensure all fields exist (Inflation)
        inflated_data = []
        for r in row_data:
            nr = {f: r.get(f) for f in fields.keys()}
            inflated_data.append(nr)
            
        df_clean = pd.DataFrame(inflated_data).rename(columns=fields)
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
    """Identify which rows need completion vs generation and track indices with provider-parity order."""
    payload = []
    partial_indices = []
    empty_indices = []
    header_to_field = {h: f for f, h in headers.items()}
    
    for t_id in target_ids:
        idx = next((i for i, r in enumerate(row_data) if r.get('_row_id') == t_id), -1)
        if idx == -1: continue
        
        row = row_data[idx]
        # Check if row has any existing data as guidance
        guidance = {h: row.get(f) for h, f in header_to_field.items() 
                    if row.get(f) not in [None, "", "NaN"]}
        
        if guidance:
            payload.append(guidance)
            partial_indices.append(idx)
        else:
            empty_indices.append(idx)
    
    # Combined indices must match the LLM provider's return order: Partials then New
    combined_indices = partial_indices + empty_indices
    empty_count = len(empty_indices)
    
    logger.info(f"Target Mapping: {len(payload)} partial rows, {empty_count} entirely new rows.")
    return payload, combined_indices, empty_count

def _process_results(rows, target_indices, result_bundle, algo, all_fields, rules_hierarchy=None):
    """Integrate LLM rows into the local grid rowData using strict positional mapping and rule-aware audit."""
    if not result_bundle:
        logger.error("Result Processing: No result bundle received.")
        return rows
        
    rows_from_llm = result_bundle.get('data', [])
    stats = result_bundle.get('stats', {'total_tokens': 0})
    ts = datetime.now().strftime("%H:%M:%S")
    
    # Pre-build Rule Validator Model
    pydantic_model = None
    if rules_hierarchy:
        active_rules = list(rules_hierarchy.get("integrity", []))
        if rules_hierarchy.get("enforce_boundaries"):
            active_rules.extend(rules_hierarchy.get("boundaries", []))
        if rules_hierarchy.get("enforce_user"):
            active_rules.extend(rules_hierarchy.get("user", []))
        
        from services.rule_validator import RuleValidator
        pydantic_model = RuleValidator.create_pydantic_model(active_rules)

    # Integration logic: LLM Row 1 -> Target Index 1
    for i, idx in enumerate(target_indices):
        if i < len(rows_from_llm):
            new_row_data = rows_from_llm[i]
            has_missing_or_null = False
            has_rule_violations = False
            
            # 1. Basic Type Guard
            if not isinstance(new_row_data, dict):
                logger.error(f"Result Processing: Row {i} is NOT a dictionary. Marking for retry.")
                rows[idx]['ai_audit'] = f"Error | Malformed format (Retrying...) | {ts}"
                rows[idx]['is_target'] = True
                continue

            # 2. Extract and Sanitize
            clean_row = {}
            for f in all_fields:
                val = new_row_data.get(f)
                
                # Cleaning logic
                clean_val = val
                if val is not None:
                    s_val = str(val).strip()
                    if s_val.replace('.', '').replace('-', '').isdigit():
                        try: clean_val = float(s_val) if '.' in s_val else int(s_val)
                        except: pass
                
                # Null Check
                is_null = False
                if clean_val is None:
                    is_null = True
                elif isinstance(clean_val, str) and not clean_val.strip():
                    is_null = True
                elif str(clean_val).lower() in ["null", "none", "nan"]:
                    is_null = True
                
                if is_null:
                    has_missing_or_null = True
                
                clean_row[f] = clean_val

            # 3. Rule Validation (if model exists)
            if pydantic_model and not has_missing_or_null:
                try:
                    pydantic_model(**clean_row)
                except Exception as e:
                    logger.warning(f"Result Processing: Row at index {idx} failed rule validation: {e}")
                    has_rule_violations = True

            # 4. State Update / Auditing
            if has_missing_or_null or has_rule_violations:
                reason = "Partial/Null Data" if has_missing_or_null else "Rule Violation"
                rows[idx]['is_target'] = True
                rows[idx]['ai_audit'] = f"Retry | {reason} ({algo}) | {ts}"
            else:
                # Success - Inject data
                for f, v in clean_row.items():
                    rows[idx][f] = v
                    rows[idx][f"is_ai_{f}"] = True
                
                rows[idx]['is_target'] = False
                rows[idx]['is_context'] = True 
                rows[idx]['ai_audit'] = f"Gen AI | {algo} | {stats.get('total_tokens', 0)} tok | {ts}"
        else:
            # Shortfall handling
            rows[idx]['ai_audit'] = f"Retry | Shortfall (Retrying...) | {ts}"
            rows[idx]['is_target'] = True

    return rows

# --- THE MASTER AI GENERATION ENGINE ---
@app.callback(
    Output({'type': 'data-grid', 'index': MATCH}, 'rowData', allow_duplicate=True),
    Output({'type': 'data-grid', 'index': MATCH}, 'columnDefs', allow_duplicate=True),
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
    State({'type': 'check-enforce-rules', 'index': MATCH}, 'value'),
    State({'type': 'check-enforce-boundaries', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def run_ai_generation_pm(n_clicks, row_data, target_ids, context_ids, full_ctx,
                         col_defs, algo, tone, consistency,
                         temp, cp_names, cp_types,
                         cp_texts, cp_mins, cp_maxs, desc, chunk_sz, max_c, strat, enforce_rules, enforce_boundaries):

    if not n_clicks or not target_ids:
        return dash.no_update, dash.no_update, dash.no_update

    logger.info(f"--- STARTING TABULLM GENERATION PIPELINE ({len(target_ids)} targets) ---")

    # 1. Rules aggregation
    all_headers = [c.get('headerName', c['field']) for c in col_defs]
    inst, user_rules = _build_instructions(tone, consistency, cp_names, cp_types, cp_texts, cp_mins, cp_maxs, all_headers)
    
    # Infer rules from current dataset — ensure columns are preserved for empty sets
    all_fields = [c['field'] for c in col_defs if c['field'] not in ['ai_audit', '_row_id']]
    df_for_rules = pd.DataFrame(row_data)
    if df_for_rules.empty:
        df_for_rules = pd.DataFrame(columns=all_fields)
    
    inferred_rules = RuleInference.infer_dataset_rules(df_for_rules)
    
    # Hierarchical rules: Integrity (Level 1), Boundaries (Level 2), User (Level 3)
    # We pass all of them, but also pass the toggles to decide which are 'Hard'
    rules_hierarchy = {
        "integrity": [r for r in inferred_rules if r['level'] == 1],
        "boundaries": [r for r in inferred_rules if r['level'] == 2],
        "user": user_rules,
        "enforce_user": bool(enforce_rules and len(enforce_rules) > 0),
        "enforce_boundaries": bool(enforce_boundaries and len(enforce_boundaries) > 0)
    }

    # 2. Pipeline preparation
    logger.debug(f"Compiled Instructions:\n{inst}")
    
    # 2. Execution Loop (Autonomous Regeneration)
    max_rounds = 3
    current_round = 1
    current_target_ids = list(target_ids)
    
    while current_target_ids and current_round <= max_rounds:
        logger.info(f"--- GENERATION ROUND {current_round}/{max_rounds} ({len(current_target_ids)} targets) ---")
        
        ctx = _prepare_context(row_data, context_ids, full_ctx, strat, max_c, col_defs, desc)
        payload, target_indices, empty_count = _map_targets(row_data, current_target_ids, ctx['header_to_field'])

        result_bundle = generate_bulk_data(
            payload, ctx, ctx['column_headers'], all_fields, algo, inst, temp,
            num_empty=empty_count, chunk_size=chunk_sz, 
            rules_metadata=rules_hierarchy, enforce_rules=True
        )

        # 3. Post-processing (Merges into row_data)
        row_data = _process_results(list(row_data), target_indices, result_bundle, algo, all_fields, rules_hierarchy=rules_hierarchy)
        
        # 3b. Identify remaining targets (those that failed the completeness audit)
        current_target_ids = [r['_row_id'] for r in row_data if r.get('is_target') and r['_row_id'] in target_ids]
        
        if current_target_ids:
            logger.warning(f"Round {current_round} finished with {len(current_target_ids)} failures. Retrying in next round...")
            current_round += 1
        else:
            logger.info("All targets successfully generated and audited.")
            break

    # 4. Final Updates (Schema & Remaining)
    updated_cols = list(col_defs)
    schema_updates = result_bundle.get("schema_updates", {}) # Grab last bundle's schema info
    if schema_updates:
        logger.info(f"Schema Evolution: Applying {len(schema_updates)} automated renames.")
        def _canonical(s):
            return str(s).lower().replace("_", "").replace(" ", "").strip()
            
        for old_name, new_name in schema_updates.items():
            canon_old = _canonical(old_name)
            for col in updated_cols:
                h_name = col.get("headerName")
                f_name = col.get("field")
                if _canonical(h_name) == canon_old or _canonical(f_name) == canon_old:
                    col["headerName"] = new_name
                    from utils.constants import ai_highlight_style
                    col["cellStyle"] = ai_highlight_style
                    break

    # Final list of IDs that are still Targets (RED)
    remaining_ids = [r['_row_id'] for r in row_data if r.get('is_target') and r['_row_id'] in target_ids]
    
    logger.info(f"--- GENERATION PIPELINE FINISHED. Success: {len(target_ids) - len(remaining_ids)}/{len(target_ids)} ---")
    return row_data, updated_cols, remaining_ids


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
        
        # 1. Manual Context (Blue Rows) - Exact measurement
        manual_ctx_df = df[df['_row_id'].isin(context_ids or [])].drop(
            columns=['_row_id', 'is_context', 'is_target'], errors='ignore')
        all_cols = manual_ctx_df.columns.tolist()
        manual_ctx_rows = manual_ctx_df.to_dict('records')
        manual_ctx_compact = to_compact_json(manual_ctx_rows, all_cols)
        manual_ctx_str = json.dumps(manual_ctx_compact)
        
        # 2. Global Context (Pool) - Measured fraction of total pool
        global_ctx_len = 0
        n_rows = min(max_rows if max_rows else 0, len(df))
        
        if n_rows > 0:
            # Measure precisely the characters of the entire available pool
            clean_df = df.copy()
            for k in ['_row_id', 'is_context', 'is_target', 'ai_audit']:
                if k in clean_df.columns: clean_df.drop(columns=[k], inplace=True)
            
            pool_records = clean_df.to_dict('records')
            compact_pool = to_compact_json(pool_records, clean_df.columns.tolist())
            
            # The context window sent to the LLM is 'n_rows' large
            # We estimate based on the full measured weight of the table pool
            pool_data_str = json.dumps(compact_pool['data'])
            avg_row_len = len(pool_data_str) / len(clean_df) if len(clean_df) > 0 else 0
            global_ctx_len = len(json.dumps(compact_pool['cols'])) + (avg_row_len * n_rows)

        # 3. Final Calibration
        # Precision Ratio: 2.2 is accurate for high-symbol-density Compact JSON
        text_tokens = (len(desc_metadata) / 3.5) if desc_metadata else 0
        json_tokens_est = (len(manual_ctx_str) + global_ctx_len) / 2.2
        
        # Fixed prompt overhead (instructions/templates)
        prompt_overhead = 600
        
        est_tokens = int(text_tokens + json_tokens_est + prompt_overhead)
        fraction = (n_rows / len(df)) * 100 if len(df) > 0 else 0
        suffix = " (Off)" if not full_ctx_toggle else ""
        
        return f"~{est_tokens} Tok ({fraction:.0f}%){suffix}"
    except Exception as e:
        logger.error(f"Token estimation error: {e}")
        return "~Err Tok"
