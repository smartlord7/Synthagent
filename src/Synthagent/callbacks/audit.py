import pandas as pd
import dash
from dash import Input, Output, State, MATCH, html, dcc
import dash_bootstrap_components as dbc
from core import app
from services.auditor import auditor
from services.llm_client import call_llm
from services.prompt_manager import prompt_manager
from utils.logging_config import logger

def _get_help_icon(target_id, text):
    """Returns a small help icon with a tooltip."""
    return html.Span([
        html.I(className="bi bi-question-circle ms-1 cursor-pointer xx-small opacity-50", id=target_id),
        dbc.Tooltip(text, target=target_id, placement="top", className="small shadow-lg")
    ])

@app.callback(
    Output({"type": "audit-struct-container", "index": MATCH}, "children"),
    Output({"type": "audit-fairness-container", "index": MATCH}, "children"),
    Output({"type": "audit-privacy-container", "index": MATCH}, "children"),
    Output({"type": "audit-quality-container", "index": MATCH}, "children"),
    Input({"type": "btn-run-audit", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "target-cols-dropdown", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def run_dataset_audit(n_clicks, row_data, targets):
    if not n_clicks or not row_data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    logger.info(f"--- STARTING EXPANSIVE MODULAR AUDIT (n={len(row_data)}) ---")
    df = pd.DataFrame(row_data)
    
    sensitive_keywords = ['gender', 'sex', 'age', 'race', 'ethnicity', 'nationality', 'religion', 'zip', 'income', 'stage', 'location', 'origin', 'education', 'marital', 'disability']
    sensitive_cols = [c for c in df.columns if any(k in c.lower() for k in sensitive_keywords) 
                      and not (c.startswith('is_ai_') or c in ['is_context', 'is_target'])]
    
    target_col = targets[0] if targets else None
    
    # 1. Calculate Comprehensive Metrics
    report = auditor.get_audit_report(df, target_col, sensitive_cols)
    struct = report['structural']
    quality = report['quality']
    fairness = report['fairness']
    privacy = report['privacy']

    # 2. Build Structural Layout (10+ Metrics)
    struct_layout = [
        html.Div([
            html.H6([html.I(className="bi bi-cpu me-2"), "Physical & Structural Core"], className="fw-bold x-small text-info text-uppercase mb-3 mt-2"),
            dbc.Row([
                dbc.Col([
                    html.Small(["Density", _get_help_icon("help-density", "Ratio of non-null cells. High density is better for model stability.")]),
                    dbc.Progress(value=struct['density_ratio']*100, color="info", className="mb-2", style={"height": "4px"})
                ], width=6),
                dbc.Col([
                    html.Small(["Memory Efficiency", _get_help_icon("help-mem", "Total footprint in memory.")]),
                    html.Div(struct['memory_readable'], className="text-light fw-bold xx-small")
                ], width=6),
            ], className="g-3 text-muted xx-small mb-3"),
            
            html.Div([
                html.Div([html.Span("Rows / Columns", className="xx-small"), html.Span(f"{struct['row_count']} / {struct['col_count']}", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
                html.Div([html.Span("Total Cells", className="xx-small"), html.Span(f"{struct['total_cells']:,}", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
                html.Div([html.Span("Empty Rows", className="xx-small"), html.Span(str(struct['empty_rows']), className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
                html.Div([html.Span("Avg Cardinality", className="xx-small"), html.Span(f"{struct['avg_cardinality']:.1f}", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            ], className="text-muted xx-small mb-3"),
            
            html.Hr(className="border-secondary opacity-25 my-2"),
            html.Small([html.I(className="bi bi-tags me-1"), "Dynamic Type Mapping"], className="text-muted xx-small fw-bold d-block mb-2"),
            html.Div([
                dbc.Badge(f"Numeric: {struct['type_distribution']['numeric']}", color="info", className="me-1 xx-small"),
                dbc.Badge(f"Categorical: {struct['type_distribution']['categorical']}", color="success", className="me-1 xx-small"),
                dbc.Badge(f"Objects: {struct['type_distribution']['object']}", color="secondary", className="me-1 xx-small"),
            ])
        ])
    ]

    # 3. Build Quality Layout (10+ Metrics)
    q_score = quality['quality_score']
    q_color = "success" if q_score > 90 else ("warning" if q_score > 70 else "danger")
    quality_layout = [
        html.H6([html.I(className="bi bi-check-circle me-2"), "Data Quality & Signal-to-Noise"], className="fw-bold x-small text-primary text-uppercase mb-3 mt-2"),
        html.Div([
            html.Div([html.Small("Signal Quality Score", className="xx-small text-muted"), html.Small(f"{q_score:.0f}/100", className=f"ms-auto text-{q_color} fw-bold")], className="d-flex mb-1"),
            dbc.Progress(value=q_score, color=q_color, className="mb-3", style={"height": "10px"})
        ]),
        html.Div([
            html.Div([html.Span("Duplicates", className="xx-small"), html.Span(f"{quality['duplicate_count']} ({quality['duplicate_ratio']*100:.1f}%)", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            html.Div([html.Span("Outlier Records", className="xx-small"), html.Span(f"{quality['outlier_count']} ({quality['outlier_ratio']*100:.1f}%)", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            html.Div([html.Span("Avg Skewness", className="xx-small"), html.Span(f"{quality['avg_skewness']:.2f}", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            html.Div([html.Span("Avg Kurtosis", className="xx-small"), html.Span(f"{quality['avg_kurtosis']:.2f}", className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            html.Div([html.Span("Conflicting Targets", className="xx-small"), html.Span(str(quality['conflicting_labels']), className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
        ], className="text-muted xx-small mb-3"),
        
        html.Small([html.I(className="bi bi-droplet-half me-1"), "Missingness Landscape"], className="text-muted xx-small fw-bold d-block mb-1"),
        html.Div([
            dbc.Badge(f"{quality['columns_with_missing']} cols with nulls", color="warning" if quality['columns_with_missing'] > 0 else "success", className="xx-small")
        ])
    ]

    # 4. Build Fairness Layout (Deep Metrics)
    fairness_layout = [html.H6([html.I(className="bi bi-balance me-2"), "Fairness & Ethical Representation"], className="fw-bold x-small text-warning text-uppercase mb-3 mt-2")]
    
    if "status" in fairness:
        fairness_layout.append(dbc.Alert(fairness["status"], color="secondary", className="xx-small p-2"))
    else:
        fairness_layout.append(html.Div([
            html.Div([html.Span("Detected Bias Zones", className="xx-small text-muted"), html.Span(str(fairness['detected_bias_count']), className=f"ms-auto badge bg-{'danger' if fairness['detected_bias_count'] > 0 else 'success'}")], className="d-flex mb-2 align-items-center"),
        ]))
        
        for col, data in fairness['metrics_per_attr'].items():
            di = data['disparate_impact']
            color = "success" if (0.8 <= di <= 1.25) else ("danger" if di < 0.8 else "warning")
            
            # Find top distribution shift
            shifts = data.get('distribution_shift', {})
            top_shift_col = max(shifts, key=shifts.get) if shifts else None
            shift_val = shifts[top_shift_col] if top_shift_col else 0
            
            fairness_layout.append(html.Div([
                html.Div([html.Small(col.title(), className="fw-bold"), html.Small(f"DI: {di:.2f}", className=f"ms-auto text-{color}")], className="d-flex"),
                dbc.Progress(value=min(100, (di/1.25)*100), color=color, className="my-1", style={"height": "6px"}),
                html.Div([
                    html.Small(f"Maj: {data['majority_group']}", className="xx-small text-muted"), 
                    html.Small(f"Rep: {data['representation_ratio']*100:.1f}%", className="ms-auto xx-small text-muted")
                ], className="d-flex mb-1"),
                
                # NEW: Grantular shift alerts
                html.Div([
                    html.I(className="bi bi-info-circle me-1 text-info"),
                    html.Span(f"Context Shift: {top_shift_col} (+{shift_val:.2f})", className="xx-small opacity-75")
                ], className="text-muted italic mb-2") if shift_val > 0.2 else None
            ], className="mb-3 p-1 border-start border-3 border-secondary bg-white bg-opacity-5"))

    # 5. Build Privacy Layout (Deep Metrics)
    p_risk = privacy['identity_risk_level']
    p_color = "success" if p_risk == 'Low' else ("warning" if p_risk == 'Medium' else "danger")
    
    privacy_layout = [
        html.H6([html.I(className="bi bi-shield-lock me-2"), "Privacy Erosion & Identity Risk"], className="fw-bold x-small text-danger text-uppercase mb-3 mt-2"),
        html.Div([
            html.Div([html.Small(f"Risk Profile: {p_risk}", className="xx-small text-muted"), html.Small(f"{privacy['uniqueness_ratio']*100:.1f}% unique", className=f"ms-auto text-{p_color}")], className="d-flex mb-1"),
            dbc.Progress(value=privacy['uniqueness_ratio']*100, color=p_color, className="mb-3", style={"height": "10px"}),
        ]),
        
        # New K-Anonymity & Entropy Row
        dbc.Row([
            dbc.Col([
                html.Small("Avg K-Anon", className="xx-small text-muted d-block"),
                html.Strong(f"{privacy['avg_k_anonymity']:.1f}", className="text-light small")
            ], width=6),
            dbc.Col([
                html.Small("Max Entropy", className="xx-small text-muted d-block"),
                html.Strong(f"{privacy['max_column_entropy']:.2f}", className="text-light small")
            ], width=6),
        ], className="mb-3 text-center"),

        html.Small([html.I(className="bi bi-list-stars me-1"), "Top Identifiability Leakers"], className="text-muted xx-small fw-bold d-block mb-1"),
        html.Div([
            html.Div([
                html.Span(col, className="xx-small"),
                dbc.Badge(m['identifiability'], color="danger" if m['identifiability'] == "High" else "secondary", className="ms-auto xx-small")
            ], className="d-flex mb-1 border-bottom border-secondary border-opacity-10 pb-1")
            for col, m in sorted(privacy['attr_identifiability'].items(), key=lambda x: x[1]['uniqueness_signal'], reverse=True)[:5]
        ], className="mb-3"),
        
        html.Small([html.I(className="bi bi-fingerprint me-1"), "Suspected QID Cluster"], className="text-muted xx-small fw-bold d-block mb-1"),
        html.Div([
            dbc.Badge(qid, color="danger", className="me-1 mb-1 xx-small") for qid in privacy['known_sensitive_patterns'][:5]
        ] if privacy['known_sensitive_patterns'] else [html.Small("None detected by pattern match.", className="text-muted xx-small italic")])
    ]

    logger.info("--- PHASE 3 MODULAR AUDIT COMPLETE ---")
    return struct_layout, fairness_layout, privacy_layout, quality_layout

@app.callback(
    Output({"type": "audit-strategy-markdown", "index": MATCH}, "children"),
    Output({"type": "generated-rules-store", "index": MATCH}, "data"),
    Output({"type": "btn-apply-audit-rules", "index": MATCH}, "style"),
    Output({"type": "tooltip-apply-audit-rules", "index": MATCH}, "children"),
    Input({"type": "btn-ponder-audit", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "target-cols-dropdown", "index": MATCH}, "value"),
    State({"type": "algo-selector", "index": MATCH}, "value"),
    State({"type": "temperature-slider", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def ponder_audit_strategy(n_clicks, row_data, targets, algo, temp):
    if not n_clicks or not row_data:
        return dash.no_update
    
    # 1. Run Metrics (Transiently) - Ensure technical cols are filtered
    df_raw = pd.DataFrame(row_data)
    df = auditor._filter_data(df_raw)
    
    target_col = targets[0] if targets else None
    sensitive_keywords = ['gender', 'sex', 'age', 'race', 'ethnicity', 'nationality', 'religion', 'zip', 'income', 'stage', 'location', 'origin', 'education', 'marital', 'disability']
    sensitive_cols = [c for c in df.columns if any(k in c.lower() for k in sensitive_keywords) and c not in ['is_context', 'is_target']]

    report = auditor.get_audit_report(df, target_col, sensitive_cols)

    # 2. Render Prompt
    context = {
        "report": report,
        "target_col": target_col,
        "sensitive_cols": sensitive_cols
    }
    prompt = prompt_manager.get_prompt("ponder_audit.j2", **context)
    
    # 3. Call LLM for Strategy
    try:
        logger.info(f"AI Pondering: Generating Ethics-Aware Strategy using {algo}")
        strategy = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model_id=algo,
            temperature=temp
        )
        
        # 4. Generate Rules from Strategy (with Data Summary for grounding)
        logger.info("AI Pondering: Transforming Strategy into Actionable Rules")
        
        # Build a small data summary to ground the LLM
        data_summary = {}
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    c_min = df[col].min()
                    c_max = df[col].max()
                    data_summary[col] = {
                        "type": "numeric", 
                        "min": float(c_min) if pd.notnull(c_min) else 0, 
                        "max": float(c_max) if pd.notnull(c_max) else 0
                    }
                else:
                    unique_vals = [str(v) for v in df[col].dropna().unique()[:10]]
                    data_summary[col] = {"type": "categorical", "sample_values": unique_vals}
            except Exception as de:
                logger.warning(f"Data summary failed for col {col}: {de}")
                data_summary[col] = {"type": "object", "sample_values": []}
        
        rules_prompt = prompt_manager.get_prompt(
            "audit_to_rules", 
            strategy=strategy, 
            attributes=df.columns.tolist(),
            data_summary=data_summary
        )
        logger.debug(f"Rules Prompt: {rules_prompt}")
        rules_json = call_llm(
            messages=[{"role": "user", "content": rules_prompt}],
            model_id=algo,
            temperature=0.1
        )
        
        import json
        try:
            # Clean up potential markdown formatting if LLM included it
            clean_json = rules_json.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
            rules = json.loads(clean_json)
            # Create a nice preview for the tooltip
            preview = [html.Div([
                html.B(f"{r.get('name')}: ", className="text-warning"), 
                html.Span(f"{r.get('value')}")
            ], className="mb-1") for r in rules[:5]]
            if len(rules) > 5:
                preview.append(html.Small(f"... and {len(rules)-5} more rules", className="italic opacity-75"))
            
            if not preview:
                preview = "Apply the identified strategic rules to generation"
                
            return strategy, rules, {"display": "block"}, preview
            
        except Exception as e:
            logger.error(f"Failed to parse rules JSON: {e}")
            return strategy, [], {"display": "none"}, "Error parsing rules"

    except Exception as e:
        logger.error(f"Audit Pondering Error: {e}")
        return f"**Generation Failed**: {str(e)}", [], {"display": "none"}, dash.no_update
