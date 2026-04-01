import pandas as pd
import dash
from dash import Input, Output, State, MATCH, html, dcc
import dash_bootstrap_components as dbc
from core import app
from services.auditor import auditor
from services.llm_client import provider
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
    
    logger.info(f"--- STARTING PREMIUM TABBED AUDIT (n={len(row_data)}) ---")
    df = pd.DataFrame(row_data)
    
    # 1. Calculate Comprehensive Metrics
    struct = auditor.get_structural_metrics(df)
    privacy = auditor.get_privacy_metrics(df)
    outliers = auditor.get_outlier_metrics(df)
    cardinality = auditor.get_cardinality_metrics(df)
    
    sensitive_keywords = ['gender', 'sex', 'age', 'race', 'ethnicity', 'nationality', 'religion', 'zip', 'income', 'stage', 'location', 'origin', 'education', 'marital', 'disability']
    sensitive_cols = [c for c in df.columns if any(k in c.lower() for k in sensitive_keywords) 
                      and not (c.startswith('is_ai_') or c in ['is_context', 'is_target'])]
    
    target_col = targets[0] if targets else None
    fairness = auditor.get_fairness_metrics(df, target_col, sensitive_cols)
    proxies = auditor.get_proxy_risks(df, sensitive_cols)
    quality = auditor.get_quality_metrics(df, target_col)

    # 2. Build Structural Layout
    struct_layout = [
        html.Div([
            html.H6([html.I(className="bi bi-cpu me-2"), "Integrity & Distribution"], className="fw-bold x-small text-info text-uppercase mb-3 mt-2"),
            dbc.Row([
                dbc.Col([
                    html.Small(["Sparsity", _get_help_icon("help-sparsity", "Percentage of missing values. High sparsity weakens local model patterns.")]),
                    dbc.Progress(value=struct['overall_sparsity']*100, color="info", className="mb-2", style={"height": "4px"})
                ], width=6),
                dbc.Col([
                    html.Small(["Outliers", _get_help_icon("help-outliers", "Statistically extreme rows (Z-score > 3). Excessive outliers can bias generation.")]),
                    dbc.Progress(value=outliers['outlier_ratio']*1000 if outliers['outlier_ratio'] < 0.1 else 100, color="danger" if outliers['total_outliers'] > 0 else "success", className="mb-2", style={"height": "4px"})
                ], width=6),
            ], className="g-3 text-muted xx-small mb-3"),
            
            html.Hr(className="border-secondary opacity-25 my-2"),
            html.Small([html.I(className="bi bi-graph-up-arrow me-1"), "Distribution & Cardinality Alerts"], className="text-muted xx-small fw-bold d-block mb-2"),
        ])
    ]
    
    struct_alerts = []
    # Add Skewness Alerts
    if 'skewness' in struct:
        for col, skew in struct['skewness'].items():
            if abs(skew) > 2:
                struct_alerts.append(dbc.Badge([f"Skews: {col}"], color="warning", className="me-1 mb-1 xx-small", id=f"skew-{col}"))
                struct_alerts.append(dbc.Tooltip(f"Highly Asymmetrical: {col} has skewness of {skew:.2f}. May need normalization.", target=f"skew-{col}"))
    if cardinality:
        for c_col, c_data in cardinality.items():
            struct_alerts.append(dbc.Badge([f"High Card: {c_col}"], color="danger", className="me-1 mb-1 xx-small", id=f"card-{c_col}"))
            struct_alerts.append(dbc.Tooltip(f"Volatility: {c_col} has {c_data['count']} uniques. Can lead to model hallucinations.", target=f"card-{c_col}"))
    if not struct_alerts:
        struct_alerts = [html.Small("Distributions within normal ranges.", className="text-muted xx-small italic")]
    struct_layout.append(html.Div(struct_alerts))

    # 3. Build Fairness Layout
    fairness_layout = [html.H6([html.I(className="bi bi-balance me-2"), "Bias & Parity Scorecard"], className="fw-bold x-small text-warning text-uppercase mb-3 mt-2")]
    
    if not target_col:
        fairness_layout.append(dbc.Alert([
            html.I(className="bi bi-exclamation-triangle-fill me-2"),
            "Action Required: Select a Target Variable in the 'Metadata' section to enable fairness parity analysis."
        ], color="warning", className="xx-small p-2 border-warning opacity-75"))
    elif not sensitive_cols:
        fairness_layout.append(dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            "Note: No sensitive features (gender, race, age, etc.) were auto-detected. Check your column names or rename them to trigger audit metrics."
        ], color="secondary", className="xx-small p-2 border-secondary"))
    elif fairness:
        for col, data in fairness.items():
            di = data['disparate_impact']
            impact_val = min(100, max(0, (di / 1.25) * 100))
            color = "success" if not data['bias_detected'] else ("danger" if di < 0.8 else "warning")
            fairness_layout.append(html.Div([
                html.Div([html.Span(f"{col.title()}", className="fw-bold x-small"), html.Span(f"DI: {di:.2f}", className=f"ms-auto text-{color}")], className="d-flex align-items-center mb-1"),
                dbc.Progress(value=impact_val, color=color, className="mb-1", style={"height": "10px", "borderRadius": "5px"}),
                html.Div([html.Small([f"Ref: {data['majority_group']}", _get_help_icon(f"help-di-{col}", "Measures if groups receive equal outcomes. (Goal: 0.8 - 1.25)")]), html.Small(f"Gap: {data['parity_gap']:.2f}", className="ms-auto")], className="d-flex justify-content-between text-muted xx-small mb-2")
            ], className="mb-3 p-2 border border-secondary rounded bg-black shadow-sm"))
            
        # Add Gini Stability here
        if 'stability' in struct:
            stab_items = []
            for col, stab in struct['stability'].items():
                if stab > 0.7:
                    stab_items.append(html.Div([
                        html.Small(f"Stuck Group: {col}", className="xx-small text-muted"),
                        dbc.Progress(value=stab*100, color="danger", className="mb-1", style={"height": "2px"})
                    ], className="mb-1"))
            if stab_items:
                fairness_layout.append(html.Div([
                    html.Hr(className="border-secondary opacity-25 my-3"),
                    html.Small([html.I(className="bi bi-virus me-2"), "Category Dominance (Gini Stability)"], className="text-muted xx-small fw-bold d-block mb-2"),
                    html.Div(stab_items)
                ]))
    else:
        fairness_layout.append(dbc.Alert("Fairness metrics could not be calculated. Select more diverse sensitive features.", color="secondary", className="xx-small p-2 border-secondary"))

    # 4. Build Privacy Layout
    qids = auditor.get_quasi_identifiers(df)
    privacy_color = "success" if privacy['risk_level'] == 'Low' else ("warning" if privacy['risk_level'] == 'Medium' else "danger")
    
    privacy_layout = [
        html.H6([html.I(className="bi bi-shield-lock me-2"), "Privacy & Identity Integrity"], className="fw-bold x-small text-danger text-uppercase mb-3 mt-2"),
        html.Div([
            html.Small([f"Re-identification Risk ({privacy['risk_level']})", _get_help_icon("help-privacy", "Identity uniqueness based on all feature combinations (including numeric QIDs).")]),
            dbc.Progress(value=privacy['uniqueness_ratio']*100, color=privacy_color, className="mb-1", style={"height": "10px"}),
            html.Div([
                html.Small(f"Ratio: {privacy['uniqueness_ratio']*100:.1f}%", className="text-muted xx-small"),
                html.Small(f"At-Risk: {privacy['at_risk_count']} rows", className="ms-auto text-muted xx-small")
            ], className="d-flex justify-content-between mb-3")
        ]),
        
        html.Hr(className="border-secondary opacity-25 my-2"),
        html.Small([html.I(className="bi bi-fingerprint me-1"), "Potential Quasi-Identifiers"], className="text-muted xx-small fw-bold d-block mb-2"),
        html.Div([
            dbc.Badge(qid, color="danger" if any(k in qid.lower() for k in ['zip', 'id', 'addr', 'loc', 'dob']) else "warning", 
                      className="me-1 mb-1 xx-small") for qid in qids
        ] if qids else [html.Small("No sensitive QIDs found.", className="text-muted xx-small italic")]),

        html.Hr(className="border-secondary opacity-25 my-2"),
        html.Small([html.I(className="bi bi-link-45deg me-1"), "Social Correlation Proxies (0.30+)"], className="text-muted xx-small fw-bold d-block mb-2"),
    ]
    
    proxy_badges = []
    if proxies:
        for s_col, p_map in proxies.items():
            for p_col in p_map.keys():
                proxy_badges.append(dbc.Badge([f"{p_col} → {s_col}"], color="warning", className="me-1 mb-1 xx-small", id=f"proxy-{p_col}"))
                proxy_badges.append(dbc.Tooltip(f"Hidden Bias: {p_col} has {p_map[p_col]:.2f} correlation with {s_col}.", target=f"proxy-{p_col}"))
    if not proxy_badges:
        proxy_badges = [html.Small("No significant proxy linkages detected.", className="text-muted xx-small italic")]
    privacy_layout.append(html.Div(proxy_badges))

    # 5. Build Quality Layout
    q_score = quality['quality_score']
    q_color = "success" if q_score > 90 else ("warning" if q_score > 70 else "danger")
    quality_layout = [
        html.H6([html.I(className="bi bi-check-circle me-2"), "Data Quality & Consistency"], className="fw-bold x-small text-primary text-uppercase mb-3 mt-2"),
        html.Div([
            html.Div([html.Small("Cleanliness Score", className="xx-small text-muted"), html.Small(f"{q_score:.0f}/100", className=f"ms-auto text-{q_color} fw-bold")], className="d-flex mb-1"),
            dbc.Progress(value=q_score, color=q_color, className="mb-3", style={"height": "10px"})
        ]),
        html.Div([
            html.Div([html.Span("Duplicate Records", className="xx-small"), html.Span(str(quality['total_duplicates']), className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            html.Div([html.Span("Conflicting Targets", className="xx-small"), html.Span(str(quality['conflicting_labels']), className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
            html.Div([html.Span("Constant Features", className="xx-small"), html.Span(str(len(quality['constant_features'])), className="ms-auto text-light")], className="d-flex mb-1 border-bottom border-secondary border-opacity-25 pb-1"),
        ], className="text-muted xx-small mb-3"),
    ]
    
    if quality['constant_features']:
        quality_layout.append(html.Div([
            html.Small([html.I(className="bi bi-exclamation-triangle me-1"), "Zero Variance Alert"], className="text-warning xx-small d-block mb-1"),
            html.Div([dbc.Badge(c, color="secondary", className="me-1 mb-1 xx-small") for c in quality['constant_features']])
        ]))

    logger.info("--- PREMIUM TABBED AUDIT COMPLETE ---")
    return struct_layout, fairness_layout, privacy_layout, quality_layout

@app.callback(
    Output({"type": "audit-strategy-markdown", "index": MATCH}, "children"),
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
    
    struct = auditor.get_structural_metrics(df)
    privacy = auditor.get_privacy_metrics(df)
    sensitive_keywords = ['gender', 'sex', 'age', 'race', 'ethnicity', 'nationality', 'religion', 'zip', 'income', 'stage', 'location', 'origin', 'education', 'marital', 'disability']
    sensitive_cols = [c for c in df.columns if any(k in c.lower() for k in sensitive_keywords) and c not in ['is_context', 'is_target']]
    
    target_col = targets[0] if targets else "None Selected"
    fairness = auditor.get_fairness_metrics(df, targets[0] if targets else None, sensitive_cols)
    proxies = auditor.get_proxy_risks(df, sensitive_cols)
    quality = auditor.get_quality_metrics(df, targets[0] if targets else None)
    outliers = auditor.get_outlier_metrics(df)

    # 2. Render Prompt
    context = {
        "overall_sparsity": struct['overall_sparsity'] * 100,
        "top_sparse_cols": [c for c, s in struct['sparsity'].items() if s > 0.05][:3],
        "skewness": {c: v for c, v in struct.get('skewness', {}).items() if abs(v) > 2},
        "target_col": target_col,
        "fairness": fairness,
        "privacy": privacy,
        "proxies": proxies,
        "quality": quality,
        "outliers": outliers
    }
    prompt = prompt_manager.get_prompt("ponder_audit.j2", **context)
    
    # 3. Call LLM
    try:
        logger.info(f"AI Pondering: Generating Ethics-Aware Strategy using {algo}")
        response = provider.call_llm(
            messages=[{"role": "user", "content": prompt}],
            model_id=algo,
            temperature=temp
        )
        return response
    except Exception as e:
        logger.error(f"Audit Pondering Error: {e}")
        return f"**Generation Failed**: {str(e)}"
