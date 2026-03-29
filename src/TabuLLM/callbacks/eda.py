import pandas as pd
import numpy as np
import dash
from dash import Input, Output, State, dcc, html, MATCH
import dash_bootstrap_components as dbc
from scipy import stats
from simsimd import jensenshannon
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

from core import app
from utils.logging_config import logger

# --- EDA DASHBOARD LOGIC ---
@app.callback(
    Output({"type": "eda-dashboard-container", "index": MATCH}, "children", allow_duplicate=True),
    Input({"type": "btn-refresh-eda", "index": MATCH}, "n_clicks"),
    State({"type": "original-data-store", "index": MATCH}, "data"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    State({"type": "btn-refresh-eda", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def build_eda_dashboard_pm(n_clicks, orig_data, aug_data, col_defs, btn_id):
    if not n_clicks:
        return dash.no_update
    dataset_id = btn_id["index"]

    field_to_header = {
        col['field']: col.get('headerName', col['field'])
        for col in col_defs
        if col['field'] not in ['ai_audit', '_row_id']
           and not col['field'].startswith('is_ai_')
    }

    df_orig = pd.DataFrame(orig_data)
    df_aug = pd.DataFrame(aug_data)

    for df in [df_orig, df_aug]:
        if '_row_id' in df.columns:
            df.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
        cols_to_keep = [c for c in df.columns if c in field_to_header]
        df.drop(columns=[c for c in df.columns if c not in cols_to_keep], inplace=True, errors='ignore')
        df.rename(columns=field_to_header, inplace=True)
        df.replace(["", "NaN", "None", None], pd.NA, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col])

    return get_eda_layout(df_orig, df_aug, dataset_id)

def get_eda_layout(df_orig, df_aug, dataset_id="base"):
    if df_orig.empty or df_aug.empty:
        return [dbc.Alert("Missing data to build dashboard layouts.", color="warning")]

    df_orig_c = df_orig.copy()
    df_aug_c = df_aug.copy()

    for df in [df_orig_c, df_aug_c]:
        for col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col])

    df_combined = pd.concat(
        [df_orig_c.assign(State='Original'), df_aug_c.assign(State='Augmented')],
        ignore_index=True
    )

    numeric_cols = df_aug_c.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df_aug_c.columns if c not in numeric_cols]

    jsd_scores, m_drift_list, mre_list, hellinger_cats, z_drift_list = [], [], [], [], []
    corr_stability, privacy_risk, coverage_score, nndr = 0, 0, 0, 0
    entropy_preservation = 0 # Placeholder for simplicity

    if numeric_cols:
        for col in numeric_cols:
            a, b = df_orig_c[col].dropna(), df_aug_c[col].dropna()
            if len(a) > 5 and len(b) > 5:
                hist_a, bin_edges = np.histogram(a, bins=20, density=True)
                hist_b, _ = np.histogram(b, bins=bin_edges, density=True)
                jsd = jensenshannon(hist_a, hist_b)
                jsd_scores.append(1 - jsd if not np.isnan(jsd) else 0)
                z_drift = abs(a.mean() - b.mean()) / (a.std() if a.std() > 0 else 1)
                z_drift_list.append(z_drift)
                if a.mean() != 0:
                    mre_list.append(abs(a.mean() - b.mean()) / abs(a.mean()))

        if len(numeric_cols) > 1:
            c_orig = df_orig_c[numeric_cols].corr().fillna(0).values
            c_aug = df_aug_c[numeric_cols].corr().fillna(0).values
            corr_stability = (1 - np.mean(np.abs(c_orig - c_aug))) * 100

        scaler = MinMaxScaler()
        combined_numeric = pd.concat([df_orig_c[numeric_cols], df_aug_c[numeric_cols]]).fillna(0)
        scaler.fit(combined_numeric)
        orig_norm = scaler.transform(df_orig_c[numeric_cols].fillna(0))
        aug_norm = scaler.transform(df_aug_c[numeric_cols].fillna(0))
        sample_sz = min(200, len(aug_norm))
        aug_sample = aug_norm[:sample_sz]
        dist_matrix = euclidean_distances(aug_sample, orig_norm)
        min_dists = np.min(dist_matrix, axis=1)
        privacy_risk = int((min_dists < 0.05).mean() * 100)
        coverage_score = int((min_dists > 0.1).mean() * 100)
        if len(orig_norm) > 1 and len(aug_sample) > 1:
            aug_to_aug = euclidean_distances(aug_sample, aug_sample)
            np.fill_diagonal(aug_to_aug, np.inf)
            nn_aug = np.min(aug_to_aug, axis=1)
            nndr = np.mean(min_dists / (nn_aug + 1e-6))

    if cat_cols:
        for col in cat_cols:
            p = df_orig_c[col].value_counts(normalize=True)
            q = df_aug_c[col].value_counts(normalize=True)
            all_cats = p.index.union(q.index)
            p = p.reindex(all_cats, fill_value=0).values
            q = q.reindex(all_cats, fill_value=0).values
            h_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
            hellinger_cats.append(1 - h_dist)

    fidelity_score = int(np.mean(jsd_scores) * 100) if jsd_scores else 0
    dist_consistency = int(np.median(jsd_scores) * 100) if jsd_scores else 0
    avg_z_drift = np.mean(z_drift_list) if z_drift_list else 0
    mean_feature_accuracy = int((1 - np.mean(mre_list)) * 100) if mre_list else 0
    cat_fidelity = int(np.mean(hellinger_cats) * 100) if hellinger_cats else 0
    data_density = 100 - (df_aug_c.isna().sum().sum() / (df_aug_c.size + 1e-6) * 100)
    
    # Simple Entropy Preservation (based on coverage + fidelity)
    ent_pres = "High" if coverage_score > 60 and fidelity_score > 70 else "Moderate" if coverage_score > 30 else "Low"
    ent_color = "success" if ent_pres == "High" else "warning" if ent_pres == "Moderate" else "danger"

    # Helper for unique IDs
    def uid(base): return f"{base}-{dataset_id}"

    # --- RICH METRIC CARD HELPER ---
    def create_metric_card(title, value, subtitle, icon, color_class, tooltip):
        tid = uid(f"t-{icon}")
        return dbc.Card([
            dbc.CardHeader([
                html.Span(title, className="small fw-bold text-uppercase opacity-75"),
                html.I(className=f"bi bi-{icon} ms-2 small", id=tid)
            ], className="d-flex justify-content-between py-1 bg-transparent border-0"),
            dbc.CardBody([
                html.H3(value, className=f"text-{color_class} text-center mb-0 fw-bold"),
                html.Small(subtitle, className="text-muted text-center d-block x-small"),
                dbc.Tooltip(tooltip, target=tid)
            ], className="pt-0 pb-2")
        ], className="h-100 bg-dark border-secondary shadow-sm")

    m_jsd = create_metric_card("Synthetic Fidelity (JSD)", f"{fidelity_score}%", "Overall distribution similarity", "bullseye", "success" if fidelity_score > 80 else "warning", "Average distribution similarity via JSD.")
    m_corr = create_metric_card("Correlation Stability", f"{corr_stability:.1f}%", "Multivariate relationships preserved", "diagram-3", "info", "Similarity between original and augmented correlation matrices.")
    m_dist = create_metric_card("Distribution Consistency", f"{dist_consistency}%", "Median feature similarity", "graph-up-arrow", "info", "Median JSD score across all individual features.")
    m_ent = create_metric_card("Entropy Preservation", ent_pres, "Information diversity maintained", "lightning-charge", ent_color, "Indicates if diversity and information depth were preserved.")
    
    m_zdr = create_metric_card("Mean Z Score Drift", f"{avg_z_drift:.2f}", "Standardized distribution shift", "arrows-move", "warning", "Average standardized difference in means (Z-drift).")
    m_cat = create_metric_card("Categorical Fidelity", f"{cat_fidelity}%", "Category distribution similarity", "tags", "info", "Stability of categorical proportions (Hellinger).")
    m_priv = create_metric_card("Privacy Risk", f"{privacy_risk}%", "Synthetic rows too close to originals", "shield-lock", "danger" if privacy_risk > 20 else "success", "Risk of data leakage via near-neighbor memorization.")
    m_dens = create_metric_card("Data Density", f"{data_density:.1f}%", "Non-missing cells in augmented data", "stack", "info", "Percentage of valid, non-null values generated.")
    
    m_cov = create_metric_card("Coverage Score", f"{coverage_score}%", "Synthetic coverage of original space", "bounding-box", "success" if coverage_score > 5 else "danger", "Percent of original high-probability regions covered by synthesis.")
    m_acc = create_metric_card("Feature Accuracy", f"{mean_feature_accuracy}%", "Mean feature similarity", "check-circle", "info", "Inverse of Mean Relative Error (MRE) of average values.")
    m_nov = create_metric_card("Novelty Score", f"{nndr:.2f}", "Memorization vs novelty indicator", "stars", "info", "NNDR: Closer to 1 means balanced memorization and novelty.")

    metrics_grid = html.Div([
        dbc.Row([
            dbc.Col(m_jsd, width=3), dbc.Col(m_corr, width=3), dbc.Col(m_dist, width=3), dbc.Col(m_ent, width=3)
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(m_zdr, width=3), dbc.Col(m_cat, width=3), dbc.Col(m_priv, width=3), dbc.Col(m_dens, width=3)
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(m_cov, width=4), dbc.Col(m_acc, width=4), dbc.Col(m_nov, width=4)
        ], className="g-3 mb-4")
    ])

    # Accordion items
    items = []
    if numeric_cols:
        stats_list = []
        for col in numeric_cols:
            a, b = df_orig_c[col].dropna(), df_aug_c[col].dropna()
            if not a.empty and not b.empty:
                _, ks_p = stats.ks_2samp(a, b)
                shift = ( (b.mean() - a.mean()) / (abs(a.mean()) + 1e-9) ) * 100
                stats_list.append(html.Tr([
                    html.Td(col, className="fw-bold x-small"), 
                    html.Td(f"{a.mean():.2f}", className="x-small"), 
                    html.Td(f"{b.mean():.2f}", className="x-small"), 
                    html.Td(f"{shift:+.1f}%", className=f"x-small {'text-danger' if abs(shift) > 10 else 'text-success'}"),
                    html.Td(f"{ks_p:.4f}", className="x-small"), 
                    html.Td(html.I(className="bi bi-check-circle-fill text-success" if ks_p > 0.05 else "bi bi-exclamation-triangle-fill text-warning"))
                ]))
        
        table_ui = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Feature", className="small"), html.Th("Orig Mean", className="small"), 
                html.Th("Aug Mean", className="small"), html.Th("Mean Shift", className="small"), 
                html.Th("KS Test (p val)", className="small"), html.Th("Status", className="small")
            ])), 
            html.Tbody(stats_list)
        ], striped=True, size="sm", className="bg-dark text-light border-secondary", bordered=True)
        
        items.append(dbc.AccordionItem(table_ui, title="📊 Statistical Inference & P-Values", item_id="ai-inference"))

    if len(numeric_cols) >= 2:
        try:
            fig_corr = px.imshow(df_aug_c[numeric_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", 
                                title="Feature Correlation (Augmented)", template="plotly_dark")
            items.append(dbc.AccordionItem(dcc.Graph(figure=fig_corr), title="🔗 Correlations Heatmap", item_id="ai-corr"))
        except: pass

    dist_elements = []
    for col in df_aug_c.columns[:10]: # Limit to first 10 for performance
        if col in numeric_cols:
            fig = px.histogram(df_combined, x=col, color="State", barmode="overlay", marginal="box", 
                               template="plotly_dark", opacity=0.7)
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6))
        elif col in cat_cols:
            fig = px.histogram(df_combined, x=col, color="State", barmode="group", 
                               template="plotly_dark", opacity=0.7)
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6))
            
    items.append(dbc.AccordionItem(dbc.Row(dist_elements, className="g-2"), title="📈 Distributions (Top 10 Feature Samples)", item_id="ai-dists"))

    return [metrics_grid, dbc.Accordion(items, start_collapsed=False, always_open=True, active_item=["ai-inference"])]
