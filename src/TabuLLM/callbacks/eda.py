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

def get_eda_layout(df_orig, df_aug, dataset_id="base", results_list=None):
    if df_orig.empty or df_aug.empty:
        return [dbc.Alert("Missing data to build dashboard layouts.", color="warning")]

    df_orig_c = df_orig.copy()
    df_aug_c = df_aug.copy()

    for df in [df_orig_c, df_aug_c]:
        cols_to_drop = [c for c in df.columns if c.startswith('is_ai_') or c in ['_row_id', 'is_context', 'is_target', 'ai_audit']]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        for col in df.columns:
             if col != '_fold_id':
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col])

    df_combined = pd.concat(
        [df_orig_c.assign(State='Original'), df_aug_c.assign(State='Augmented')],
        ignore_index=True
    )

    internal_cols = ['_fold_id', '_row_id', 'is_context', 'is_target', 'ai_audit']
    numeric_cols = [c for c in df_aug_c.select_dtypes(include='number').columns if c not in internal_cols]
    cat_cols = [c for c in df_aug_c.columns if c not in numeric_cols and c not in internal_cols]
    
    # Store _fold_id separately if present for aggregation
    fold_id_col = '_fold_id' if '_fold_id' in df_aug_c.columns else None

    jsd_scores, m_drift_list, mre_list, hellinger_cats, z_drift_list = [], [], [], [], []
    corr_stability, privacy_risk, coverage_score, nndr = 0, 0, 0, 0
    entropy_preservation = 0 # Placeholder for simplicity

    # Prepare Encoded Data for All Features
    df_orig_enc = df_orig_c.copy()
    df_aug_enc = df_aug_c.copy()
    all_features = numeric_cols + cat_cols
    
    for col in cat_cols:
        combined = pd.concat([df_orig_c[col], df_aug_c[col]])
        codes, _ = pd.factorize(combined)
        df_orig_enc[col] = codes[:len(df_orig_c)]
        df_aug_enc[col] = codes[len(df_orig_c):]

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

        if len(all_features) > 1:
            c_orig = df_orig_enc[all_features].corr().fillna(0).values
            c_aug = df_aug_enc[all_features].corr().fillna(0).values
            corr_stability = (1 - np.mean(np.abs(c_orig - c_aug))) * 100

    if all_features:
        # Use encoded data for distance-based metrics
        scaler = MinMaxScaler()
        combined_enc = pd.concat([df_orig_enc[all_features], df_aug_enc[all_features]]).fillna(0)
        scaler.fit(combined_enc)
        orig_norm = scaler.transform(df_orig_enc[all_features].fillna(0))
        aug_norm = scaler.transform(df_aug_enc[all_features].fillna(0))
        
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

    # --- AGGREGATION LOGIC ---
    agg_stats = {}
    cv_summary = None
    if results_list:
        keys = [
            "Synthetic Fidelity (JSD)", "Correlation Stability", "Distribution Consistency",
            "Entropy Preservation", "Mean Z Score Drift", "Categorical Fidelity", 
            "Privacy Risk", "Data Density", "Coverage Score", "Feature Accuracy", "Novelty Score"
        ]
        for k in keys:
            vals = [r.get(k, 0) for r in results_list if k in r]
            if vals:
                agg_stats[k] = {"mean": np.mean(vals), "std": np.std(vals)}
            else:
                agg_stats[k] = {"mean": 0, "std": 0}
        
    def get_val_std(key, fallback_val):
        if key in agg_stats:
            m = agg_stats[key]["mean"]
            s = agg_stats[key]["std"]
            # Formatting logic
            if key in ["Mean Z Score Drift", "Novelty Score"]:
                return f"{m:.2f}", f"{s:.2f}"
            else:
                return f"{m:.1f}%", f"{s:.1f}"
        return str(fallback_val), None

    # --- RICH METRIC CARD HELPER ---
    def create_metric_card(title, value, subtitle, icon, color_class, tooltip, std=None):
        tid = uid(f"t-{icon}")
        return dbc.Card([
            dbc.CardHeader([
                html.Span(title, className="small fw-bold text-uppercase opacity-75"),
                html.I(className=f"bi bi-{icon} ms-2 small", id=tid)
            ], className="d-flex justify-content-between py-1 bg-transparent border-0"),
            dbc.CardBody([
                html.Div([
                    html.H3(value, className=f"text-{color_class} mb-0 fw-bold d-inline-block"),
                    html.Span(f" ±{std}", className="text-muted ms-2 small") if std else None
                ], className="d-flex align-items-baseline justify-content-center"),
                html.Small(subtitle, className="text-muted text-center d-block x-small"),
                dbc.Tooltip(tooltip, target=tid)
            ], className="pt-0 pb-2")
        ], className="h-100 bg-dark border-secondary shadow-sm")

    v_jsd, s_jsd = get_val_std("Synthetic Fidelity (JSD)", f"{fidelity_score}%")
    v_corr, s_corr = get_val_std("Correlation Stability", f"{corr_stability:.1f}%")
    v_dist, s_dist = get_val_std("Distribution Consistency", f"{dist_consistency}%")
    
    v_zdr, s_zdr = get_val_std("Mean Z Score Drift", f"{avg_z_drift:.2f}")
    v_cat, s_cat = get_val_std("Categorical Fidelity", f"{cat_fidelity}%")
    v_priv, s_priv = get_val_std("Privacy Risk", f"{privacy_risk}%")
    v_dens, s_dens = get_val_std("Data Density", f"{data_density:.1f}%")
    
    v_cov, s_cov = get_val_std("Coverage Score", f"{coverage_score}%")
    v_acc, s_acc = get_val_std("Feature Accuracy", f"{mean_feature_accuracy}%")
    v_nov, s_nov = get_val_std("Novelty Score", f"{nndr:.2f}")
    
    v_ent, s_ent = get_val_std("Entropy Preservation", ent_pres)
    if s_ent is not None:
        ent_pres = v_ent
        ent_color = "success" if float(v_ent.strip('%')) > 80 else "warning" if float(v_ent.strip('%')) > 50 else "danger"

    m_jsd = create_metric_card("Synthetic Fidelity (JSD)", v_jsd, "Overall distribution similarity", "bullseye", "success" if fidelity_score > 80 else "warning", "Average distribution similarity via JSD.", std=s_jsd)
    m_corr = create_metric_card("Correlation Stability", v_corr, "Multivariate relationships preserved", "diagram-3", "info", "Similarity between original and augmented correlation matrices.", std=s_corr)
    m_dist = create_metric_card("Distribution Consistency", v_dist, "Median feature similarity", "graph-up-arrow", "info", "Median JSD score across all individual features.", std=s_dist)
    m_ent = create_metric_card("Entropy Preservation", v_ent, "Information diversity maintained", "lightning-charge", ent_color, "Indicates if diversity and information depth were preserved.", std=s_ent)
    
    m_zdr = create_metric_card("Mean Z Score Drift", v_zdr, "Standardized distribution shift", "arrows-move", "warning", "Average standardized difference in means (Z-drift).", std=s_zdr)
    m_cat = create_metric_card("Categorical Fidelity", v_cat, "Category distribution similarity", "tags", "info", "Stability of categorical proportions (Hellinger).", std=s_cat)
    m_priv = create_metric_card("Privacy Risk", v_priv, "Synthetic rows too close to originals", "shield-lock", "danger" if privacy_risk > 20 else "success", "Risk of data leakage via near-neighbor memorization.", std=s_priv)
    m_dens = create_metric_card("Data Density", v_dens, "Non-missing cells in augmented data", "stack", "info", "Percentage of valid, non-null values generated.", std=s_dens)
    
    m_cov = create_metric_card("Coverage Score", v_cov, "Synthetic coverage of original space", "bounding-box", "success" if coverage_score > 5 else "danger", "Percent of original high-probability regions covered by synthesis.", std=s_cov)
    m_acc = create_metric_card("Feature Accuracy", v_acc, "Mean feature similarity", "check-circle", "info", "Inverse of Mean Relative Error (MRE) of average values.", std=s_acc)
    m_nov = create_metric_card("Novelty Score", v_nov, "Memorization vs novelty indicator", "stars", "info", "NNDR: Closer to 1 means balanced memorization and novelty.", std=s_nov)

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
        # Check if we have fold info for feature-level aggregation
        has_folds = '_fold_id' in df_aug_c.columns
        fold_groups = None
        if has_folds:
            try:
                fold_groups = df_aug_c.groupby('_fold_id')
            except Exception as ge:
                logger.warning(f"Grouping by fold failed: {ge}")
                has_folds = False

        for col in all_features:
            # Skip if column missing
            if col not in df_orig_c.columns or col not in df_aug_c.columns:
                continue
            
            is_num = col in numeric_cols
            a = df_orig_c[col].dropna()
            b_full = df_aug_c[col].dropna()
            
            if not a.empty and not b_full.empty:
                if is_num:
                    # 1. High-level Mean Shift (on pooled data)
                    shift = ( (b_full.mean() - a.mean()) / (abs(a.mean()) + 1e-9) ) * 100
                    
                    if has_folds and fold_groups is not None and len(fold_groups) > 1:
                        try:
                            fold_means = fold_groups[col].mean()
                            b_mean = fold_means.mean()
                            b_std = fold_means.std()
                            val_str = f"{b_mean:.2f} ±{b_std:.2f}"
                        except:
                            val_str = f"{b_full.mean():.2f}"
                    else:
                        val_str = f"{b_full.mean():.2f}"
                    
                    _, ks_p = stats.ks_2samp(a, b_full)
                    orig_val = f"{a.mean():.2f}"
                else:
                    # Categorical handling: Mode and Distribution Shift
                    orig_val = f"'{a.mode()[0]}'" if not a.mode().empty else "N/A"
                    
                    if has_folds and fold_groups is not None:
                        try:
                            # Hellinger distance per fold for std
                            h_folds, e_means = [], []
                            for _, f_df in fold_groups:
                                # For shift calculation
                                p_f = a.value_counts(normalize=True)
                                q_f = f_df[col].value_counts(normalize=True)
                                cats = p_f.index.union(q_f.index)
                                p_f = p_f.reindex(cats, fill_value=0).values
                                q_f = q_f.reindex(cats, fill_value=0).values
                                h_folds.append(np.sqrt(np.sum((np.sqrt(p_f) - np.sqrt(q_f))**2)) / np.sqrt(2))
                                
                                # For encoded stats
                                e_means.append(df_aug_enc.loc[f_df.index, col].mean())
                            
                            if len(e_means) > 1:
                                e_m, e_s = np.mean(e_means), np.std(e_means)
                                val_str = f"'{b_full.mode()[0]}' ({e_m:.1f} ±{e_s:.1f})" if not b_full.mode().empty else "N/A"
                            else:
                                e_m = np.mean(e_means)
                                val_str = f"'{b_full.mode()[0]}' ({e_m:.1f})" if not b_full.mode().empty else "N/A"
                        except:
                            val_str = f"'{b_full.mode()[0]}'" if not b_full.mode().empty else "N/A"
                    else:
                        e_m = df_aug_enc[col].mean()
                        val_str = f"'{b_full.mode()[0]}' ({e_m:.1f})" if not b_full.mode().empty else "N/A"
                    
                    # Original Mode + Encoded Mean
                    orig_e = df_orig_enc[col].mean()
                    orig_val = f"'{a.mode()[0]}' ({orig_e:.1f})" if not a.mode().empty else "N/A"
                    
                    # Chi-square for p-value
                    p_obs = a.value_counts(normalize=True)
                    q_obs = b_full.value_counts(normalize=True)
                    cats = p_obs.index.union(q_obs.index)
                    p_vals = p_obs.reindex(cats, fill_value=0).values * len(b_full)
                    q_vals = q_obs.reindex(cats, fill_value=0).values * len(b_full)
                    # Add tiny epsilon to avoid zero in expected
                    p_vals = np.where(p_vals == 0, 1e-5, p_vals)
                    try:
                        _, ks_p = stats.chisquare(q_vals, f_exp=p_vals)
                    except: ks_p = 0.0
                    
                    # For shift, use Hellinger * 100 as a proxy %
                    p_h = p_obs.reindex(cats, fill_value=0).values
                    q_h = q_obs.reindex(cats, fill_value=0).values
                    shift = (np.sqrt(np.sum((np.sqrt(p_h) - np.sqrt(q_h))**2)) / np.sqrt(2)) * 100
                    
                stats_list.append(html.Tr([
                    html.Td(col, className="fw-bold x-small"), 
                    html.Td(orig_val, className="x-small"), 
                    html.Td(val_str, className="x-small"), 
                    html.Td(f"{shift:+.1f}%", className=f"x-small {'text-danger' if abs(shift) > 10 else 'text-success'}"),
                    html.Td(f"{ks_p:.4f}", className="x-small"), 
                    html.Td(html.I(className="bi bi-check-circle-fill text-success" if ks_p > 0.05 else "bi bi-exclamation-triangle-fill text-warning"))
                ]))
        
        table_ui = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Feature", className="small"), html.Th("Orig Mean", className="small"), 
                html.Th("Aug Mean ± Std", className="small"), html.Th("Mean Shift", className="small"), 
                html.Th("KS Test (p val)", className="small"), html.Th("Status", className="small")
            ])), 
            html.Tbody(stats_list)
        ], striped=True, size="sm", className="bg-dark text-light border-secondary", bordered=True)
        
        items.append(dbc.AccordionItem(
            table_ui, 
            title=html.Div([html.I(className="bi bi-calculator me-2"), "Statistical Inference & P-Values"]), 
            item_id="ai-inference"
        ))

    if len(all_features) >= 2:
        try:
            # 1. Original Correlation Matrix
            c_orig = df_orig_enc[all_features].corr()
            
            # 2. Augmented Correlation Matrix (Averaged over folds if present)
            if has_folds and fold_groups is not None:
                fold_corrs = []
                for _, g_df in fold_groups:
                    if g_df.empty: continue
                    fold_corrs.append(df_aug_enc.loc[g_df.index, all_features].corr().fillna(0))
                if fold_corrs:
                    c_aug_avg = sum(fold_corrs) / len(fold_corrs)
                else:
                    c_aug_avg = df_aug_enc[all_features].corr().fillna(0)
            else:
                c_aug_avg = df_aug_enc[all_features].corr().fillna(0)
            
            fig_orig = px.imshow(c_orig, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 title="Relational Structure: Original baseline", template="plotly_dark")
            fig_aug = px.imshow(c_aug_avg, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                title=f"Relational Stability: Augmented average ({'Fold-Aggregated' if has_folds else 'Pooled'})", template="plotly_dark")
            
            items.append(dbc.AccordionItem(
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_orig), width=6),
                    dbc.Col(dcc.Graph(figure=fig_aug), width=6)
                ]), 
                title=html.Div([html.I(className="bi bi-link-45deg me-2"), "Multivariate Correlation Stability Audit"]), 
                item_id="ai-corr"
            ))
        except Exception as e:
            logger.warning(f"Correlation plot failed: {e}")

    # --- PCA GEOMETRIC COVERAGE ---
    if len(numeric_cols) >= 2:
        try:
            pca = PCA(n_components=2)
            scaler = MinMaxScaler()
            df_full = pd.concat([df_orig_c[numeric_cols], df_aug_c[numeric_cols]]).fillna(0)
            scaler.fit(df_full)
            
            z_orig = pca.fit_transform(scaler.transform(df_orig_c[numeric_cols].fillna(0)))
            z_aug = pca.transform(scaler.transform(df_aug_c[numeric_cols].fillna(0)))
            
            df_pca = pd.concat([
                pd.DataFrame(z_orig, columns=['PC1', 'PC2']).assign(State='Original'),
                pd.DataFrame(z_aug, columns=['PC1', 'PC2']).assign(State='Augmented')
            ])
            
            fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='State', 
                                title="Geometric Space Coverage (PCA)",
                                template="plotly_dark", opacity=0.6,
                                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})
            
            items.append(dbc.AccordionItem(
                dcc.Graph(figure=fig_pca),
                title=html.Div([html.I(className="bi bi-bounding-box-circles me-2"), "Dimension Reduction (PCA Cluster Analysis)"]),
                item_id="ai-pca"
            ))
        except Exception as e:
            logger.warning(f"PCA generation failed: {e}")

    dist_elements = []
    # Contemplate ALL features as requested
    for col in df_aug_c.columns: 
        if col in numeric_cols:
            fig = px.histogram(df_combined, x=col, color="State", barmode="overlay", marginal="box", 
                               template="plotly_dark", opacity=0.7)
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6))
        elif col in cat_cols:
            fig = px.histogram(df_combined, x=col, color="State", barmode="group", 
                               template="plotly_dark", opacity=0.7)
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6))
            
    items.append(dbc.AccordionItem(
        dbc.Row(dist_elements, className="g-2"), 
        title=html.Div([html.I(className="bi bi-graph-up me-2"), "Distributions (Top 10 Feature Samples)"]), 
        item_id="ai-dists"
    ))

    layout = [metrics_grid, dbc.Accordion(items, start_collapsed=False, always_open=True, active_item=["ai-inference"])]
    if cv_summary:
        layout = [cv_summary] + layout
    return layout
