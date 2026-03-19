import pandas as pd
import numpy as np
import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
from scipy.stats import stats
from simsimd import jensenshannon
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

from core import app

# --- EDA DASHBOARD LOGIC ---
@app.callback(
    Output("eda-dashboard-container", "children", allow_duplicate=True),
    Input("btn-refresh-eda", "n_clicks"),
    State("original-data-store", "data"),
    State("data-grid", "rowData"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def build_eda_dashboard(n_clicks, orig_data, aug_data, col_defs):
    if not n_clicks:
        return dash.no_update

    # -----------------------------
    # DATA PREPARATION
    # -----------------------------
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
            df.drop(columns=['_row_id'], inplace=True, errors='ignore')

        cols_to_keep = [c for c in df.columns if c in field_to_header]

        df.drop(columns=[c for c in df.columns if c not in cols_to_keep],
                inplace=True,
                errors='ignore')

        df.rename(columns=field_to_header, inplace=True)

        df.replace(["", "NaN", "None", None], pd.NA, inplace=True)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    df_combined = pd.concat(
        [
            df_orig.assign(State='Original'),
            df_aug.assign(State='Augmented')
        ],
        ignore_index=True
    )

    numeric_cols = df_aug.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df_aug.columns if c not in numeric_cols]

    # -----------------------------
    # METRIC ACCUMULATORS
    # -----------------------------
    jsd_scores = []
    mre_list = []
    hellinger_cats = []
    z_drift_list = []

    corr_stability = 0
    privacy_risk = 0
    coverage_score = 0
    nndr = 0

    # -----------------------------
    # NUMERIC ANALYSIS
    # -----------------------------
    if numeric_cols:

        for col in numeric_cols:

            a = df_orig[col].dropna()
            b = df_aug[col].dropna()

            if len(a) > 5 and len(b) > 5:

                hist_a, bin_edges = np.histogram(a, bins=20, density=True)
                hist_b, _ = np.histogram(b, bins=bin_edges, density=True)

                jsd = jensenshannon(hist_a, hist_b)

                jsd_scores.append(1 - jsd if not np.isnan(jsd) else 0)

                # Z drift
                z_drift = abs(a.mean() - b.mean()) / (a.std() if a.std() > 0 else 1)

                z_drift_list.append(z_drift)

                # mean relative error
                if a.mean() != 0:
                    mre = abs(a.mean() - b.mean()) / abs(a.mean())
                    mre_list.append(mre)

        # Correlation stability
        if len(numeric_cols) > 1:
            c_orig = df_orig[numeric_cols].corr().fillna(0).values
            c_aug = df_aug[numeric_cols].corr().fillna(0).values

            corr_stability = (1 - np.mean(np.abs(c_orig - c_aug))) * 100

        # -----------------------------
        # PRIVACY / DISTANCE METRICS
        # -----------------------------

        scaler = MinMaxScaler()

        combined_numeric = pd.concat(
            [df_orig[numeric_cols], df_aug[numeric_cols]]
        ).fillna(0)

        scaler.fit(combined_numeric)

        orig_scaled = scaler.transform(df_orig[numeric_cols].fillna(0))
        aug_scaled = scaler.transform(df_aug[numeric_cols].fillna(0))

        sample_size = min(200, len(aug_scaled))
        aug_sample = aug_scaled[:sample_size]

        dist_matrix = euclidean_distances(aug_sample, orig_scaled)

        min_distances = np.min(dist_matrix, axis=1)

        privacy_risk = int((min_distances < 0.05).mean() * 100)
        coverage_score = int((min_distances > 0.1).mean() * 100)

        # NN distance ratio
        if len(orig_scaled) > 1 and len(aug_sample) > 1:
            aug_to_aug = euclidean_distances(aug_sample, aug_sample)

            np.fill_diagonal(aug_to_aug, np.inf)

            nn_aug = np.min(aug_to_aug, axis=1)
            nn_orig = min_distances

            nndr = np.mean(nn_orig / (nn_aug + 1e-6))

    # -----------------------------
    # CATEGORICAL ANALYSIS
    # -----------------------------
    if cat_cols:

        for col in cat_cols:
            p = df_orig[col].value_counts(normalize=True)
            q = df_aug[col].value_counts(normalize=True)

            all_cats = p.index.union(q.index)

            p = p.reindex(all_cats, fill_value=0).values
            q = q.reindex(all_cats, fill_value=0).values

            h_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

            hellinger_cats.append(1 - h_dist)

    # -----------------------------
    # FINAL METRICS
    # -----------------------------
    fidelity_score = int(np.mean(jsd_scores) * 100) if jsd_scores else 0
    avg_z_drift = np.mean(z_drift_list) if z_drift_list else 0
    mean_feature_error = int((1 - np.mean(mre_list)) * 100) if mre_list else 0
    cat_fidelity = int(np.mean(hellinger_cats) * 100) if hellinger_cats else 0
    distribution_consistency = int(np.median(jsd_scores) * 100) if jsd_scores else 0

    # -----------------------------
    # COMPLETENESS
    # -----------------------------
    orig_nulls = df_orig.isna().sum().sum()
    aug_nulls = df_aug.isna().sum().sum()

    null_reduction = max(0, int(orig_nulls - aug_nulls))

    total_cells_aug = df_aug.shape[0] * df_aug.shape[1]

    completeness = (
        (total_cells_aug - aug_nulls) / total_cells_aug * 100
        if total_cells_aug > 0
        else 0
    )

    score_color = (
        "success"
        if fidelity_score > 80
        else "warning"
        if fidelity_score > 50
        else "danger"
    )

    # -----------------------------
    # DASHBOARD ROW 1
    # -----------------------------
    row_1 = dbc.Row([

        dbc.Col(dbc.Card([
            dbc.CardHeader("Synthetic Fidelity (JSD)", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{fidelity_score}%", className=f"text-{score_color} text-center"),
                html.P("Overall distribution similarity", className="small text-muted text-center")
            ])
        ], outline=True, color=score_color), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Correlation Stability", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{corr_stability:.1f}%", className="text-info text-center"),
                html.P("Multivariate relationships preserved", className="small text-muted text-center")
            ])
        ], outline=True, color="info"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Distribution Consistency", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{distribution_consistency}%", className="text-secondary text-center"),
                html.P("Median feature similarity", className="small text-muted text-center")
            ])
        ], outline=True, color="secondary"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Entropy Preservation", className="fw-bold"),
            dbc.CardBody([
                html.H2("High" if fidelity_score > 75 else "Low", className="text-info text-center"),
                html.P("Information diversity maintained", className="small text-muted text-center")
            ])
        ], outline=True, color="info"), width=3)

    ], className="g-4 mb-4")

    # -----------------------------
    # DASHBOARD ROW 2
    # -----------------------------
    row_2 = dbc.Row([

        dbc.Col(dbc.Card([
            dbc.CardHeader("Mean Z-Score Drift", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{avg_z_drift:.2f}", className="text-warning text-center"),
                html.P("Standardized distribution shift", className="small text-muted text-center")
            ])
        ], outline=True, color="warning"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Categorical Fidelity", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{cat_fidelity}%", className="text-primary text-center"),
                html.P("Category distribution similarity", className="small text-muted text-center")
            ])
        ], outline=True, color="primary"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Privacy Risk", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{privacy_risk}%", className="text-danger text-center"),
                html.P("Synthetic rows too close to originals", className="small text-muted text-center")
            ])
        ], outline=True, color="danger"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Data Density", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{completeness:.1f}%", className="text-info text-center"),
                html.P("Non-missing cells in augmented data", className="small text-muted text-center")
            ])
        ], outline=True, color="info"), width=3)

    ], className="g-4 mb-4")

    # -----------------------------
    # DASHBOARD ROW 3
    # -----------------------------
    row_3 = dbc.Row([

        dbc.Col(dbc.Card([
            dbc.CardHeader("Coverage Score", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{coverage_score}%", className="text-success text-center"),
                html.P("Synthetic coverage of original space", className="small text-muted text-center")
            ])
        ], outline=True, color="success"), width=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Feature Accuracy", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{mean_feature_error}%", className="text-primary text-center"),
                html.P("Mean feature similarity", className="small text-muted text-center")
            ])
        ], outline=True, color="primary"), width=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader("NN Distance Ratio", className="fw-bold"),
            dbc.CardBody([
                html.H2(f"{nndr:.2f}", className="text-secondary text-center"),
                html.P("Memorization vs novelty indicator", className="small text-muted text-center")
            ])
        ], outline=True, color="secondary"), width=4)

    ], className="g-4 mb-4")

    # --- 4. ACCORDION SECTIONS ---
    accordion_items = []

    # Section 2: Statistical Inference
    if numeric_cols:
        stats_list = []
        for col in numeric_cols:
            a, b = df_orig[col].dropna(), df_aug[col].dropna()
            if not a.empty and not b.empty:
                _, ks_p = stats.ks_2samp(a, b)
                # Use icon logic for the status column
                status_icon = html.I(className="bi bi-check-circle-fill text-success") if ks_p > 0.05 else html.I(
                    className="bi bi-exclamation-triangle-fill text-warning")

                stats_list.append({
                    "Feature": col,
                    "Orig Mean": round(a.mean(), 2),
                    "Aug Mean": round(b.mean(), 2),
                    "Mean Shift": f"{((b.mean() - a.mean()) / a.mean() * 100):.1f}%" if a.mean() != 0 else "N/A",
                    "KS Test (p-val)": round(ks_p, 4),
                    "Status": status_icon
                })

        # Note: We build the table manually now to support html components (icons) in cells
        table_header = [
            html.Thead(html.Tr([html.Th("Feature"), html.Th("Orig Mean"), html.Th("Aug Mean"), html.Th("Mean Shift"),
                                html.Th("KS Test (p-val)"), html.Th("Status")]))
        ]
        table_body = [
            html.Tbody([
                html.Tr([
                    html.Td(row["Feature"]), html.Td(row["Orig Mean"]), html.Td(row["Aug Mean"]),
                    html.Td(row["Mean Shift"]), html.Td(row["KS Test (p-val)"]), html.Td(row["Status"])
                ]) for row in stats_list
            ])
        ]
        table = dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True, size="sm")

        accordion_items.append(
            dbc.AccordionItem(table,
                              title=[html.I(className="bi bi-calculator me-2"), "Statistical Inference & P-Values"],
                              item_id="section-stats")
        )

    # Section 3: Correlations & Multivariate
    if len(numeric_cols) >= 2:
        corr_aug = df_aug[numeric_cols].corr()
        fig_corr = px.imshow(corr_aug, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix")

        fig_scatter = px.scatter_matrix(df_combined, dimensions=numeric_cols, color="State", opacity=0.4,
                                        title="Feature Interaction Matrix")
        fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3))

        corr_content = html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_corr), width=5), dbc.Col(dcc.Graph(figure=fig_scatter), width=7)])
        ])
        accordion_items.append(dbc.AccordionItem(
            corr_content,
            title=[html.I(className="bi bi-diagram-3 me-2"), "Multivariate Relationships & Correlations"],
            item_id="section-corr")
        )

    # Section 4: Distributions (Violin / Histograms)
    dist_elements = []
    for col in df_aug.columns:
        if col in numeric_cols:
            fig = px.violin(df_combined, y=col, x="State", color="State", box=True, points="all",
                            color_discrete_map={"Original": "#3498db", "Augmented": "#2ecc71"})
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6))
            fig_h = px.histogram(df_combined, x=col, color="State", barmode="overlay", marginal="box")
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig_h), width=6))
        elif col in cat_cols:
            fig_c = px.histogram(df_combined, x=col, color="State", barmode="group", title=f"Frequency: {col}")
            dist_elements.append(dbc.Col(dcc.Graph(figure=fig_c), width=6))

    dist_grid = [dbc.Row(dist_elements[i:i + 2]) for i in range(0, len(dist_elements), 2)]
    accordion_items.append(
        dbc.AccordionItem(
            html.Div(dist_grid),
            title=[html.I(className="bi bi-bar-chart me-2"), "Feature-by-Feature Distributions"],
            item_id="section-dist")
    )

    # Section 5: High-Inference Power-Ups (PCA / Q-Q / Parallel)
    insight_elements = []

    # PCA Cluster Drift
    if len(numeric_cols) >= 2:
        df_pca_clean = df_combined.dropna(subset=numeric_cols)
        if len(df_pca_clean) > 5:
            pca = PCA(n_components=2)
            comps = pca.fit_transform(df_pca_clean[numeric_cols])
            df_res = pd.DataFrame(comps, columns=['PC1', 'PC2'])
            df_res['State'] = df_pca_clean['State'].values
            fig_pca = px.scatter(df_res, x='PC1', y='PC2', color='State', title="PCA Structural Drift Analysis")
            insight_elements.append(dbc.Row(dbc.Col(dcc.Graph(figure=fig_pca), width=12)))

    # Q-Q Plots
    if numeric_cols:
        col = numeric_cols[0]
        orig_s, aug_s = np.sort(df_orig[col].dropna()), np.sort(df_aug[col].dropna())
        if len(orig_s) > 0 and len(aug_s) > 0:
            aug_interp = np.interp(np.linspace(0, 1, len(orig_s)), np.linspace(0, 1, len(aug_s)), aug_s)
            fig_qq = px.scatter(x=orig_s, y=aug_interp, title=f"Quantile-Quantile Stability: {col}")
            fig_qq.add_shape(type="line", x0=min(orig_s), y0=min(orig_s), x1=max(orig_s), y1=max(orig_s),
                             line=dict(color="red", dash="dash"))
            insight_elements.append(dbc.Row(dbc.Col(dcc.Graph(figure=fig_qq), width=12)))

    accordion_items.append(
        dbc.AccordionItem(
            html.Div(insight_elements),
            title=[html.I(className="bi bi-brain me-2"), "High-Inference Latent Logic"],
            item_id="section-logic")
    )

    # Final Assembly
    return [
        row_1, row_2,
        dbc.Accordion(accordion_items, start_collapsed=False, always_open=True, id="eda-accordion")
    ]
