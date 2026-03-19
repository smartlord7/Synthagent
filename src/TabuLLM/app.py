import uuid
from datetime import datetime
import json

import numpy as np
import pandas as pd
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from scipy.stats import stats
from simsimd import jensenshannon
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

from services.data_manager import parse_uploaded_file, create_empty_dataframe, df_to_aggrid_format
from services.gemini_client import get_dataset_description, generate_bulk_data

import logging

# Configure logging to both Console and a File
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debug_ai_workspace.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]
)
app.title = "TabuLLM"

# Initial Data Setup
initial_df = create_empty_dataframe(10, 5)
initial_cols = []
for i, col in enumerate(initial_df.columns):
    col_def = {"field": col, "headerName": col}
    if i == 0:
        col_def["checkboxSelection"] = True
        col_def["headerCheckboxSelection"] = True
    initial_cols.append(col_def)

# Create a reusable style function
ai_highlight_style = {
    "styleConditions": [
        {
            "condition": "params.data['is_ai_' + params.colDef.field]",
            "style": {"backgroundColor": "#e8f5e9", "borderLeft": "3px solid #4caf50"}
        }
    ]
}
# Update initial columns
initial_cols = []
for i, col in enumerate(initial_df.columns):
    col_def = {"field": col, "headerName": col, "cellStyle": ai_highlight_style}
    if i == 0:
        col_def["checkboxSelection"] = True
        col_def["headerCheckboxSelection"] = True
    initial_cols.append(col_def)

# Add the Audit Column (Read-only)
initial_cols.append({
    "field": "ai_audit",
    "headerName": "Generation Audit",
    "editable": False,
    "cellStyle": {"color": "#666", "fontSize": "0.8em", "fontStyle": "italic"}
})

initial_data = df_to_aggrid_format(initial_df)
for row in initial_data:
    row['_row_id'] = str(uuid.uuid4())

row_class_rules = {
    "row-context": "params.data.is_context",
    "row-target": "params.data.is_target",
}

# --- UI LAYOUT ---
app.layout = dbc.Container([
    # Stores for tracking Original vs Augmented state
    dcc.Store(id="original-data-store", data=initial_data),
    dcc.Store(id="active-col-store"),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Store(id="context-row-ids", data=[]),
    dcc.Store(id="target-row-ids", data=[]),
    dcc.Store(id="selection-mode", data="target"), # Default mode

    dbc.Row([
        dbc.Col(html.H2([html.I(className="bi bi-cpu me-2"), "TabuLLM"], className="mt-4 mb-3"), width=8),
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Dataset"],
                                    color="primary", className="mt-4 float-end"),
                multiple=False
            )
        ], width=4)
    ]),

    html.Hr(),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Button([html.I(className="bi bi-plus-square me-1"), "Row"], id="btn-add-row", color="secondary",
                       className="me-2", size="sm"),
            dbc.Button([html.I(className="bi bi-layout-three-columns me-1"), "Column"], id="btn-add-col",
                       color="secondary", className="me-2", size="sm"),
            html.Span(" | ", className="mx-2 text-muted"),
            dbc.Button([html.I(className="bi bi-magic me-1"), "Generate"], id="btn-generate", color="success",
                       className="me-2", size="sm"),
            html.Span(" | ", className="mx-2 text-muted"),
            # NEW: Export Button
            dbc.Button([html.I(className="bi bi-download me-1"), "Export CSV"], id="btn-export", color="dark",
                       outline=True, size="sm"),
        ], width="auto"),
        dbc.Col([
            dbc.Select(
                id='algo-selector',
                options=[{'label': 'Contextual Bulk', 'value': 'bulk'},
                         {'label': 'Independent Fill', 'value': 'independent'}],
                value='bulk', style={'width': '200px'}, size="sm"
            )
        ], width="auto"),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button([html.I(className="bi bi-info-circle me-1"), "Set Context"], id="btn-mode-context",
                           color="info", outline=True, size="sm"),
                dbc.Button([html.I(className="bi bi-bullseye me-1"), "Set Targets"], id="btn-mode-target",
                           color="danger", outline=True, size="sm"),
            ], className="me-2"),

            dbc.Checklist(
                options=[{"label": "Use entire dataset as context", "value": 1}],
                value=[], id="check-full-context", switch=True, inline=True, className="small text-muted"
            ),
        ], width="auto"),
    ], className="mb-3 align-items-center"),

    # Dynamic Generation Parameters
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Style & Length:", className="fw-bold text-muted small"),
                            dcc.Dropdown(id="param-tone",
                                         options=['Concise / Standard', 'Descriptive', 'Professional', 'Casual'],
                                         value='Concise / Standard', clearable=False)
                        ], width=4),
                        dbc.Col([
                            html.Label("Data Consistency:", className="fw-bold text-muted small"),
                            dcc.Dropdown(id="param-consistency",
                                         options=['Strict (Perfectly uniform)', 'Realistic (Natural variations)'],
                                         value='Strict (Perfectly uniform)', clearable=False)
                        ], width=4),
                        dbc.Col([
                            html.Label("Value Uniqueness:", className="fw-bold text-muted small"),
                            dcc.Dropdown(id="param-uniqueness",
                                         options=['Allow Duplicates (Standard)', 'Prefer Unique Values'],
                                         value='Allow Duplicates (Standard)', clearable=False)
                        ], width=4),
                        dbc.Col([
                            html.Label("Creativity (Temperature):", className="fw-bold text-muted small"),
                            dcc.Slider(id="temperature-slider", min=0.0, max=1.0, step=0.1, value=0.3,
                                       marks={0: 'Strict', 1: 'Creative'})
                        ], width=6, className="mt-3")
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6([html.I(className="bi bi-sliders me-2"), "Custom Rules & Constraints"],
                            className="text-muted mb-3"),
                    html.Div(id="custom-params-container", children=[]),
                    dbc.Button([html.I(className="bi bi-plus-circle me-1"), "Add Custom Rule"], id="btn-add-param",
                               size="sm", color="secondary", outline=True, className="mt-2")
                ], title="⚙️ Generation Parameters & Rules", item_id="params-item")
            ], start_collapsed=True, className="mb-3")
        ], width=12)
    ]),

    # Live In-Situ Renaming Bar
    dbc.Row([
        dbc.Col([
            html.I(className="bi bi-pencil-square me-2 text-muted"),
            html.Span("Quick Rename:", className="text-muted me-2", style={"fontSize": "0.9em"}),
            dbc.Input(id="quick-rename-input", type="text", placeholder="Click any cell to rename its column...",
                      size="sm", style={"width": "300px", "display": "inline-block"})
        ], width=12, className="mb-2 bg-light p-2 rounded border")
    ]),

    # The Grid
    dbc.Row([
        dbc.Col([
            dag.AgGrid(
                id="data-grid",
                columnDefs=initial_cols,
                rowData=initial_data,
                rowClassRules=row_class_rules, # Add this
                defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
                dashGridOptions={"rowSelection": "multiple", "undoRedoCellEditing": True},
                style={"height": "400px", "width": "100%"},
            )
        ], width=12)
    ]),

    html.Hr(className="my-5"),

    # --- NEW: EXPLORATORY DATA ANALYSIS DASHBOARD ---
    dbc.Row([
        dbc.Col([
            html.H3([html.I(className="bi bi-bar-chart-fill me-2"), "Exploratory Data Analysis"], className="mb-3"),
            html.P(
                "Compare the structural integrity and distribution of your Original Dataset vs. the AI-Augmented Dataset.",
                className="text-muted"),
            dbc.Button([html.I(className="bi bi-arrow-repeat me-2"), "Refresh Dashboard"], id="btn-refresh-eda",
                       color="info", className="mb-4")
        ], width=8)
    ]),

    # Container for dynamic EDA graphs
    html.Div(id="eda-dashboard-container", children=[
        dbc.Alert("Click 'Refresh Dashboard' to generate insights.", color="light")
    ], className="mb-5")

], fluid=True)


# --- EXPORT LOGIC ---
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("data-grid", "rowData"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def export_augmented_data(n_clicks, row_data, col_defs):
    if not n_clicks or not row_data: return dash.no_update

    # Map internal fields to user-facing headers and drop hidden IDs
    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}
    df = pd.DataFrame(row_data)
    # Filter out internal tracking columns (IDs and Highlight flags)
    cols_to_exclude = [c for c in df.columns if c.startswith('is_ai_') or c == '_row_id']
    df = df.drop(columns=cols_to_exclude)

    # Only keep visible columns and rename them
    df = df[[k for k in field_to_header.keys() if k in df.columns]]
    df = df.rename(columns=field_to_header)

    return dcc.send_data_frame(df.to_csv, "AI_Augmented_Dataset.csv", index=False)


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
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics.pairwise import euclidean_distances

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


# --- DYNAMIC PARAMETER LOGIC ---
@app.callback(
    Output('custom-params-container', 'children'),
    Input('btn-add-param', 'n_clicks'),
    State('custom-params-container', 'children'),
    prevent_initial_call=True
)
def add_custom_param_row(n_clicks, children):
    new_idx = n_clicks
    new_row = dbc.Row([
        dbc.Col(
            dbc.Input(id={'type': 'cp-name', 'index': new_idx}, placeholder="Rule Name (e.g. Price Range)", size="sm"),
            width=3),
        dbc.Col(dbc.Select(
            id={'type': 'cp-type', 'index': new_idx},
            options=[{'label': 'Text Instruction', 'value': 'text'}, {'label': 'Specific Options', 'value': 'options'},
                     {'label': 'Number Range', 'value': 'number'}],
            value='text', size="sm"
        ), width=3),
        dbc.Col([
            dbc.Input(id={'type': 'cp-val-text', 'index': new_idx}, type="text", placeholder="e.g. Under 50 words...",
                      size="sm", style={'display': 'block'}),
            dbc.InputGroup([
                dbc.InputGroupText("Min"),
                dbc.Input(id={'type': 'cp-val-min', 'index': new_idx}, type="number"),
                dbc.InputGroupText("Max"),
                dbc.Input(id={'type': 'cp-val-max', 'index': new_idx}, type="number")
            ], id={'type': 'cp-val-num-group', 'index': new_idx}, size="sm", style={'display': 'none'})
        ], width=6)
    ], className="mb-2 align-items-center")
    children.append(new_row)
    return children


@app.callback(
    Output({'type': 'cp-val-text', 'index': MATCH}, 'style'),
    Output({'type': 'cp-val-text', 'index': MATCH}, 'placeholder'),
    Output({'type': 'cp-val-num-group', 'index': MATCH}, 'style'),
    Input({'type': 'cp-type', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_param_inputs(p_type):
    if p_type == 'number':
        return {'display': 'none'}, "", {'display': 'flex'}
    elif p_type == 'options':
        return {'display': 'block'}, "e.g., Active, Pending, Closed", {'display': 'none'}
    else:
        return {'display': 'block'}, "e.g., Must be under 50 words", {'display': 'none'}


# --- IN-SITU RENAMING LOGIC ---
@app.callback(
    Output("quick-rename-input", "value"),
    Output("active-col-store", "data"),
    Input("data-grid", "cellClicked"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def select_column_to_rename(cell_clicked, col_defs):
    if not cell_clicked: return dash.no_update, dash.no_update
    col_id = cell_clicked["colId"]
    current_name = next((col.get("headerName", col_id) for col in col_defs if col["field"] == col_id), col_id)
    return current_name, col_id


@app.callback(
    Output("data-grid", "columnDefs", allow_duplicate=True),
    Input("quick-rename-input", "value"),
    State("active-col-store", "data"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def apply_live_rename(new_name, active_col_id, col_defs):
    if not new_name or not active_col_id: return dash.no_update
    updated_cols = list(col_defs)
    for col in updated_cols:
        if col["field"] == active_col_id:
            col["headerName"] = new_name
            break
    return updated_cols


# --- TABLE MECHANICS (Including tracking original state) ---
@app.callback(
    Output('data-grid', 'rowData'),
    Output('data-grid', 'columnDefs'),
    Output('original-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_table_from_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update

    logger.info(f"Uploading and parsing file: {filename}")

    # Parse the file using your existing service
    df = parse_uploaded_file(contents, filename)

    # 1. Build Column Definitions with AI styling and Audit column
    new_cols = []
    for i, col in enumerate(df.columns):
        col_def = {
            "field": col,
            "headerName": col,
            "cellStyle": ai_highlight_style,
            "editable": True
        }
        # Add selection checkboxes to the first column only
        if i == 0:
            col_def["checkboxSelection"] = True
            col_def["headerCheckboxSelection"] = True
        new_cols.append(col_def)

    # 2. Append the Audit Column (the hidden logic Gemini populates)
    new_cols.append({
        "field": "ai_audit",
        "headerName": "Generation Audit",
        "editable": False,
        "cellStyle": {"color": "#666", "fontSize": "0.8em", "fontStyle": "italic"}
    })

    # 3. Format Data and Inject Unique IDs
    new_data = df_to_aggrid_format(df)
    for row in new_data:
        row['_row_id'] = str(uuid.uuid4())
        row['is_context'] = False
        row['is_target'] = False
        row['ai_audit'] = ""  # Initialize empty audit string

    logger.info(f"Successfully loaded {len(new_data)} rows with {len(new_cols)} columns.")

    # Return the same data to both the Grid and the Original Store baseline
    return new_data, new_cols, new_data


@app.callback(
    Output('data-grid', 'rowData', allow_duplicate=True),
    Input('btn-add-row', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('data-grid', 'columnDefs'),
    prevent_initial_call=True
)
def add_new_row(n_clicks, current_rows, current_cols):
    if not n_clicks: return dash.no_update
    new_row = {col['field']: None for col in current_cols}
    new_row['_row_id'] = str(uuid.uuid4())
    current_rows.append(new_row)
    return current_rows


@app.callback(
    Output('data-grid', 'columnDefs', allow_duplicate=True),
    Input('btn-add-col', 'n_clicks'),
    State('data-grid', 'columnDefs'),
    prevent_initial_call=True
)
def add_new_column(n_clicks, current_cols):
    if not n_clicks: return dash.no_update
    current_cols.append({"field": f"custom_{len(current_cols)}", "headerName": "New Column", "editable": True})
    return current_cols


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

    # 3. Target Mapping (FIXED INDENTATION STARTS HERE)
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


@app.callback(
    Output("context-row-ids", "data"),
    Output("target-row-ids", "data"),
    Output("data-grid", "rowData", allow_duplicate=True),
    Input("btn-mode-context", "n_clicks"),
    Input("btn-mode-target", "n_clicks"),
    State("data-grid", "selectedRows"),
    State("data-grid", "rowData"),
    State("context-row-ids", "data"),
    State("target-row-ids", "data"),
    State("check-full-context", "value"),  # Added this state
    prevent_initial_call=True
)
def manage_selections(btn_ctx, btn_tgt, selected_rows, current_data, ctx_ids, tgt_ids, full_ctx_val):
    ctx_ids = set(ctx_ids or [])
    tgt_ids = set(tgt_ids or [])
    is_full_ctx = len(full_ctx_val or []) > 0

    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    selected_uuids = [r["_row_id"] for r in selected_rows] if selected_rows else []

    # 1. Update the ID stores (Manual selections)
    for uuid in selected_uuids:
        if triggered_id == "btn-mode-context":
            if uuid in ctx_ids:
                ctx_ids.remove(uuid)
            else:
                ctx_ids.add(uuid)
                if uuid in tgt_ids: tgt_ids.remove(uuid)
        else:
            if uuid in tgt_ids:
                tgt_ids.remove(uuid)
            else:
                tgt_ids.add(uuid)
                if uuid in ctx_ids: ctx_ids.remove(uuid)

    # 2. Map visual flags back to rowData
    for row in current_data:
        row_id = row["_row_id"]

        # Target logic is always the same (Red)
        row["is_target"] = row_id in tgt_ids

        # Context logic (Blue) now respects the global switch
        if is_full_ctx:
            # If global switch is ON, show blue unless it's a target
            row["is_context"] = not row["is_target"]
        else:
            # If global switch is OFF, only show blue if manually selected
            row["is_context"] = row_id in ctx_ids

    return list(ctx_ids), list(tgt_ids), current_data


@app.callback(
    Output("data-grid", "rowData", allow_duplicate=True),
    Input("check-full-context", "value"),
    State("data-grid", "rowData"),
    State("context-row-ids", "data"),
    State("target-row-ids", "data"),
    prevent_initial_call=True
)
def toggle_full_dataset_visual(full_ctx_value, current_data, ctx_ids, tgt_ids):
    is_full_ctx = len(full_ctx_value) > 0
    ctx_ids_set = set(ctx_ids or [])

    for row in current_data:
        # If full context is ON, everyone is blue unless they are specifically a red target
        if is_full_ctx:
            row["is_context"] = not row.get("is_target", False)
        else:
            # Revert to only the manually selected IDs
            row["is_context"] = row["_row_id"] in ctx_ids_set

    return current_data

if __name__ == '__main__':
    app.run(debug=True)