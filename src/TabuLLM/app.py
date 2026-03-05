import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from services.data_manager import parse_uploaded_file, create_empty_dataframe, df_to_aggrid_format
from services.gemini_client import get_dataset_description, generate_bulk_data

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]
)
app.title = "AI Data Workspace"

# Initial Data Setup
initial_df = create_empty_dataframe(10, 5)
initial_cols = []
for i, col in enumerate(initial_df.columns):
    col_def = {"field": col, "headerName": col}
    if i == 0:
        col_def["checkboxSelection"] = True
        col_def["headerCheckboxSelection"] = True
    initial_cols.append(col_def)

initial_data = df_to_aggrid_format(initial_df)
for row in initial_data:
    row['_row_id'] = str(uuid.uuid4())

# --- UI LAYOUT ---
app.layout = dbc.Container([
    # Stores for tracking Original vs Augmented state
    dcc.Store(id="original-data-store", data=initial_data),
    dcc.Store(id="active-col-store"),
    dcc.Download(id="download-dataframe-csv"),

    dbc.Row([
        dbc.Col(html.H2([html.I(className="bi bi-cpu me-2"), "AI Data Workspace"], className="mt-4 mb-3"), width=8),
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
    if '_row_id' in df.columns:
        df = df.drop(columns=['_row_id'])

    # Only keep visible columns and rename them
    df = df[[k for k in field_to_header.keys() if k in df.columns]]
    df = df.rename(columns=field_to_header)

    return dcc.send_data_frame(df.to_csv, "AI_Augmented_Dataset.csv", index=False)


# --- EDA DASHBOARD LOGIC ---
@app.callback(
    Output("eda-dashboard-container", "children"),
    Input("btn-refresh-eda", "n_clicks"),
    State("original-data-store", "data"),
    State("data-grid", "rowData"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def build_eda_dashboard(n_clicks, orig_data, aug_data, col_defs):
    if not n_clicks: return dash.no_update

    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}

    # 1. Prepare DataFrames
    df_orig = pd.DataFrame(orig_data)
    df_aug = pd.DataFrame(aug_data)

    # Clean up hidden rows and rename
    for df in [df_orig, df_aug]:
        if '_row_id' in df.columns: df.drop(columns=['_row_id'], inplace=True)
        # Drop columns that don't exist in field_to_header to prevent errors
        cols_to_keep = [c for c in df.columns if c in field_to_header]
        df.drop(columns=[c for c in df.columns if c not in cols_to_keep], inplace=True)
        df.rename(columns=field_to_header, inplace=True)

    # Clean up empty strings to true NaNs for accurate counting
    df_orig.replace(["", "NaN", "None"], pd.NA, inplace=True)
    df_aug.replace(["", "NaN", "None"], pd.NA, inplace=True)

    dashboard_elements = []

    # 2. METRICS CARDS (Top Level Insight)
    orig_nulls = df_orig.isna().sum().sum()
    aug_nulls = df_aug.isna().sum().sum()
    null_reduction = orig_nulls - aug_nulls

    metrics_row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Original Dimensions", className="text-muted"),
            html.H3(f"{df_orig.shape[0]} Rows x {df_orig.shape[1]} Cols")
        ])), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Augmented Dimensions", className="text-muted"),
            html.H3(f"{df_aug.shape[0]} Rows x {df_aug.shape[1]} Cols")
        ])), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Missing Values Resolved by AI", className="text-muted"),
            html.H3(f"{null_reduction} Cells Filled", className="text-success" if null_reduction > 0 else "text-info")
        ])), width=4),
    ], className="mb-4")
    dashboard_elements.append(metrics_row)

    # 3. MISSING VALUES COMPARISON (Bar Chart)
    nulls_df = pd.DataFrame({
        'Column': df_orig.columns,
        'Original Missing': df_orig.isna().sum().values,
        'Augmented Missing': df_aug.isna().sum().values
    }).melt(id_vars='Column', var_name='State', value_name='Missing Count')

    fig_nulls = px.bar(nulls_df, x='Column', y='Missing Count', color='State', barmode='group',
                       title="Missing Values per Column (Original vs Augmented)",
                       color_discrete_map={"Original Missing": "#e74c3c", "Augmented Missing": "#2ecc71"})

    dashboard_elements.append(dbc.Row(dbc.Col(dcc.Graph(figure=fig_nulls), width=12), className="mb-4"))

    # 4. COLUMN DISTRIBUTIONS (Dynamic Generation)
    dist_row_elements = []

    for col in df_aug.columns:
        if col not in df_orig.columns: continue

        # Check if numerical
        is_num = pd.api.types.is_numeric_dtype(pd.to_numeric(df_aug[col], errors='ignore'))

        if is_num:
            # Numerical Overlay Histogram
            s_orig = pd.to_numeric(df_orig[col], errors='coerce').dropna()
            s_aug = pd.to_numeric(df_aug[col], errors='coerce').dropna()

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s_orig, name='Original', opacity=0.5, marker_color='blue'))
            fig.add_trace(go.Histogram(x=s_aug, name='Augmented', opacity=0.5, marker_color='green'))
            fig.update_layout(barmode='overlay', title=f"Distribution: {col}", margin=dict(l=20, r=20, t=40, b=20))
            dist_row_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6, className="mb-4"))
        else:
            # Categorical Frequency Bar
            val_counts_orig = df_orig[col].value_counts().head(10).reset_index()
            val_counts_orig['State'] = 'Original'
            val_counts_aug = df_aug[col].value_counts().head(10).reset_index()
            val_counts_aug['State'] = 'Augmented'

            cat_df = pd.concat([val_counts_orig, val_counts_aug])
            if not cat_df.empty:
                cat_df.columns = ['Category', 'Count', 'State']
                fig = px.bar(cat_df, x='Category', y='Count', color='State', barmode='group',
                             title=f"Top 10 Frequencies: {col}",
                             color_discrete_map={"Original": "blue", "Augmented": "green"})
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                dist_row_elements.append(dbc.Col(dcc.Graph(figure=fig), width=6, className="mb-4"))

    if dist_row_elements:
        dashboard_elements.append(
            html.H5("Feature Distributions & Augmented Variance", className="mb-3 mt-4 text-muted"))
        # Chunk into rows of 2
        for i in range(0, len(dist_row_elements), 2):
            dashboard_elements.append(dbc.Row(dist_row_elements[i:i + 2]))

    return dashboard_elements


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
    Output('original-data-store', 'data'),  # NEW: Update original state on upload
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_table_from_upload(contents, filename):
    if contents is None: return dash.no_update, dash.no_update, dash.no_update
    df = parse_uploaded_file(contents, filename)
    new_cols = [{"field": col, "headerName": col, "checkboxSelection": i == 0, "headerCheckboxSelection": i == 0} for
                i, col in enumerate(df.columns)]
    new_data = df_to_aggrid_format(df)
    for row in new_data: row['_row_id'] = str(uuid.uuid4())
    # The uploaded data becomes the new "Original" baseline
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
    Input('btn-generate', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('data-grid', 'selectedRows'),
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
def run_ai_generation(n_clicks, row_data, selected_rows, col_defs, algorithm,
                      param_tone, param_consistency, param_uniqueness, temperature,
                      cp_names, cp_types, cp_texts, cp_mins, cp_maxs):
    if not n_clicks or not selected_rows: return dash.no_update

    compiled_instructions = f"- Global Text Style: {param_tone}\n- Data Consistency: {param_consistency}\n- Uniqueness rule: {param_uniqueness}\n"
    for i in range(len(cp_names)):
        if not cp_names[i]: continue
        if cp_types[i] in ['text', 'options'] and cp_texts[i]:
            compiled_instructions += f"- {cp_names[i]} ({cp_types[i]}): {cp_texts[i]}\n"
        elif cp_types[i] == 'number':
            compiled_instructions += f"- {cp_names[i]} (limit): Min {cp_mins[i]}, Max {cp_maxs[i]}\n"

    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}
    header_to_field = {v: k for k, v in field_to_header.items()}
    df = pd.DataFrame(row_data).rename(columns=field_to_header)
    context_summary = get_dataset_description(df.head(5))

    updated_rows = list(row_data)
    payload, indices = [], []

    for sel_row in selected_rows:
        target_id = sel_row.get('_row_id')
        idx = next((i for i, r in enumerate(updated_rows) if r.get('_row_id') == target_id), -1)
        if idx == -1: continue

        semantic = {field_to_header[f]: (v if v not in [None, "", "NaN"] else None) for f, v in sel_row.items() if
                    f in field_to_header}
        payload.append(semantic)
        indices.append(idx)

    filled_data = generate_bulk_data(
        rows_payload=payload, context=context_summary, all_columns=list(field_to_header.values()),
        algorithm=algorithm, user_instructions=compiled_instructions, temperature=temperature
    )

    if filled_data:
        for i, gen_row in enumerate(filled_data):
            target_idx = indices[i]
            for header, val in gen_row.items():
                if header in header_to_field:
                    updated_rows[target_idx][header_to_field[header]] = val

    return updated_rows


if __name__ == '__main__':
    app.run(debug=True)