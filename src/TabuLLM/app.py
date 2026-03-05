import uuid
import json
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, callback, MATCH, ALL
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
    dbc.Row([
        dbc.Col(html.H2([html.I(className="bi bi-cpu me-2"), "AI Data Workspace"], className="mt-4 mb-3"), width=8),
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Dataset"],
                                    color="primary", className="mt-4 float-end"),
                multiple=False
            ), width=4
        )
    ]),

    html.Hr(),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Button([html.I(className="bi bi-plus-square me-1"), "Row"], id="btn-add-row", color="secondary",
                       className="me-2", size="sm"),
            dbc.Button([html.I(className="bi bi-columns me-1"), "Column"], id="btn-add-col", color="secondary",
                       className="me-2", size="sm"),
            html.Span(" | ", className="mx-2 text-muted"),
            dbc.Button([html.I(className="bi bi-magic me-1"), "Generate"], id="btn-generate", color="success",
                       className="me-2", size="sm"),
        ], width="auto"),
        dbc.Col([
            dbc.Select(
                id='algo-selector',
                options=[
                    {'label': 'Contextual Bulk', 'value': 'bulk'},
                    {'label': 'Independent Fill', 'value': 'independent'}
                ],
                value='bulk',
                style={'width': '200px'},
                size="sm"
            )
        ], width="auto"),
    ], className="mb-3 align-items-center"),

    # --- NEW: Dynamic Generation Parameters ---
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    # 1. Default Parameters (Pre-selected dropdowns)
                    dbc.Row([
                        dbc.Col([
                            html.Label("Style & Length:", className="fw-bold text-muted small"),
                            dcc.Dropdown(
                                id="param-tone",
                                options=['Concise / Standard', 'Descriptive', 'Professional', 'Casual'],
                                value='Concise / Standard',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Data Consistency:", className="fw-bold text-muted small"),
                            dcc.Dropdown(
                                id="param-consistency",
                                options=['Strict (Perfectly uniform)', 'Realistic (Natural variations)'],
                                value='Strict (Perfectly uniform)',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Value Uniqueness:", className="fw-bold text-muted small"),
                            dcc.Dropdown(
                                id="param-uniqueness",
                                options=['Allow Duplicates (Standard)', 'Prefer Unique Values'],
                                value='Allow Duplicates (Standard)',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Creativity (Temperature):", className="fw-bold text-muted small"),
                            dcc.Slider(id="temperature-slider", min=0.0, max=1.0, step=0.1, value=0.3,
                                       marks={0: 'Strict', 1: 'Creative'})
                        ], width=6)
                    ], className="mb-3"),

                    html.Hr(),

                    # 2. Custom Rule Builder
                    html.H6([html.I(className="bi bi-sliders me-2"), "Custom Rules & Constraints"],
                            className="text-muted mb-3"),

                    # Container where dynamic rows will spawn
                    html.Div(id="custom-params-container", children=[]),

                    # Button to add new rules
                    dbc.Button([html.I(className="bi bi-plus-circle me-1"), "Add Custom Rule"], id="btn-add-param",
                               size="sm", color="secondary", outline=True, className="mt-2")

                ], title="⚙️ Generation Parameters & Rules", item_id="params-item")
            ], start_collapsed=False, className="mb-3")
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
                style={"height": "500px", "width": "100%"},
            )
        ], width=12)
    ]),

    dcc.Store(id="active-col-store")
], fluid=True)


# --- DYNAMIC PARAMETER LOGIC (Pattern Matching Callbacks) ---

@app.callback(
    Output('custom-params-container', 'children'),
    Input('btn-add-param', 'n_clicks'),
    State('custom-params-container', 'children'),
    prevent_initial_call=True
)
def add_custom_param_row(n_clicks, children):
    """Spawns a new constraint row when the user clicks 'Add Custom Rule'"""
    new_idx = n_clicks

    new_row = dbc.Row([
        # Name
        dbc.Col(
            dbc.Input(id={'type': 'cp-name', 'index': new_idx}, placeholder="Rule Name (e.g. Price Range)", size="sm"),
            width=3),
        # Type Selector
        dbc.Col(dbc.Select(
            id={'type': 'cp-type', 'index': new_idx},
            options=[
                {'label': 'Text Instruction', 'value': 'text'},
                {'label': 'Specific Options', 'value': 'options'},
                {'label': 'Number Range', 'value': 'number'}
            ],
            value='text', size="sm"
        ), width=3),
        # Dynamic Values Container (Shows/Hides based on type)
        dbc.Col([
            dbc.Input(id={'type': 'cp-val-text', 'index': new_idx}, type="text", placeholder="e.g. Under 50 words...",
                      size="sm", style={'display': 'block'}),

            # FIXED: Removed size="sm" from InputGroupText, moved it to the parent InputGroup
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
    """Dynamically changes the input field style depending on whether Text, Options, or Number is selected."""
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


# --- TABLE MECHANICS LOGIC ---

@app.callback(
    Output('data-grid', 'rowData'),
    Output('data-grid', 'columnDefs'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_table_from_upload(contents, filename):
    if contents is None: return dash.no_update, dash.no_update
    df = parse_uploaded_file(contents, filename)
    new_cols = [{"field": col, "headerName": col, "checkboxSelection": i == 0, "headerCheckboxSelection": i == 0} for
                i, col in enumerate(df.columns)]
    new_data = df_to_aggrid_format(df)
    for row in new_data: row['_row_id'] = str(uuid.uuid4())
    return new_data, new_cols


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
    # Table States
    State('data-grid', 'rowData'),
    State('data-grid', 'selectedRows'),
    State('data-grid', 'columnDefs'),
    State('algo-selector', 'value'),
    # Parameter States (The NEW Defaults)
    State('param-tone', 'value'),
    State('param-consistency', 'value'),
    State('param-uniqueness', 'value'),
    State('temperature-slider', 'value'),
    # Parameter States (Dynamic ALL matching)
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

    # 1. Compile all user rules into a single strict prompt block
    compiled_instructions = (
        f"- Global Text Style: {param_tone}\n"
        f"- Data Consistency: {param_consistency}\n"
        f"- Uniqueness rule: {param_uniqueness}\n"
    )

    # Append the custom dynamic rules
    for i in range(len(cp_names)):
        name = cp_names[i]
        p_type = cp_types[i]
        if not name: continue  # Skip if user left the name blank

        if p_type in ['text', 'options']:
            if cp_texts[i]: compiled_instructions += f"- {name} ({p_type}): {cp_texts[i]}\n"
        elif p_type == 'number':
            compiled_instructions += f"- {name} (number limit): Min {cp_mins[i]}, Max {cp_maxs[i]}\n"

    print("\n--- Sending instructions to LLM ---")
    print(compiled_instructions)

    # 2. Prepare the table mapping
    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}
    header_to_field = {v: k for k, v in field_to_header.items()}
    df = pd.DataFrame(row_data).rename(columns=field_to_header)
    context_summary = get_dataset_description(df.head(5))

    updated_rows = list(row_data)
    payload, indices = [], []

    # 3. Extract the exact rows to fill
    for sel_row in selected_rows:
        target_id = sel_row.get('_row_id')
        idx = next((i for i, r in enumerate(updated_rows) if r.get('_row_id') == target_id), -1)
        if idx == -1: continue

        semantic = {field_to_header[f]: (v if v not in [None, "", "NaN"] else None) for f, v in sel_row.items() if
                    f in field_to_header}
        payload.append(semantic)
        indices.append(idx)

    # 4. Fire the API call
    filled_data = generate_bulk_data(
        rows_payload=payload,
        context=context_summary,
        all_columns=list(field_to_header.values()),
        algorithm=algorithm,
        user_instructions=compiled_instructions,
        temperature=temperature
    )

    # 5. Unpack results into the grid
    if filled_data:
        for i, gen_row in enumerate(filled_data):
            target_idx = indices[i]
            for header, val in gen_row.items():
                if header in header_to_field:
                    updated_rows[target_idx][header_to_field[header]] = val

    return updated_rows


if __name__ == '__main__':
    app.run(debug=True)