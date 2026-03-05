import uuid

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import json

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
# NEW: Inject hidden IDs into initial rows
for row in initial_data:
    row['_row_id'] = str(uuid.uuid4())

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2([html.I(className="bi bi-cpu me-2"), "AI Data Workspace"], className="mt-4 mb-3")
        ], width=8),
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload"], color="primary",
                                    className="mt-4 float-end"),
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

# --- NEW: Generation Parameters Accordion ---
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        # Custom Instructions
                        dbc.Col([
                            html.Label("Custom Instructions:", className="fw-bold text-muted small"),
                            dbc.Textarea(
                                id="user-instruction",
                                placeholder="e.g., 'Make names sound European', 'Ensure prices are under $100'...",
                                size="sm",
                                style={"height": "60px"}
                            ),
                        ], width=8),
                        # Temperature Slider
                        dbc.Col([
                            html.Label("Creativity (Temperature):", className="fw-bold text-muted small mb-3"),
                            dcc.Slider(
                                id="temperature-slider",
                                min=0.0, max=1.0, step=0.1, value=0.3,
                                marks={0: 'Strict', 0.5: 'Balanced', 1: 'Creative'},
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], width=4)
                    ])
                ], title="⚙️ Generation Parameters", item_id="params-item")
            ], start_collapsed=True, className="mb-3")
        ], width=12)
    ]),

    # --- NEW: Live In-Situ Renaming Bar ---
    dbc.Row([
        dbc.Col([
            html.I(className="bi bi-pencil-square me-2 text-muted"),
            html.Span("Quick Rename:", className="text-muted me-2", style={"fontSize": "0.9em"}),
            dbc.Input(
                id="quick-rename-input",
                type="text",
                placeholder="Click any cell in the grid to rename its column...",
                size="sm",
                style={"width": "300px", "display": "inline-block"}
            )
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
                dashGridOptions={
                    "rowSelection": "multiple",
                    "undoRedoCellEditing": True
                },
                style={"height": "500px", "width": "100%"},
            )
        ], width=12)
    ]),

    # Store to remember which column is currently selected for renaming
    dcc.Store(id="active-col-store")

], fluid=True)


# --- Live Column Renaming Logic ---

@app.callback(
    Output("quick-rename-input", "value"),
    Output("active-col-store", "data"),
    Input("data-grid", "cellClicked"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def select_column_to_rename(cell_clicked, col_defs):
    """When a user clicks ANY cell, grab its column name and put it in the input box."""
    if not cell_clicked:
        return dash.no_update, dash.no_update

    col_id = cell_clicked["colId"]

    # Find the current human-readable header name
    current_name = col_id
    for col in col_defs:
        if col["field"] == col_id:
            current_name = col.get("headerName", col_id)
            break

    return current_name, col_id


@app.callback(
    Output("data-grid", "columnDefs", allow_duplicate=True),
    Input("quick-rename-input", "value"),
    State("active-col-store", "data"),
    State("data-grid", "columnDefs"),
    prevent_initial_call=True
)
def apply_live_rename(new_name, active_col_id, col_defs):
    """As the user types in the input box, update the column header live."""
    if not new_name or not active_col_id:
        return dash.no_update

    # Create a fresh copy of column definitions to trigger UI update
    updated_cols = list(col_defs)
    for col in updated_cols:
        if col["field"] == active_col_id:
            col["headerName"] = new_name
            break

    return updated_cols


# --- Standard Grid Logic (Upload, Add Row/Col, Bulk AI) ---

@app.callback(
    Output('data-grid', 'rowData'),
    Output('data-grid', 'columnDefs'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_table_from_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update
    df = parse_uploaded_file(contents, filename)
    new_cols = []
    for i, col in enumerate(df.columns):
        col_def = {"field": col, "headerName": col}
        if i == 0: col_def["checkboxSelection"] = True; col_def["headerCheckboxSelection"] = True
        new_cols.append(col_def)

    new_data = df_to_aggrid_format(df)
    # NEW: Inject hidden IDs into uploaded rows
    for row in new_data:
        row['_row_id'] = str(uuid.uuid4())

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
    # NEW: Give the new row a unique hidden ID
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
    new_field = f"custom_{len(current_cols)}"
    current_cols.append({"field": new_field, "headerName": "New Column", "editable": True})
    return current_cols


@app.callback(
    Output('data-grid', 'rowData', allow_duplicate=True),
    Input('btn-generate', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('data-grid', 'selectedRows'),
    State('data-grid', 'columnDefs'),
    State('algo-selector', 'value'),
    State('user-instruction', 'value'),     # NEW: Pulls the custom instructions
    State('temperature-slider', 'value'),   # NEW: Pulls the slider value
    prevent_initial_call=True
)
def run_ai_generation(n_clicks, row_data, selected_rows, col_defs, algorithm, user_instruction, temperature):
    if not n_clicks or not selected_rows:
        return dash.no_update

    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}
    header_to_field = {v: k for k, v in field_to_header.items()}

    df = pd.DataFrame(row_data).rename(columns=field_to_header)
    context_summary = get_dataset_description(df.head(5))

    updated_rows = list(row_data)
    payload = []
    indices = []

    for sel_row in selected_rows:
        # FIXED: Robust row matching using _row_id so multiple empty rows don't overwrite each other
        target_id = sel_row.get('_row_id')
        idx = next((i for i, r in enumerate(updated_rows) if r.get('_row_id') == target_id), -1)
        if idx == -1: continue

        # Create semantic row, safely ignoring the hidden _row_id
        semantic = {
            field_to_header[f]: (v if v not in [None, "", "NaN"] else None)
            for f, v in sel_row.items()
            if f in field_to_header
        }
        payload.append(semantic)
        indices.append(idx)

    # Pass the new parameters to the backend
    filled_data = generate_bulk_data(
        payload, context_summary, list(field_to_header.values()),
        algorithm, user_instruction, temperature
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