import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd

# Import the data manager and gemini client
from services.data_manager import parse_uploaded_file, create_empty_dataframe, df_to_aggrid_format, aggrid_to_df
from services.gemini_client import get_dataset_description, generate_data

# Initialize the Dash app with a Bootstrap theme AND Bootstrap Icons
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]
)
app.title = "AI Data Workspace"

# Initialize with an empty 10x5 dataframe
initial_df = create_empty_dataframe(10, 5)

# Safely add checkboxes ONLY to the first column
initial_cols = []
for i, col in enumerate(initial_df.columns):
    col_def = {"field": col, "headerName": col}
    if i == 0:
        col_def["checkboxSelection"] = True
        col_def["headerCheckboxSelection"] = True # Adds a handy "Select All" box at the top
    initial_cols.append(col_def)

initial_data = df_to_aggrid_format(initial_df)

# --- Layout Definition ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2([html.I(className="bi bi-database-fill-gear me-2"), "AI Data Workspace"], className="mt-4 mb-3")
        ], width=8),
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=dbc.Button([html.I(className="bi bi-upload me-2"), "Upload CSV/Excel"], color="primary",
                                    className="mt-4 float-end"),
                multiple=False
            ), width=4
        )
    ]),

    html.Hr(),

    # Control Panel 1: Data Manipulation
    dbc.Row([
        dbc.Col([
            dbc.Button([html.I(className="bi bi-plus-lg me-1"), "Add Row"], id="btn-add-row", color="secondary",
                       className="me-2", size="sm"),
            dbc.Button([html.I(className="bi bi-layout-three-columns me-1"), "Add Column"], id="btn-add-col",
                       color="secondary", className="me-2", size="sm"),

            # Column Renaming Tools
            html.Span(" | ", className="mx-2 text-muted"),
            dbc.Select(id="rename-col-dropdown", placeholder="Select column...",
                       style={'display': 'inline-block', 'width': '150px'}, className="me-2", size="sm"),
            dbc.Input(id="rename-col-input", placeholder="New name...",
                      style={'display': 'inline-block', 'width': '150px'}, className="me-2", size="sm"),
            dbc.Button([html.I(className="bi bi-input-cursor-text me-1"), "Rename"], id="btn-rename-col", color="info",
                       size="sm"),
        ], width=12)
    ], className="mb-3"),

    # Control Panel 2: AI Generation
    dbc.Row([
        dbc.Col([
            dbc.Button([html.I(className="bi bi-magic me-1"), "Generate with AI (Gemini)"], id="btn-generate",
                       color="success", className="me-2", size="sm"),
        ], width="auto"),
        dbc.Col([
            dbc.Select(
                id='algo-selector',
                options=[
                    {'label': 'Row-by-Row (Contextual)', 'value': 'row-by-row'},
                    {'label': 'Global Context', 'value': 'global'}
                ],
                value='row-by-row',
                style={'width': '250px'},
                size="sm"
            )
        ], width="auto")
    ], className="mb-3 align-items-center"),

    # The Ag-Grid Data Table
    dbc.Row([
        dbc.Col([
            dag.AgGrid(
                id="data-grid",
                columnDefs=initial_cols,
                rowData=initial_data,
                defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
                dashGridOptions={
                    "rowSelection": "multiple",
                    "undoRedoCellEditing": True,
                    "animateRows": True
                },
                style={"height": "400px", "width": "100%"},
            )])

    ]),

    html.Br(),

    # Placeholder for EDA Graphs
    dbc.Row([
        dbc.Col(html.H4([html.I(className="bi bi-bar-chart-line-fill me-2"), "Exploratory Data Analysis"]), width=12),
        dbc.Col(id="eda-graphs-container", width=12, className="mt-2")
    ])

], fluid=True)


# --- Core Callbacks ---

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

    # Safely apply checkboxes to the first column of the uploaded dataset
    new_columns = []
    for i, col in enumerate(df.columns):
        col_def = {"field": col, "headerName": col}
        if i == 0:
            col_def["checkboxSelection"] = True
            col_def["headerCheckboxSelection"] = True
        new_columns.append(col_def)

    new_data = df_to_aggrid_format(df)

    return new_data, new_columns


@app.callback(
    Output('data-grid', 'rowData', allow_duplicate=True),
    Input('btn-add-row', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('data-grid', 'columnDefs'),
    prevent_initial_call=True
)
def add_new_row(n_clicks, current_rows, current_cols):
    if not n_clicks:
        return dash.no_update
    new_row = {col['field']: None for col in current_cols}
    current_rows.append(new_row)
    return current_rows


@app.callback(
    Output('data-grid', 'columnDefs', allow_duplicate=True),
    Input('btn-add-col', 'n_clicks'),
    State('data-grid', 'columnDefs'),
    prevent_initial_call=True
)
def add_new_column(n_clicks, current_cols):
    if not n_clicks:
        return dash.no_update
    new_col_name = f"Custom_Col_{len(current_cols) + 1}"
    current_cols.append({"field": new_col_name, "headerName": new_col_name, "editable": True})
    return current_cols


# --- New Callbacks for Column Renaming ---

@app.callback(
    Output("rename-col-dropdown", "options"),
    Input("data-grid", "columnDefs")
)
def populate_rename_dropdown(col_defs):
    """Keeps the rename dropdown updated with the current column headers."""
    return [{"label": col.get("headerName", col["field"]), "value": col["field"]} for col in col_defs]


@app.callback(
    Output('data-grid', 'columnDefs', allow_duplicate=True),
    Input('btn-rename-col', 'n_clicks'),
    State('rename-col-dropdown', 'value'),
    State('rename-col-input', 'value'),
    State('data-grid', 'columnDefs'),
    prevent_initial_call=True
)
def rename_column(n_clicks, target_field, new_name, current_cols):
    """Changes the display header of a column without breaking the underlying data keys."""
    if not n_clicks or not target_field or not new_name:
        return dash.no_update

    for col in current_cols:
        if col['field'] == target_field:
            col['headerName'] = new_name
            break

    return current_cols


# --- AI Generation Callback ---

@app.callback(
    Output('data-grid', 'rowData', allow_duplicate=True),
    Input('btn-generate', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('data-grid', 'selectedRows'),
    State('data-grid', 'columnDefs'),
    State('algo-selector', 'value'),
    prevent_initial_call=True
)
def run_ai_generation(n_clicks, row_data, selected_rows, col_defs, algorithm):
    if not n_clicks or not selected_rows:
        return dash.no_update

    print(f"\n--- Starting Generation for {len(selected_rows)} row(s) ---")

    # Create a mapping dictionary so we can send REAL names to Gemini, not "Custom_Col_1"
    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}
    all_headers = list(field_to_header.values())
    all_fields = list(field_to_header.keys())

    df = pd.DataFrame(row_data)
    # Rename df columns for the context generator
    df_renamed = df.rename(columns=field_to_header)
    context_summary = get_dataset_description(df_renamed.head(5))

    updated_rows = list(row_data)

    for selected_row in selected_rows:
        # 1. Find the row index robustly
        row_idx = -1
        first_col = all_fields[0]
        for i, row in enumerate(updated_rows):
            if str(row.get(first_col)) == str(selected_row.get(first_col)):
                row_idx = i
                break

        if row_idx == -1:
            continue

        # 2. Translate the backend row into a semantic, human-readable row for Gemini
        semantic_row = {field_to_header[k]: v for k, v in selected_row.items() if k in field_to_header}

        # 3. Scan for empty cells and fill them
        for field in all_fields:
            val = selected_row.get(field)
            if pd.isna(val) or val == "" or val is None:
                target_header = field_to_header[field]
                print(f"Generating value for: {target_header}...")

                new_value = generate_data(
                    target_column=target_header,
                    context_data=semantic_row,
                    dataset_context=context_summary,
                    all_columns=all_headers,
                    algorithm=algorithm
                )

                print(f" -> Result: {new_value}")

                # Update our backend lists
                updated_rows[row_idx][field] = new_value
                selected_row[field] = new_value

                # Crucially, update the semantic row so the next empty cell
                # in this same row knows what Gemini just generated!
                semantic_row[target_header] = new_value

    print("--- Generation Complete ---")
    return updated_rows


if __name__ == '__main__':
    app.run(debug=True)