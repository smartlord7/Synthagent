import uuid
import dash
from dash import Input, Output, State, html, dcc

import dash_bootstrap_components as dbc

from core import app, logger
from services.data_manager import parse_uploaded_file, df_to_aggrid_format
from utils.constants import ai_highlight_style

# --- TABLE MECHANICS (Including tracking original state) ---
@app.callback(
    Output('data-grid', 'rowData'),
    Output('data-grid', 'columnDefs'),
    Output('original-data-store', 'data'),
    Output('upload-file-info', 'children'),
    Output('dataset-description-store', 'data', allow_duplicate=True),
    Output('dataset-stats-grid', 'children'),
    Output('dataset-description-output', 'children', allow_duplicate=True),
    Output('data-grid', 'dashGridOptions'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('data-grid', 'dashGridOptions'),
    prevent_initial_call=True
)
def update_table_from_upload(contents, filename, current_options):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    logger.info(f"Uploading and parsing file: {filename}")
    import pandas as pd
    df = parse_uploaded_file(contents, filename)

    # 1. Build Column Definitions
    new_cols = []
    for i, col in enumerate(df.columns):
        col_def = {"field": col, "headerName": col, "cellStyle": ai_highlight_style, "editable": True}
        if i == 0:
            col_def["checkboxSelection"] = True
            col_def["headerCheckboxSelection"] = True
        new_cols.append(col_def)

    new_cols.append({
        "field": "ai_audit", "headerName": "Generation Audit", "editable": False,
        "cellStyle": {"color": "#666", "fontSize": "0.8em", "fontStyle": "italic"}
    })

    new_data = df_to_aggrid_format(df)
    for row in new_data:
        row['_row_id'] = str(uuid.uuid4())
        row['is_context'] = False
        row['is_target'] = False
        row['ai_audit'] = ""

    file_info = html.Div([
        html.Span(f"📄 {filename}", className="fw-bold text-success me-2"),
        html.Span(f"({df.shape[0]} rows x {df.shape[1]} cols)", className="badge bg-secondary")
    ])

    # Calculate Quick Stats
    missing_count = int(df.isnull().sum().sum())
    memory_use = df.memory_usage(deep=True).sum() / 1024 # KB
    
    stats_grid = dbc.Row([
        dbc.Col([html.H4(f"{df.shape[0]}", className="text-info fw-bold mb-0"), html.Small("Rows", className="text-muted")], width=6, className="text-center mb-3"),
        dbc.Col([html.H4(f"{df.shape[1]}", className="text-info fw-bold mb-0"), html.Small("Columns", className="text-muted")], width=6, className="text-center mb-3"),
        dbc.Col([html.H4(f"{missing_count}", className="text-warning mb-0"), html.Small("Missing Vals", className="text-muted")], width=6, className="text-center"),
        dbc.Col([html.H4(f"{memory_use:.1f} KB", className="text-success mb-0", style={"fontSize": "1.1rem" if memory_use > 1000 else "1.2rem"}), html.Small("Memory", className="text-muted")], width=6, className="text-center")
    ], className="g-2")
    
    # Clear leftover pinned row options if any from previous cache streams
    updated_options = dict(current_options) if current_options else {}
    if "pinnedBottomRowData" in updated_options:
        del updated_options["pinnedBottomRowData"]


    return new_data, new_cols, new_data, file_info, "", stats_grid, "", updated_options


@app.callback(
    Output("dataset-description-output", "children"),
    Output("dataset-description-store", "data"),
    Input("btn-gen-desc", "n_clicks"),
    State("original-data-store", "data"),
    prevent_initial_call=True
)
def fetch_description_metadata(n_clicks, orig_data):
    if not n_clicks or not orig_data: return dash.no_update, dash.no_update
    import pandas as pd
    from services.llm_client import get_dataset_description
    
    df = pd.DataFrame(orig_data)
    df = df[[c for c in df.columns if c not in ['ai_audit', '_row_id', 'is_context', 'is_target', 'is_ai_']]]
    
    sample_df = df.sample(min(15, len(df)))
    description = get_dataset_description(sample_df)
    

    return dcc.Markdown(description), description



# Deprecated toggle_meta_frame removed because dbc.Accordion handles toggle natively.








@app.callback(
    Output('data-grid', 'rowData', allow_duplicate=True),
    Input('btn-add-row', 'n_clicks'),
    State('data-grid', 'rowData'),
    State('data-grid', 'columnDefs'),
    prevent_initial_call=True
)
def add_new_row(n_clicks, current_rows, current_cols):
    # Standard n_clicks check
    if not n_clicks: return dash.no_update
    new_row = {col['field']: None for col in current_cols if col['field'] not in ['_row_id', 'is_context', 'is_target', 'ai_audit']}
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
    import random
    new_field = f"custom_{random.randint(1000, 9999)}"
    current_cols.append({"field": new_field, "headerName": "New Col", "editable": True})
    return current_cols


