import uuid
import dash
from dash import Input, Output, State
from core import app, logger
from services.data_manager import parse_uploaded_file, df_to_aggrid_format
from utils.constants import ai_highlight_style

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
