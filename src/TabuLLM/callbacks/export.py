import dash
import pandas as pd
from dash import Input, Output, State, dcc
from core import app

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
