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


@app.callback(
    Output("download-pdf-report", "data"),
    Input("btn-export-pdf", "n_clicks"),
    State("original-data-store", "data"),
    State("data-grid", "rowData"),
    State("dataset-description-store", "data"),
    State("algo-selector", "value"),
    State("param-tone", "value"),
    State("param-consistency", "value"),
    prevent_initial_call=True
)
def export_augmented_pdf(n_clicks, orig_data, row_data, description, algo, tone, consistency):
    if not n_clicks or not orig_data or not row_data: return dash.no_update
    
    from utils.metrics import calculate_metrics
    from services.report_generator import generate_pdf_report

    # Clean up Augmented row_data (which includes tracking fields)
    df_orig = pd.DataFrame(orig_data)
    df_aug = pd.DataFrame(row_data)

    cols_to_exclude = ['ai_audit', '_row_id', 'is_context', 'is_target', 'is_ai_']
    df_orig = df_orig[[c for c in df_orig.columns if c not in cols_to_exclude]]
    df_aug = df_aug[[c for c in df_aug.columns if c not in cols_to_exclude]]

    # 1. Compile Stats Dictionary
    missing_count = int(df_orig.isnull().sum().sum())
    stats_vals = {
        "Rows": df_orig.shape[0],
        "Cols": df_orig.shape[1],
        "Missing": missing_count,
        "Memory": f"{df_orig.memory_usage(deep=True).sum()/1024:.1f} KB"
    }

    # 2. Compile Parameters Dictionary
    params_dict = {
        "algorithm": algo,
        "tone": tone,
        "consistency": consistency
    }

    # 3. Calculate Metrics
    try:
        metrics_dict = calculate_metrics(df_orig, df_aug)
    except Exception as e:
        metrics_dict = {"Error": str(e)}

    # 4. Generate PDF bytes
    pdf_bytes = generate_pdf_report("TabuLLM_Augmented_Report", stats_vals, description, params_dict, metrics_dict, df_orig, df_aug)

    # Wrap in stream buffer correctly
    return dcc.send_bytes(bytes(pdf_bytes), "TabuLLM_Report.pdf")



