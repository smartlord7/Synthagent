import dash
import pandas as pd
from dash import Input, Output, State, dcc, MATCH
from core import app

# --- EXPORT LOGIC ---
@app.callback(
    Output({"type": "download-csv", "index": MATCH}, "data"),
    Input({"type": "btn-export", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "data-grid", "index": MATCH}, "columnDefs"),
    prevent_initial_call=True
)
def export_augmented_data_pm(n_clicks, row_data, col_defs):
    if not n_clicks or not row_data: return dash.no_update

    field_to_header = {col['field']: col.get('headerName', col['field']) for col in col_defs}
    df = pd.DataFrame(row_data)
    cols_to_exclude = [c for c in df.columns if c.startswith('is_ai_') or c in ['_row_id', 'is_context', 'is_target', 'ai_audit']]
    df = df.drop(columns=cols_to_exclude, errors='ignore')

    df = df[[k for k in field_to_header.keys() if k in df.columns]]
    df = df.rename(columns=field_to_header)

    return dcc.send_data_frame(df.to_csv, "AI_Augmented_Dataset.csv", index=False)


@app.callback(
    Output({"type": "download-pdf", "index": MATCH}, "data"),
    Input({"type": "btn-export-pdf", "index": MATCH}, "n_clicks"),
    State({"type": "original-data-store", "index": MATCH}, "data"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "dataset-description-store", "index": MATCH}, "data"),
    State({"type": "algo-selector", "index": MATCH}, "value"),
    State({"type": "param-tone", "index": MATCH}, "value"),
    State({"type": "param-consistency", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def export_augmented_pdf_pm(n_clicks, orig_data, row_data, description, algo, tone, consistency):
    if not n_clicks or not orig_data or not row_data: return dash.no_update
    
    from utils.metrics import calculate_metrics
    from services.report_generator import generate_pdf_report

    df_orig = pd.DataFrame(orig_data)
    df_aug = pd.DataFrame(row_data)

    cols_to_exclude = ['ai_audit', '_row_id', 'is_context', 'is_target', 'is_ai_']
    df_orig = df_orig[[c for c in df_orig.columns if c not in cols_to_exclude]]
    df_aug = df_aug[[c for c in df_aug.columns if c not in cols_to_exclude]]

    stats_vals = {
        "Rows": df_orig.shape[0],
        "Cols": df_orig.shape[1],
        "Missing": int(df_orig.isnull().sum().sum()),
        "Memory": f"{df_orig.memory_usage(deep=True).sum()/1024:.1f} KB"
    }

    params_dict = {
        "algorithm": algo,
        "tone": tone,
        "consistency": consistency
    }

    try:
        metrics_dict = calculate_metrics(df_orig, df_aug)
    except Exception:
        metrics_dict = {"Error": "Metric calculation failed"}

    pdf_bytes = generate_pdf_report("TabuLLM_Report", stats_vals, description, params_dict, metrics_dict, df_orig, df_aug)
    return dcc.send_bytes(bytes(pdf_bytes), "TabuLLM_Report.pdf")
