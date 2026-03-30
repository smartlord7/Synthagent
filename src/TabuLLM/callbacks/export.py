import json
import io
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import Input, Output, State, dcc, html, callback_context, MATCH, ALL
from core import app
from services.report_generator import generate_comprehensive_pdf
from utils.metrics import calculate_metrics
from utils.logging_config import logger

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
    State({"index": MATCH, "type": "column-roles-store"}, "data"),
    prevent_initial_call=True
)
def export_augmented_pdf_pm(n_clicks, orig_data, row_data, description, algo, tone, consistency, roles):
    if not n_clicks or not orig_data or not row_data: return dash.no_update
    
    from utils.metrics import calculate_metrics

    df_orig = pd.DataFrame(orig_data)
    df_aug = pd.DataFrame(row_data)

    # Prepare single dataset report in the NEW format expected by generator
    dataset_id = dash.callback_context.triggered_id['index']
    
    report_data = {
        "executive_summary": {"avg_fidelity": "Calculating...", "status": "Single Dataset Export"},
        "datasets": {
            dataset_id: {
                "filename": f"Dataset_{dataset_id}",
                "description": description or "No AI description.",
                "params": {
                    "generation": {"algorithm": algo, "tone": tone, "consistency": consistency},
                    "model": {} # No pipeline context here
                },
                "metrics": calculate_metrics(df_orig, df_aug) if not df_orig.empty else {},
                "plots": [],
                "samples": {"orig": df_orig, "aug": df_aug}
            }
        }
    }
    
    pdf_bytes = generate_comprehensive_pdf(report_data)
    return dcc.send_bytes(bytes(pdf_bytes), f"TabuLLM_Audit_{dataset_id}.pdf")


@app.callback(
    Output("global-download-pdf", "data"),
    Input("btn-global-report", "n_clicks"),
    State("datasets-metadata-store", "data"),
    State("pipeline-results-store", "data"),
    State({"type": "original-data-store", "index": ALL}, "data"),
    State({"type": "data-grid", "index": ALL}, "rowData"),
    State({"type": "dataset-description-store", "index": ALL}, "data"),
    State({"type": "algo-selector", "index": ALL}, "value"),
    State({"type": "param-tone", "index": ALL}, "value"),
    State({"type": "param-consistency", "index": ALL}, "value"),
    State({"type": "temperature-slider", "index": ALL}, "value"),
    State({"type": "gen-chunk-size", "index": ALL}, "value"),
    State({"type": "gen-max-context", "index": ALL}, "value"),
    State({"type": "gen-context-strategy", "index": ALL}, "value"),
    State({"type": "target-cols-dropdown", "index": ALL}, "value"),
    State({"type": "original-data-store", "index": ALL}, "id"),
    prevent_initial_call=True
)
def generate_global_report_pm(n_clicks, meta_store, pipeline_results, all_orig, all_aug_rows, 
                              all_desc, all_algo, all_tone, all_consistency, 
                              all_temp, all_chunk, all_max_ctx, all_strategy, all_targets, all_ids):
    if not n_clicks or not meta_store: return dash.no_update
    
    import numpy as np
    from scipy import stats
    from utils.metrics import calculate_metrics
    
    # 1. Map IDs to lists
    id_map = {item['index']: i for i, item in enumerate(all_ids)}
    
    report_data = {
        "executive_summary": {
            "avg_fidelity": "N/A",
            "global_metrics": {}
        },
        "datasets": {}
    }
    
    all_dataset_metrics = [] # List of dicts
    
    all_fidelities = []
    
    for d_id, d_meta in meta_store.items():
        # 1. Skip Workbench
        if d_id == "default-dataset": continue
        if d_id not in id_map: continue
        idx = id_map[d_id]
        
        df_orig = pd.DataFrame(all_orig[idx]) if all_orig[idx] else pd.DataFrame()
        df_aug = pd.DataFrame(all_aug_rows[idx]) if all_aug_rows[idx] else pd.DataFrame()
        
        if pipeline_results and d_id in pipeline_results:
            results_list = pipeline_results[d_id].get("results", [])
            # --- AGGREGATE METRICS (Mean ± Std) ---
            agg_metrics = {}
            target_vars = all_targets[idx] or []
            
            metric_keys = [
                "Synthetic Fidelity (JSD)", "Correlation Stability", "Distribution Consistency", 
                "Entropy Preservation", "Mean Z Score Drift", "Categorical Fidelity", 
                "Privacy Risk", "Data Density", "Coverage Score", "Feature Accuracy", "Novelty Score"
            ]
            for k in metric_keys:
                vals = [r.get(k, 0) for r in results_list if k in r]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    # For metrics already formatted as strings in overall_ml, we aggregate here manually for reporting
                    agg_metrics[k] = f"{mean_val:.2f} ± {std_val:.2f}"
                    if k == 'Synthetic Fidelity (JSD)':
                        all_fidelities.append(mean_val)
            
            # --- ML PERFORMANCE ---
            ml_perf = []
            for r in results_list:
                fold_ml = r.get("ml_eval", {})
                for m_type, m_data in fold_ml.items():
                    # We'll aggregate these in the generator or pass raw folds
                    pass
            
            # --- STATISTICAL INFERENCE TABLE (Aggregated Across Folds) ---
            inference_data = [] # [Feature, Orig Mean, Aug Mean ± Std, Shift ± Std, KS ± Std, Status]
            num_cols = df_orig.select_dtypes(include='number').columns
            
            # Map features to their fold-level stats
            feature_stats = {col: {"shifts": [], "ks_ps": [], "aug_means": []} for col in num_cols}
            
            for r in results_list:
                fold_df = pd.DataFrame(r.get("synthetic_data", []))
                if fold_df.empty: continue
                for col in num_cols:
                    if col in fold_df.columns:
                        a = df_orig[col].dropna()
                        b = fold_df[col].dropna()
                        if not a.empty and not b.empty:
                            m_a = a.mean()
                            m_b = b.mean()
                            shift = ((m_b - m_a) / (abs(m_a) + 1e-9)) * 100
                            _, ks_p = stats.ks_2samp(a, b)
                            feature_stats[col]["shifts"].append(shift)
                            feature_stats[col]["ks_ps"].append(ks_p)
                            feature_stats[col]["aug_means"].append(m_b)
            
            for col, stats_obj in feature_stats.items():
                if stats_obj["ks_ps"]:
                    m_shift, s_shift = np.mean(stats_obj["shifts"]), np.std(stats_obj["shifts"])
                    m_ks, s_ks = np.mean(stats_obj["ks_ps"]), np.std(stats_obj["ks_ps"])
                    m_aug, s_aug = np.mean(stats_obj["aug_means"]), np.std(stats_obj["aug_means"])
                    
                    status = "Pass" if m_ks > 0.05 else "Drift"
                    inference_data.append([
                        col, 
                        f"{df_orig[col].mean():.2f}", 
                        f"{m_aug:.2f} ± {s_aug:.2f}",
                        f"{m_shift:+.1f} ± {s_shift:.1f}%", 
                        f"{m_ks:.4f} ± {s_ks:.2f}", 
                        status
                    ])
        else:
            # Fallback for manual generation without pipeline
            m = calculate_metrics(df_orig, df_aug) if not df_orig.empty else {}
            agg_metrics = {k.replace("_", " ").title(): f"{v:.1f}" for k, v in m.items()}
            inference_data = []

        # Metadata for Dataset Portrait
        metadata_portrait = {
            "Rows": len(df_orig),
            "Cols": len(df_orig.columns),
            "Missing": int(df_orig.isnull().sum().sum()),
            "Memory": f"{df_orig.memory_usage(deep=True).sum()/1024:.1f} KB",
            "Target Variables": ", ".join(target_vars) if target_vars else "None detected"
        }

        # Appendix: Multi-Fold Trace
        appendix_folds = []
        if pipeline_results and d_id in pipeline_results:
            results_list = pipeline_results[d_id].get("results", [])
            for i, r in enumerate(results_list):
                fold_df = pd.DataFrame(r.get("synthetic_data", []))
                if not fold_df.empty:
                    appendix_folds.append({
                        "name": f"Fold {i+1}",
                        "sample": fold_df # Full dataframe for the appendix
                    })

            # --- VIZ EXPORT (PCA & CORR) ---
            plot_paths = []
            if len(num_cols) >= 2:
                try:
                    import plotly.io as pio
                    temp_dir = os.path.join(os.getcwd(), "tmp_plots")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 1. Original Correlation Heatmap
                    c_orig_df = df_orig[num_cols].corr().fillna(0)
                    fig_corr_orig = px.imshow(c_orig_df, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                             template="plotly_dark", title="Original Relational Structure")
                    orig_corr_path = os.path.join(temp_dir, f"corr_orig_{d_id}.png")
                    fig_corr_orig.write_image(orig_corr_path)
                    plot_paths.append({"title": "Relational Baseline (Original)", "path": orig_corr_path})

                    # 2. Augmented Correlation Heatmap (Average of folds)
                    if pipeline_results and d_id in pipeline_results:
                        folds = pipeline_results[d_id].get("results", [])
                        fold_corrs = []
                        for f in folds:
                            f_df = pd.DataFrame(f.get("synthetic_data", []))
                            if not f_df.empty:
                                fold_corrs.append(f_df[num_cols].corr().fillna(0))
                        if fold_corrs:
                            c_aug_df = sum(fold_corrs) / len(fold_corrs)
                        else:
                            c_aug_df = df_aug[num_cols].corr().fillna(0)
                    else:
                        c_aug_df = df_aug[num_cols].corr().fillna(0)

                    fig_corr_aug = px.imshow(c_aug_df, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                            template="plotly_dark", title="Augmented Relational Stability (Averaged)")
                    aug_corr_path = os.path.join(temp_dir, f"corr_aug_{d_id}.png")
                    fig_corr_aug.write_image(aug_corr_path)
                    plot_paths.append({"title": "Relational Stability (Averaged Augmented)", "path": aug_corr_path})
                    
                    # 2. PCA Scatter
                    pca = PCA(n_components=2)
                    scaler = MinMaxScaler()
                    df_full = pd.concat([df_orig[num_cols], df_aug[num_cols]]).fillna(0)
                    scaler.fit(df_full)
                    z_orig = pca.fit_transform(scaler.transform(df_orig[num_cols].fillna(0)))
                    z_aug = pca.transform(scaler.transform(df_aug[num_cols].fillna(0)))
                    df_pca = pd.concat([
                        pd.DataFrame(z_orig, columns=['PC1', 'PC2']).assign(State='Original'),
                        pd.DataFrame(z_aug, columns=['PC1', 'PC2']).assign(State='Augmented')
                    ])
                    fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='State', 
                                        template="plotly_dark", opacity=0.6, title="Geometric Space Coverage (PCA)")
                    pca_path = os.path.join(temp_dir, f"pca_{d_id}.png")
                    fig_pca.write_image(pca_path)
                    plot_paths.append({"title": "PCA Geometric Coverage", "path": pca_path})
                except Exception as ve:
                    logger.warning(f"Failed to export viz for {d_id}: {ve}")

            report_data["datasets"][d_id] = {
                "filename": d_meta.get("filename", d_id),
                "description": all_desc[idx] or "No AI description generated.",
                "metadata": metadata_portrait,
                "params": {
                    "generation": {
                        "Algorithm": all_algo[idx],
                        "Style": all_tone[idx],
                        "Consistency": all_consistency[idx],
                        "Temperature": all_temp[idx],
                        "Chunk Size": all_chunk[idx],
                        "Max Context": all_max_ctx[idx],
                        "Strategy": all_strategy[idx]
                    },
                    "model": pipeline_results.get(d_id, {}).get("params", {}).get("model", {}) if pipeline_results else {}
                },
                "metrics": agg_metrics,
                "inference_table": inference_data,
                "ml_results": pipeline_results.get(d_id, {}).get("overall_ml", {}),
                "plots": plot_paths,
                "samples": {"orig": df_orig, "aug": df_aug},
                "appendix": appendix_folds
            }

    # --- GLOBAL AGGREGATION FOR EXECUTIVE SUMMARY ---
    global_agg = {}
    metric_keys = [
        "Synthetic Fidelity (JSD)", "Correlation Stability", "Distribution Consistency", 
        "Entropy Preservation", "Mean Z Score Drift", "Categorical Fidelity", 
        "Privacy Risk", "Data Density", "Coverage Score", "Feature Accuracy", "Novelty Score"
    ]
    
    for k in metric_keys:
        all_vals = []
        for d_id, d_info in report_data["datasets"].items():
            if not isinstance(d_info, dict): continue
            m_dict = d_info.get("metrics", {})
            m_str = m_dict.get(k, "0")
            try:
                # Extract mean from "mean ± std" or just "val"
                val_part = m_str.split('±')[0].strip().replace('%', '')
                all_vals.append(float(val_part))
            except: pass
        if all_vals:
            m_val = np.mean(all_vals)
            s_val = np.std(all_vals)
            suffix = '%' if k not in ["Mean Z Score Drift", "Novelty Score"] else ""
            global_agg[k] = f"{m_val:.1f}{suffix} ± {s_val:.1f}"
    
    report_data["executive_summary"]["global_metrics"] = global_agg
    if "Synthetic Fidelity (JSD)" in global_agg:
        report_data["executive_summary"]["avg_fidelity"] = global_agg["Synthetic Fidelity (JSD)"]

    pdf_bytes = generate_comprehensive_pdf(report_data)
    return dcc.send_bytes(bytes(pdf_bytes), "TabuLLM_Comprehensive_Audit.pdf")
