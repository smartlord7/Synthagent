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
import tempfile
import shutil
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, callback_context, MATCH, ALL
from core import app
from services.report_generator import generate_benchmark_pdf
from utils.metrics import calculate_metrics
from utils.logging_config import logger
from datetime import datetime

# --- CONFIGURE PLOTLY EXPORT ---
# Chromium flags for stability on Windows systems
try:
    pio.kaleido.scope.chromium_args = ["--no-sandbox", "--disable-gpu", "--disable-software-rasterizer"]
    pio.kaleido.scope.mathjax = None # Avoids hangs
except: pass

def safe_image_export(fig, path, scale=3):
    """Robust image export that handles locking and environment errors."""
    try:
        # Use to_image bytes instead of write_image to avoid internal file-lock conflicts
        img_bytes = pio.to_image(fig, format="png", scale=scale)
        with open(path, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        logger.warning(f"Static export failed for {path}: {e}")
        return False

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

    # Filter internal columns for clean PDF appendix & metrics
    internal_cols = ['_fold_id', '_row_id', 'is_context', 'is_target', 'ai_audit']
    df_aug = df_aug.drop(columns=[c for c in df_aug.columns if c.startswith('is_ai_') or c in internal_cols], errors='ignore')

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
    
    pdf_bytes = generate_benchmark_pdf(report_data)
    return dcc.send_bytes(bytes(pdf_bytes), f"TabuLLM_Audit_{dataset_id}.pdf")


# --- TWO-STAGE REPORT GENERATION ---

@app.callback(
    Output("report-settings-modal", "is_open"),
    Output("report-dataset-list", "children"),
    Output("report-opt-plots", "value"),
    Output("report-opt-audit", "value"),
    Output("report-opt-appendix", "value"),
    Output("report-opt-counter", "value"),
    Input("btn-global-report", "n_clicks"),
    Input("btn-report-cancel", "n_clicks"),
    State("datasets-metadata-store", "data"),
    State("report-config-store", "data"),
    prevent_initial_call=True
)
def toggle_report_modal(n_open, n_cancel, meta_store, config_store):
    ctx = dash.callback_context
    if not ctx.triggered: return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    trig_id = ctx.triggered_id
    
    if trig_id == "btn-report-cancel":
        return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    if trig_id == "btn-global-report":
        # Dynamic dataset switches
        dataset_switches = []
        selected_ids = config_store.get("selected_datasets", [])
        
        for d_id, d_meta in meta_store.items():
            if d_id == "default-dataset": continue
            # Default to checked if new or previously selected
            is_checked = d_id in selected_ids if selected_ids else True
            
            dataset_switches.append(html.Div([
                dbc.Switch(
                    id={"type": "report-ds-switch", "index": d_id},
                    label=f"{d_meta.get('filename')} ({d_meta.get('rows')} rows)",
                    value=is_checked,
                    className="small text-info fw-bold"
                )
            ], className="mb-2 p-2 border border-secondary rounded bg-dark-subtle"))
            
        if not dataset_switches:
            dataset_switches = html.Div("No benchmarked datasets available.", className="text-muted small italic")
            
        return True, dataset_switches, \
               config_store.get("include_plots", True), \
               config_store.get("include_audit", True), \
               config_store.get("include_appendix", True), \
               config_store.get("include_counter", True)

    return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("report-config-store", "data"),
    Input("report-opt-plots", "value"),
    Input("report-opt-audit", "value"),
    Input("report-opt-appendix", "value"),
    Input("report-opt-counter", "value"),
    Input({"type": "report-ds-switch", "index": ALL}, "value"),
    State({"type": "report-ds-switch", "index": ALL}, "id"),
    State("report-config-store", "data"),
    prevent_initial_call=True
)
def sync_report_settings(plots, audit, appendix, counter, ds_values, ds_ids, current_store):
    selected_ds = [d_id["index"] for d_id, val in zip(ds_ids, ds_values) if val]
    return {
        "include_plots": plots,
        "include_audit": audit,
        "include_appendix": appendix,
        "include_counter": counter,
        "selected_datasets": selected_ds
    }

@app.callback(
    Output("global-download-report-pdf", "data"),
    Input("btn-report-confirm", "n_clicks"),
    State("report-config-store", "data"),
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
def execute_global_report_pm(n_clicks, config_store, meta_store, pipeline_results, all_orig, all_aug_rows, 
                              all_desc, all_algo, all_tone, all_consistency, 
                              all_temp, all_chunk, all_max_ctx, all_strategy, all_targets, all_ids):
    if not n_clicks or not meta_store: return dash.no_update
    
    import numpy as np
    from scipy import stats
    from utils.metrics import calculate_metrics
    
    selected_datasets = config_store.get("selected_datasets", [])
    if not selected_datasets: return dash.no_update
    
    # Use a system temporary directory to avoid triggering Dash's auto-reloader
    with tempfile.TemporaryDirectory() as temp_dir:
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
            target_vars = all_targets[idx] or []
            
            df_orig = pd.DataFrame(all_orig[idx]) if all_orig[idx] else pd.DataFrame()
            df_aug = pd.DataFrame(all_aug_rows[idx]) if all_aug_rows[idx] else pd.DataFrame()
            
            # Filter internal metadata for clean reporting
            internal_cols = ['_fold_id', '_row_id', 'is_context', 'is_target', 'ai_audit']
            df_aug = df_aug.drop(columns=[c for c in df_aug.columns if c.startswith('is_ai_') or c in internal_cols], errors='ignore')
            
            df_orig_enc = df_orig.copy()
            df_aug_enc = df_aug.copy()
            num_cols_t = [c for c in df_orig.select_dtypes(include='number').columns if c not in ['_row_id', 'is_context', 'is_target', 'ai_audit']]
            cat_cols_t = [c for c in df_orig.columns if c not in num_cols_t and c not in ['_row_id', 'is_context', 'is_target', 'ai_audit']]
            for col in cat_cols_t:
                if col in df_aug_enc.columns:
                    combined = pd.concat([df_orig_enc[col], df_aug_enc[col]])
                    codes, _ = pd.factorize(combined)
                    df_orig_enc[col] = codes[:len(df_orig)]
                    df_aug_enc[col] = codes[len(df_orig):]
            
            if pipeline_results and d_id in pipeline_results:
                results_list = pipeline_results[d_id].get("results", [])
                # --- AGGREGATE METRICS (Mean ± Std) ---
                agg_metrics = {}
                
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
                inference_data = [] # [Feature, Orig Mean/Mode, Aug Mean ± Std, Shift ± Std, p-value ± Std, Status]
                num_cols = [c for c in df_orig.select_dtypes(include='number').columns if c not in ['_row_id', 'is_context', 'is_target', 'ai_audit']]
                cat_cols = [c for c in df_orig.columns if c not in num_cols and c not in ['_row_id', 'is_context', 'is_target', 'ai_audit']]
                all_features = num_cols + cat_cols
                
                # Encode for universal feature coverage (numeric + categorical)
                # Already precomputed before the pipeline block.
                
                # Map features to their fold-level stats
                feature_stats = {col: {"shifts": [], "ps": [], "aug_vals": [], "enc_vals": []} for col in all_features}
                
                for r in results_list:
                    fold_df = pd.DataFrame(r.get("synthetic_data", []))
                    if fold_df.empty: continue
                    # Factorize this fold too? No, use a global mapping for consistency
                    for col in all_features:
                        if col not in df_orig.columns or col not in fold_df.columns: continue
                        a = df_orig[col].dropna()
                        b = fold_df[col].dropna()
                        if a.empty or b.empty: continue
                        
                        if col in num_cols:
                            m_a = a.mean()
                            m_b = b.mean()
                            shift = ((m_b - m_a) / (abs(m_a) + 1e-9)) * 100
                            _, p_val = stats.ks_2samp(a, b)
                            feature_stats[col]["aug_vals"].append(m_b)
                        else:
                            # Categorical drift (Hellinger proxy)
                            p_o = a.astype(str).value_counts(normalize=True)
                            p_a = b.astype(str).value_counts(normalize=True)
                            cats = p_o.index.union(p_a.index)
                            p_h = p_o.reindex(cats, fill_value=0).values
                            q_h = p_a.reindex(cats, fill_value=0).values
                            shift = (np.sqrt(np.sum((np.sqrt(p_h) - np.sqrt(q_h))**2)) / np.sqrt(2)) * 100
                            
                            # Chi-square p-value
                            p_exp = p_h * len(b)
                            p_exp = np.where(p_exp == 0, 1e-5, p_exp)
                            try:
                                _, p_val = stats.chisquare(q_h * len(b), f_exp=p_exp)
                            except: p_val = 0.0
                            
                            # Use global encoding for this fold
                            combined_local = pd.concat([df_orig[col], fold_df[col]])
                            codes_local, _ = pd.factorize(combined_local)
                            feature_stats[col]["enc_vals"].append(codes_local[len(df_orig):].mean())
                            feature_stats[col]["aug_vals"].append(0)
    
                        feature_stats[col]["shifts"].append(shift)
                        feature_stats[col]["ps"].append(p_val)
                
                for col, stats_obj in feature_stats.items():
                    if stats_obj["ps"]:
                        m_shift, s_shift = np.mean(stats_obj["shifts"]), np.std(stats_obj["shifts"])
                        m_p, s_p = np.mean(stats_obj["ps"]), np.std(stats_obj["ps"])
                        
                        status = "Pass" if m_p > 0.05 else "Drift"
                        if col in num_cols:
                            m_aug, s_aug = np.mean(stats_obj["aug_vals"]), np.std(stats_obj["aug_vals"])
                            orig_disp = f"{df_orig[col].mean():.2f}"
                            aug_disp = f"{m_aug:.2f} ± {s_aug:.2f}"
                        else:
                            # Categorical display: Mode (Encoded Mean)
                            e_orig = df_orig_enc[col].mean()
                            orig_disp = f"'{df_orig[col].mode()[0]}' ({e_orig:.1f})" if not df_orig[col].mode().empty else "N/A"
                            e_m, e_s = np.mean(stats_obj["enc_vals"]), np.std(stats_obj["enc_vals"])
                            aug_disp = f"'{df_aug[col].mode()[0]}' ({e_m:.1f} ±{e_s:.1f})" if not df_aug[col].mode().empty else "N/A"
                            
                        inference_data.append([
                            col, 
                            orig_disp, 
                            aug_disp,
                            f"{m_shift:+.1f} ± {s_shift:.1f}%", 
                            f"{m_p:.4f} ± {s_p:.2f}", 
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
    
            # Appendix: Multi-Fold Trace (Color-Coded combined Context + Synthetic)
            appendix_folds = []
            if pipeline_results and d_id in pipeline_results:
                results_list = pipeline_results[d_id].get("results", [])
                for i, r in enumerate(results_list):
                    ctx_raw = r.get("context_data")
                    syn_raw = r.get("synthetic_data")
                    df_ctx = pd.DataFrame(ctx_raw) if ctx_raw is not None else pd.DataFrame()
                    df_syn = pd.DataFrame(syn_raw) if syn_raw is not None else pd.DataFrame()
                    
                    if not df_ctx.empty or not df_syn.empty:
                        # Combine them
                        combined_fold = pd.concat([df_ctx, df_syn], ignore_index=True)
                        
                        # Generate row colors
                        colors = []
                        for _ in range(len(df_ctx)): colors.append((235, 245, 255)) # Light Blue for Context
                        for _ in range(len(df_syn)): colors.append((240, 255, 245)) # Light Green for Synthetic
                        
                        appendix_folds.append({
                            "name": f"Fold {i+1}",
                            "sample": combined_fold,
                            "row_colors": colors,
                            "ctx_ratio": r.get("ctx_ratio")
                        })
    
                # --- VIZ EXPORT (PCA & CORR) ---
                plot_paths = []
    
                if len(all_features) >= 2:
                    try:
                        import plotly.io as pio
                        # temp_dir is provided by the outer context manager
                        
                        # 1. Original Correlation Heatmap
                        c_orig_df = df_orig_enc[all_features].corr().fillna(0)
                        
                        # Adaptive sizing for many columns
                        dim = max(800, len(all_features) * 30)
                        font_size = max(6, min(12, 1000 // len(all_features)))
                        
                        fig_corr_orig = px.imshow(c_orig_df, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                                 template="plotly_dark", title="Original Relational Structure")
                        fig_corr_orig.update_layout(width=dim, height=dim, margin=dict(l=100, r=50, t=100, b=100))
                        fig_corr_orig.update_xaxes(tickfont=dict(size=font_size))
                        fig_corr_orig.update_yaxes(tickfont=dict(size=font_size))
                        
                        orig_corr_path = os.path.join(temp_dir, f"corr_orig_{d_id}.png")
                        if safe_image_export(fig_corr_orig, orig_corr_path):
                            plot_paths.append({"title": "Relational Baseline (Original)", "path": orig_corr_path})
    
                        # 2. Augmented Correlation Heatmap (Average of folds)
                        if pipeline_results and d_id in pipeline_results:
                            folds = pipeline_results[d_id].get("results", [])
                            fold_corrs = []
                            for f in folds:
                                f_df = pd.DataFrame(f.get("synthetic_data", []))
                                if not f_df.empty:
                                    # Simple encoding for fold corr
                                    f_df_enc = f_df.copy()
                                    for col in cat_cols:
                                        # Use same codes as main if possible, but factorize is local per fold here for simplicity
                                        f_df_enc[col] = pd.factorize(f_df_enc[col])[0]
                                    fold_corrs.append(f_df_enc[all_features].corr().fillna(0))
                            if fold_corrs:
                                c_aug_df = sum(fold_corrs) / len(fold_corrs)
                            else:
                                c_aug_df = df_aug_enc[all_features].corr().fillna(0)
                        else:
                            c_aug_df = df_aug_enc[all_features].corr().fillna(0)
    
                        fig_corr_aug = px.imshow(c_aug_df, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                                template="plotly_dark", title="Augmented Relational Stability (Averaged)")
                        fig_corr_aug.update_layout(width=dim, height=dim, margin=dict(l=100, r=50, t=100, b=100))
                        fig_corr_aug.update_xaxes(tickfont=dict(size=font_size))
                        fig_corr_aug.update_yaxes(tickfont=dict(size=font_size))
                        
                        aug_corr_path = os.path.join(temp_dir, f"corr_aug_{d_id}.png")
                        if safe_image_export(fig_corr_aug, aug_corr_path):
                            plot_paths.append({"title": "Relational Stability (Averaged Augmented)", "path": aug_corr_path})
                        
                        # 2. PCA Scatter (Encapsulating all features via encoding)
                        pca = PCA(n_components=2)
                        scaler_pca = MinMaxScaler()
                        combined_pca = pd.concat([df_orig_enc[all_features], df_aug_enc[all_features]]).fillna(0)
                        scaler_pca.fit(combined_pca)
                        o_norm = scaler_pca.transform(df_orig_enc[all_features].fillna(0))
                        a_norm = scaler_pca.transform(df_aug_enc[all_features].fillna(0))
                        
                        z_orig = pca.fit_transform(o_norm)
                        z_aug = pca.transform(a_norm)
                        df_pca = pd.concat([
                            pd.DataFrame(z_orig, columns=['PC1', 'PC2']).assign(State='Original'),
                            pd.DataFrame(z_aug, columns=['PC1', 'PC2']).assign(State='Augmented')
                        ])
                        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='State', 
                                            template="plotly_dark", opacity=0.6, title="Geometric Space Coverage (PCA)")
                        fig_pca.update_layout(width=1000, height=800)
                        pca_path = os.path.join(temp_dir, f"pca_{d_id}.png")
                        if safe_image_export(fig_pca, pca_path):
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
    
        pdf_bytes = generate_benchmark_pdf(report_data)
        # temp_dir is automatically cleaned up here by the TemporaryDirectory context manager

    return dcc.send_bytes(bytes(pdf_bytes), f"TabuLLM_Audit_{datetime.now().strftime('%Y%m%d')}.pdf")
