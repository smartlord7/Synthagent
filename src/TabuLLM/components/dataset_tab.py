import uuid
from dash import dcc, html
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from utils.constants import initial_data, initial_cols, row_class_rules, ai_highlight_style
from services.data_manager import df_to_aggrid_format
import pandas as pd

def create_dataset_tab(dataset_id, filename, df=None, saved_state=None, pipeline_res=None):
    saved_state = saved_state or {}
    
    eda_content = dbc.Alert("Click 'Refresh Analytics' to compute distribution changes and audit reports.", color="secondary", className="small")
    ml_content = dbc.Alert("Run pipeline to view machine learning benchmarks.", color="secondary", className="small")
    
    if saved_state and pipeline_res and pipeline_res.get("results"):
        try:
            from callbacks.eda import get_eda_layout
            df_orig = pd.DataFrame(saved_state.get("original-data-store", []))
            df_aug = pd.DataFrame(saved_state.get("data-grid", []))
            eda_content = get_eda_layout(df_orig, df_aug, dataset_id, results_list=pipeline_res.get("results", []))
        except Exception as e:
            print(f"Failed to restore EDA for {dataset_id}: {e}")
            
        try:
            import numpy as np
            ml_agg = {}
            for r in pipeline_res.get("results", []):
                ml_eval = r.get("ml_eval", {})
                for m_type, m_perf in ml_eval.items():
                    if not m_perf: continue
                    if m_type not in ml_agg: ml_agg[m_type] = {}
                    for m_name in ["accuracy", "f1", "rmse", "r2"]:
                        v_orig = (m_perf.get("original") or {}).get(m_name)
                        v_aug = (m_perf.get("augmented") or {}).get(m_name)
                        if v_orig is not None and v_aug is not None:
                            if m_name not in ml_agg[m_type]: ml_agg[m_type][m_name] = {"orig": [], "aug": [], "delta": []}
                            ml_agg[m_type][m_name]["orig"].append(float(v_orig))
                            ml_agg[m_type][m_name]["aug"].append(float(v_aug))
                            ml_agg[m_type][m_name]["delta"].append(float(v_aug - v_orig))
            
            def get_stats(vals):
                if not vals: return 0.0, 0.0
                m, s = float(np.mean(vals)), float(np.std(vals))
                return (m if np.isfinite(m) else 0.0), (s if np.isfinite(s) else 0.0)

            ml_cards = []
            for m_type, metrics in ml_agg.items():
                metric_rows = []
                for m_name, data in metrics.items():
                    orig_m, orig_s = get_stats(data["orig"])
                    aug_m, aug_s = get_stats(data["aug"])
                    d_m, d_s = get_stats(data["delta"])
                    diff_color = "success" if (d_m >= -0.05 if m_name != "rmse" else d_m <= 0.05) else "danger"
                    metric_rows.append(dbc.Row([
                        dbc.Col(html.Small(m_name.upper(), className="x-small text-info fw-bold"), width=3),
                        dbc.Col(html.Span(f"{orig_m:.3f}±{orig_s:.2f}", className="x-small text-muted"), width=3),
                        dbc.Col(html.Span(f"{aug_m:.3f}±{aug_s:.2f}", className="x-small text-light fw-bold"), width=3),
                        dbc.Col(html.Span(f"{d_m:+.3f}±{d_s:.2f}", className=f"text-{diff_color} x-small fw-bold"), width=3),
                    ], className="g-0 border-bottom border-secondary-subtle py-1 align-items-center"))
                
                # --- INDIVIDUAL MODEL CARD ---
                ml_cards.append(dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-cpu me-2 text-info"), 
                                html.Span(m_type.upper(), className="fw-bold small")
                            ], className="d-flex align-items-center"),
                            dbc.Button([html.I(className="bi bi-search me-1"), "Inspect"],
                                       id={"type": "btn-inspect-ml", "dataset": dataset_id, "model": m_type},
                                       color="info", size="sm", outline=True,
                                       style={"fontSize": "0.6rem", "padding": "2px 6px"},
                                       className="rounded-pill shadow-sm hover-grow")
                        ], className="d-flex align-items-center justify-content-between mb-2 border-bottom border-secondary pb-1"),
                        html.Div([
                            dbc.Row([
                                dbc.Col(html.Small("Metric", className="xx-small text-muted"), width=3), 
                                dbc.Col(html.Small("Real", className="xx-small text-muted"), width=3), 
                                dbc.Col(html.Small("Syn", className="xx-small text-muted"), width=3), 
                                dbc.Col(html.Small("Delta", className="xx-small text-muted"), width=3)
                            ], className="g-0 opacity-75 mb-1"),
                            html.Div(metric_rows)
                        ], className="flex-grow-1")
                    ], className="p-2 d-flex flex-column h-100")
                ], className="h-100 border-secondary bg-dark shadow-sm"),
                lg=3, md=6, xs=12, style={"minWidth": "250px", "maxWidth": "25%"}, className="mb-4"))

                
            if ml_cards:
                ml_content = html.Div([
                    html.H6([html.I(className="bi bi-cpu-fill me-2 text-warning"), "ML Results & Diagnostics Dashboard"], className="text-light mb-4 small fw-bold text-uppercase tracking-wider"),
                    dbc.Row(ml_cards, className="g-4 justify-content-start", style={"flexWrap": "wrap", "display": "flex"})
                ], className="mt-2 w-100")
        except Exception as e:
            print(f"Failed to restore ML Eval for {dataset_id}: {e}")

    # 1. Base Data Loading priority: 1. saved_state, 2. provided df, 3. default empty
    if "original-data-store" in saved_state:
        new_data = saved_state["original-data-store"]
        # Dummy df for stats/cols
        if new_data:
            df_stat = pd.DataFrame(new_data).drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], errors='ignore')
        else:
            df_stat = pd.DataFrame()
    elif df is not None:
        new_data = df_to_aggrid_format(df)
        for row in new_data:
            row['_row_id'] = str(uuid.uuid4())
            row['is_context'] = False
            row['is_target'] = False
            row['ai_audit'] = ""
        df_stat = df.copy()
    else:
        new_data = initial_data
        df_stat = pd.DataFrame()
    if not df_stat.empty:
        new_cols = []
        for i, col in enumerate(df_stat.columns):
            col_def = {"field": col, "headerName": col, "cellStyle": ai_highlight_style, "editable": True}
            if i == 0:
                col_def["checkboxSelection"] = True
                col_def["headerCheckboxSelection"] = True
            new_cols.append(col_def)

        # `ai_audit` column removed per user request        
        missing_count = int(df_stat.isnull().sum().sum())
        memory_use = df_stat.memory_usage(deep=True).sum() / 1024 # KB
        
        stats_grid = dbc.Row([
            dbc.Col([html.H4(f"{df_stat.shape[0]}", className="text-info fw-bold mb-0"), html.Small("Rows", className="text-muted")], width=6, className="text-center mb-3"),
            dbc.Col([html.H4(f"{df_stat.shape[1]}", className="text-info fw-bold mb-0"), html.Small("Columns", className="text-muted")], width=6, className="text-center mb-3"),
            dbc.Col([html.H4(f"{missing_count}", className="text-warning mb-0"), html.Small("Missing", className="text-muted")], width=6, className="text-center"),
            dbc.Col([html.H4(f"{memory_use:.1f} KB", className="text-success mb-0"), html.Small("Memory", className="text-muted")], width=6, className="text-center")
        ], className="g-2")
    else:
        new_cols = initial_cols
        stats_grid = dbc.Alert("Upload file to view stats.", color="secondary", className="small py-2")

    # Tab Label with Inline Delete Button (Workbench cannot be deleted)
    delete_btn = None
    if dataset_id != "default-dataset":
        delete_btn = dbc.Button(
            html.I(className="bi bi-x ms-2"),
            id={"type": "btn-delete-dataset", "index": dataset_id},
            color="link",
            size="sm",
            className="p-0 text-danger opacity-75 delete-tab-btn ms-auto",
            style={"lineHeight": "1", "fontSize": "1.1rem", "verticalAlign": "middle", "border": "none", "background": "transparent"}
        )

    tab_label = html.Span([
        html.I(className="bi bi-table me-2"),
        html.Span(filename, id={"type": "dataset-tab-filename", "index": dataset_id}, className="truncate-text", style={"maxWidth": "120px"}),
        delete_btn,
        dbc.Tooltip(filename, target={"type": "dataset-tab-filename", "index": dataset_id}, placement="top")
    ], className="d-flex align-items-center w-100")

    return dcc.Tab(
        label=tab_label,
        value=dataset_id,
        className="custom-tab",
        selected_className="custom-tab--selected",
        children=[
            html.Div([
                # Stores for tracking this dataset's state (Pattern-Matching)
                dcc.Store(id={"type": "original-data-store", "index": dataset_id}, data=new_data),
                dcc.Store(id={"type": "active-col-store", "index": dataset_id}, data=saved_state.get("active-col-store")),
                dcc.Store(id={"type": "context-row-ids", "index": dataset_id}, data=saved_state.get("context-row-ids", [])),
                dcc.Store(id={"type": "target-row-ids", "index": dataset_id}, data=saved_state.get("target-row-ids", [])),
                dcc.Store(id={"type": "selection-mode", "index": dataset_id}, data=saved_state.get("selection-mode", "target")),
                dcc.Store(id={"type": "dataset-description-store", "index": dataset_id}, data=saved_state.get("dataset-description-store", "")),
                dcc.Store(id={"type": "column-roles-store", "index": dataset_id}, data=saved_state.get("column-roles-store", {"targets": []})),
                dcc.Store(id={"type": "ml-config-store", "index": dataset_id}, data=saved_state.get("ml-config-store", {})),
                dcc.Store(id={"type": "bench-state-store", "index": dataset_id}, data=saved_state.get("bench-state-store", {"active": False, "k_total": 0, "k_current": 0, "fraction": 0.2, "results": []})),
                
                dcc.Download(id={"type": "download-csv", "index": dataset_id}),
                dcc.Download(id={"type": "download-pdf", "index": dataset_id}),

                dbc.Row([
                    # Left Widget: Dataset Metadata
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Div([
                                    html.I(className="bi bi-bar-chart-fill me-2 text-info"), "Metadata"
                                ], className="fw-bold text-light d-flex align-items-center w-100")
                            ], className="bg-dark border-secondary"),
                            dbc.CardBody([
                                html.H6([html.I(className="bi bi-grid-fill me-2 text-info"), "Quick Statistics"], className="text-muted small fw-bold mb-2 d-flex align-items-center"),
                                html.Div(id={"type": "dataset-stats-grid", "index": dataset_id}, children=stats_grid, className="mb-3"),
                                
                                html.Hr(className="border-secondary my-2"),
                                                               html.Div([
                                    html.H6([html.I(className="bi bi-cpu-fill me-2 text-info"), "Structured Description"], className="text-muted small fw-bold mb-0 d-inline-block"),
                                    html.Div([
                                        dbc.Button([html.I(className="bi bi-magic me-1"), "Insights"], id={"type": "btn-gen-desc", "index": dataset_id}, color="info", size="sm", outline=True, className="py-0 px-2 rounded-pill me-2"),
                                        dbc.Button([html.I(className="bi bi-eye")], id={"type": "btn-toggle-desc-view", "index": dataset_id}, color="secondary", size="sm", outline=True, className="py-0 px-2 rounded-pill", title="View Markdown"),
                                        dbc.Button([html.I(className="bi bi-pencil")], id={"type": "btn-toggle-desc-edit", "index": dataset_id}, color="secondary", size="sm", outline=True, className="py-0 px-2 rounded-pill ms-1", title="Edit Source"),
                                    ], className="ms-auto d-flex align-items-center")
                                ], className="d-flex align-items-center mb-2"),
                                
                                dbc.Spinner(
                                    html.Div([
                                        # View Mode (Markdown)
                                        html.Div(
                                            dcc.Markdown(
                                                id={"type": "dataset-description-markdown", "index": dataset_id},
                                                children=saved_state.get("dataset-description-store", "No description available. Click 'Insights'."),
                                                className="markdown-preview p-3 border rounded border-secondary bg-dark text-light shadow-sm",
                                                style={"minHeight": "100px", "maxHeight": "280px", "overflowY": "auto", "fontSize": "0.85rem"}
                                            ),
                                            id={"type": "dataset-description-markdown-container", "index": dataset_id},
                                            style={"display": "block"}
                                        ),
                                        # Edit Mode (Textarea)
                                        html.Div(
                                            dbc.Textarea(
                                                id={"type": "dataset-description-edit", "index": dataset_id},
                                                placeholder="Enter dataset description or click 'Insights'...",
                                                className="bg-dark text-light small",
                                                style={"height": "150px", "fontSize": "0.8rem"}
                                            ),
                                            id={"type": "dataset-description-edit-container", "index": dataset_id},
                                            style={"display": "none"}
                                        )
                                    ]),
                                    size="sm", color="info"
                                ),
                                
                                html.Hr(className="border-secondary my-2"),
                                html.Div([
                                    html.H6([html.I(className="bi bi-bullseye me-2 text-danger"), "Target Variables"], className="text-muted small fw-bold mb-0 d-inline-block"),
                                    dbc.Button([html.I(className="bi bi-magic me-1"), "Auto-Suggest"], id={"type": "btn-auto-suggest-targets", "index": dataset_id}, color="danger", size="sm", outline=True, className="ms-2 py-0 px-2 rounded-pill"),
                                ], className="d-flex align-items-center mb-1"),
                                dbc.Checkbox(
                                    id={"type": "check-single-target", "index": dataset_id},
                                    label="Force Single Target",
                                    value=True,
                                    labelClassName="text-muted x-small ms-1",
                                    className="mb-2"
                                ),
                                dcc.Dropdown(
                                    id={"type": "target-cols-dropdown", "index": dataset_id},
                                    options=[{"label": c, "value": c} for c in (df_stat.columns if df_stat is not None else [])],
                                    value=saved_state.get('target-cols-dropdown', []),
                                    multi=True, placeholder="Select Target(s)...",
                                    className="bg-dark text-light small-dropdown mb-2"
                                ),
                                html.Small("Override global target if desired.", className="text-muted x-small d-block")
                            ], className="p-3 bg-dark text-light")
                        ], className="tabu-card shadow border-secondary h-100")
                    ], md=5, className="mb-3 mb-md-0 d-flex flex-column"),

                    # Right Widget: Generation Parameters
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Div([
                                    html.I(className="bi bi-gear-fill me-2 text-success"), "Generation Parameters"
                                ], className="fw-bold text-light d-flex align-items-center")
                            ], className="bg-dark border-secondary"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Style:", className="fw-bold text-muted small"),
                                        dbc.Select(id={"type": "param-tone", "index": dataset_id},
                                                   options=[{"label": "Concise", "value": "Concise / Standard"},
                                                            {"label": "Descriptive", "value": "Descriptive"},
                                                            {"label": "Professional", "value": "Professional"}],
                                                    value=saved_state.get('param-tone', 'Concise / Standard'), size="sm")
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Consistency:", className="fw-bold text-muted small"),
                                        dbc.Select(id={"type": "param-consistency", "index": dataset_id},
                                                   options=[{"label": "Strict", "value": "Strict (Perfectly uniform)"},
                                                            {"label": "Realistic", "value": "Realistic (Natural variations)"}],
                                                    value=saved_state.get('param-consistency', 'Strict (Perfectly uniform)'), size="sm")
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Label("AI Model:", className="fw-bold text-muted small me-auto"),
                                            dbc.Button([html.I(className="bi bi-arrow-clockwise")], id={"type": "btn-refresh-models", "index": dataset_id}, color="success", size="sm", outline=True, className="py-0 px-2 rounded-pill ms-2", title="Sync with LM Studio")
                                        ], className="d-flex align-items-center mb-1"),
                                        dbc.Select(id={"type": "algo-selector", "index": dataset_id},
                                                   options=[{"label": "Gemma-3-4b", "value": "google/gemma-3-4b"},
                                                            {"label": "Llama-3.1-8b", "value": "meta/llama-3.1-8b"},
                                                            {"label": "Gemma-2-9b", "value": "google/gemma-2-9b"}],
                                                   value=saved_state.get('algo-selector', 'google/gemma-3-4b'), size="sm")
                                    ], width=4),
                                ], className="mb-2"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Temperature:", className="fw-bold text-muted small"),
                                        dcc.Slider(id={"type": "temperature-slider", "index": dataset_id}, min=0.0, max=1.0, step=0.1, value=saved_state.get('temperature-slider', 0.3),
                                                   marks={0: {'label': 'Strict', 'style': {'color': '#ffffff'}}, 1: {'label': 'Creative', 'style': {'color': '#ffffff'}}})
                                    ], width=12)
                                ], className="mb-2"),

                                html.Hr(className="border-secondary my-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Chunk Size:", className="fw-bold text-light small"),
                                        dcc.Slider(id={"type": "gen-chunk-size", "index": dataset_id}, min=10, max=100, step=10, value=saved_state.get('gen-chunk-size', 30))
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.Label("Max Context:", className="fw-bold text-light small me-auto"),
                                            html.Span("~0 Tok", id={"type": "estimated-tokens-lbl", "index": dataset_id}, className="badge bg-info ms-1")
                                        ], className="d-flex align-items-center justify-content-between"),
                                        dcc.Slider(id={"type": "gen-max-context", "index": dataset_id}, min=20, max=1000, step=20, value=saved_state.get('gen-max-context', 60))
                                    ], width=6)
                                ], className="mb-3"),

                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Context Strategy:", className="fw-bold text-light small mb-0"),
                                        dbc.Select(
                                            id={"type": "gen-context-strategy", "index": dataset_id},
                                            options=[
                                                {"label": "Random", "value": "random"},
                                                {"label": "Dense", "value": "dense"},
                                                {"label": "Diverse", "value": "diverse"}
                                            ],
                                            value=saved_state.get('gen-context-strategy', 'random'), size="sm", className="bg-dark text-light border-secondary"
                                        )
                                    ], width=12)
                                ], className="mb-2"),

                                html.Hr(className="border-secondary my-2"),
                                html.Div([
                                    html.Span([html.I(className="bi bi-funnel me-1 text-success"), "Custom Rules"], className="text-muted small fw-bold me-auto"),
                                    dbc.Button([html.I(className="bi bi-plus-circle me-1"), "Rule"], id={"type": "btn-add-param", "index": dataset_id}, size="sm", color="success", outline=True, className="py-0 px-2 rounded-pill")
                                ], className="d-flex align-items-center mb-1"),
                                html.Div(id={"type": "custom-params-container", "index": dataset_id}, children=[], style={"maxHeight": "100px", "overflowY": "auto"})
                            ], className="p-3 bg-dark text-light")
                        ], className="tabu-card shadow border-secondary h-100")
                    ], md=7)
                ], className="mb-4 g-3 mt-3"),

                # Workbench Grid
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="bi bi-pencil-square me-1 text-info"),
                                    dbc.Input(id={"type": "quick-rename-input", "index": dataset_id}, type="text", placeholder="Rename...", size="sm", className="bg-dark text-light border-secondary", style={"width": "160px", "display": "inline-block"}),
                                    dbc.Button([html.I(className="bi bi-trash")], id={"type": "btn-remove-col", "index": dataset_id}, color="danger", size="sm", outline=True, className="ms-1 py-1 px-2"),
                                    dbc.Button([html.I(className="bi bi-search")], id={"type": "btn-inspect-row", "index": dataset_id}, color="warning", size="sm", outline=True, className="ms-2 py-1 px-2"),
                                    dbc.Button([html.I(className="bi bi-plus-square")], id={"type": "btn-add-row", "index": dataset_id}, color="success", size="sm", outline=True, className="ms-2 py-1 px-2"),
                                    dbc.Button([html.I(className="bi bi-layout-three-columns")], id={"type": "btn-add-col", "index": dataset_id}, color="info", size="sm", outline=True, className="ms-1 py-1 px-2"),
                                ], className="d-flex align-items-center")
                            ], md=4, className="d-flex align-items-center"),

                            dbc.Col([
                                html.Div([
                                    dbc.ButtonGroup([
                                        dbc.Button("Context", id={"type": "btn-mode-context", "index": dataset_id}, color="info", outline=True, size="sm"),
                                        dbc.Button("Targets", id={"type": "btn-mode-target", "index": dataset_id}, color="danger", outline=True, size="sm"),
                                    ], className="me-2"),
                                    dbc.Checklist(options=[{"label": "Full Dataset", "value": 1}], value=[], id={"type": "check-full-context", "index": dataset_id}, switch=True, inline=True, className="small text-muted ms-1"),
                                    dbc.Button("Generate", id={"type": "btn-generate", "index": dataset_id}, color="success", className="px-3 fw-bold shadow-sm", size="sm", style={"borderRadius": "20px"}),
                                    dbc.Button([html.I(className="bi bi-file-earmark-spreadsheet")], id={"type": "btn-export", "index": dataset_id}, color="light", outline=True, size="sm", className="ms-2"),
                                    dbc.Button([html.I(className="bi bi-file-earmark-pdf")], id={"type": "btn-export-pdf", "index": dataset_id}, color="light", outline=True, size="sm", className="ms-1"),
                                ], className="d-flex align-items-center")
                            ], md=8, className="d-flex align-items-center justify-content-end")
                        ])
                    ], className="bg-dark border-bottom-0 p-2"),

                    dbc.CardBody([
                        dag.AgGrid(
                            id={"type": "data-grid", "index": dataset_id},
                            className="ag-theme-alpine-dark",
                            columnDefs=new_cols,
                            rowData=saved_state.get("data-grid", new_data),
                            rowClassRules=row_class_rules,
                            defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
                            dashGridOptions={"rowSelection": "multiple", "undoRedoCellEditing": True, "columnSize": "sizeToFit"},
                            style={"height": "500px", "width": "100%", "border": "none"},
                        )
                    ], className="p-0 border-0 rounded-bottom overflow-hidden")
                ], className="tabu-card mb-4 shadow-lg border-secondary"),

                # Cross-Validation Benchmark
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cross-Validation Benchmark", className="mb-2"),
                        dbc.Row([
                            dbc.Col(dbc.InputGroup([dbc.InputGroupText("Folds"), dbc.Input(id={"type": "bench-folds", "index": dataset_id}, type="number", value=3)], size="sm"), width=4),
                            dbc.Col(dbc.Button("Run Benchmark", id={"type": "btn-run-benchmark", "index": dataset_id}, color="primary", size="sm", className="w-100"), width=3)
                        ], className="mb-4 align-items-center"),
                        dbc.Collapse(dbc.Progress(id={"type": "bench-progress-bar", "index": dataset_id}, value=0, striped=True, animated=True), id={"type": "bench-progress-collapse", "index": dataset_id}, is_open=False)
                    ], className="p-4")
                ], className="tabu-card mb-5 shadow border-secondary"),

                # EDA
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Exploratory Data Analysis", className="mb-2"),
                        dbc.Button("Refresh Dashboard", id={"type": "btn-refresh-eda", "index": dataset_id}, color="warning", outline=True, className="mb-3"),
                        html.Div(id={"type": "eda-dashboard-container", "index": dataset_id}, children=eda_content)
                    ], className="p-4")
                ], className="tabu-card mb-5 shadow border-secondary"),

                # ML Evaluation Settings
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H4([html.I(className="bi bi-cpu-fill me-2 text-primary"), "ML Evaluation Results"], className="mb-0"),
                        ], className="d-flex align-items-center mb-3"),
                        html.Div(id={"type": "ml-evaluation-results-dashboard", "index": dataset_id}, children=ml_content)
                    ], className="p-4")
                ], id={"type": "ml-evaluation-card", "index": dataset_id}, className="tabu-card mb-5 shadow border-secondary"),

                # Tooltips for Workbench Actions
                dbc.Tooltip("Export augmented rows to CSV", target={"type": "btn-export", "index": dataset_id}),
                dbc.Tooltip("Generate detailed Dataset Report (PDF)", target={"type": "btn-export-pdf", "index": dataset_id}),
                dbc.Tooltip("Mark selected rows as Context (Blue)", target={"type": "btn-mode-context", "index": dataset_id}),
                dbc.Tooltip("Mark selected rows as Targets (Red)", target={"type": "btn-mode-target", "index": dataset_id}),
                dbc.Tooltip("Detect target columns automatically via LLM", target={"type": "btn-auto-suggest-targets", "index": dataset_id}),
                dbc.Tooltip("Inspect selected rows generation details", target={"type": "btn-inspect-row", "index": dataset_id}),
                dbc.Tooltip("Add new empty Row at the end", target={"type": "btn-add-row", "index": dataset_id}),
                dbc.Tooltip("Add new Column to the current view", target={"type": "btn-add-col", "index": dataset_id}),
                dbc.Tooltip("Remove the currently selected/active column", target={"type": "btn-remove-col", "index": dataset_id}),
                
                # Inspect Modal
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-search me-2"), "Inspect Generation Details"])),
                    dbc.ModalBody(id={"type": "inspect-modal-body", "index": dataset_id}),
                ], id={"type": "inspect-modal", "index": dataset_id}, size="lg", is_open=False),
            ], className="py-3 text-light w-100")
        ]
    )
