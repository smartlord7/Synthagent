import uuid
from dash import dcc, html
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from utils.constants import initial_data, initial_cols, row_class_rules, ai_highlight_style
from services.data_manager import df_to_aggrid_format

def create_dataset_tab(dataset_id, filename, df=None):
    if df is not None:
        # Build Column Definitions
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
            
        # Calculate Quick Stats
        missing_count = int(df.isnull().sum().sum())
        memory_use = df.memory_usage(deep=True).sum() / 1024 # KB
        
        stats_grid = dbc.Row([
            dbc.Col([html.H4(f"{df.shape[0]}", className="text-info fw-bold mb-0"), html.Small("Rows", className="text-muted")], width=6, className="text-center mb-3"),
            dbc.Col([html.H4(f"{df.shape[1]}", className="text-info fw-bold mb-0"), html.Small("Columns", className="text-muted")], width=6, className="text-center mb-3"),
            dbc.Col([html.H4(f"{missing_count}", className="text-warning mb-0"), html.Small("Missing", className="text-muted")], width=6, className="text-center"),
            dbc.Col([html.H4(f"{memory_use:.1f} KB", className="text-success mb-0"), html.Small("Memory", className="text-muted")], width=6, className="text-center")
        ], className="g-2")
    else:
        new_data = initial_data
        new_cols = initial_cols
        stats_grid = dbc.Alert("Upload file to view stats.", color="secondary", className="small py-2")

    return dcc.Tab(
        label=html.Span([html.I(className="bi bi-table me-2"), filename]),
        value=dataset_id,
        className="custom-tab",
        selected_className="custom-tab--selected",
        children=[
            html.Div([
                # Stores for tracking this dataset's state (Pattern-Matching)
                dcc.Store(id={"type": "original-data-store", "index": dataset_id}, data=new_data),
                dcc.Store(id={"type": "active-col-store", "index": dataset_id}),
                dcc.Store(id={"type": "context-row-ids", "index": dataset_id}, data=[]),
                dcc.Store(id={"type": "target-row-ids", "index": dataset_id}, data=[]),
                dcc.Store(id={"type": "selection-mode", "index": dataset_id}, data="target"),
                dcc.Store(id={"type": "dataset-description-store", "index": dataset_id}, data=""),
                dcc.Store(id={"type": "bench-state-store", "index": dataset_id}, data={"active": False, "k_total": 0, "k_current": 0, "fraction": 0.2, "results": []}),
                
                dcc.Download(id={"type": "download-csv", "index": dataset_id}),
                dcc.Download(id={"type": "download-pdf", "index": dataset_id}),

                dbc.Row([
                    # Left Widget: Dataset Metadata
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Div([
                                    html.I(className="bi bi-bar-chart-fill me-2 text-info"), "Metadata & AI Analysis"
                                ], className="fw-bold text-light d-flex align-items-center")
                            ], className="bg-dark border-secondary"),
                            dbc.CardBody([
                                html.H6([html.I(className="bi bi-grid-fill me-2 text-info"), "Quick Statistics"], className="text-muted small fw-bold mb-2 d-flex align-items-center"),
                                html.Div(id={"type": "dataset-stats-grid", "index": dataset_id}, children=stats_grid, className="mb-3"),
                                
                                html.Hr(className="border-secondary my-2"),
                                
                                html.Div([
                                    html.H6([html.I(className="bi bi-cpu-fill me-2 text-info"), "Structured Description"], className="text-muted small fw-bold mb-0 d-inline-block"),
                                    dbc.Button([html.I(className="bi bi-magic me-1"), "Insights"], id={"type": "btn-gen-desc", "index": dataset_id}, color="info", size="sm", outline=True, className="ms-2 py-0 px-2 rounded-pill"),
                                ], className="d-flex align-items-center mb-2"),
                                
                                dbc.Spinner(
                                    html.Div(id={"type": "dataset-description-output", "index": dataset_id}, className="text-light small mt-1", style={"maxHeight": "160px", "overflowY": "auto"}),
                                    size="sm", color="info"
                                )
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
                                                   value='Concise / Standard', size="sm")
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Consistency:", className="fw-bold text-muted small"),
                                        dbc.Select(id={"type": "param-consistency", "index": dataset_id},
                                                   options=[{"label": "Strict", "value": "Strict (Perfectly uniform)"},
                                                            {"label": "Realistic", "value": "Realistic (Natural variations)"}],
                                                   value='Strict (Perfectly uniform)', size="sm")
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Uniqueness:", className="fw-bold text-muted small"),
                                        dbc.Select(id={"type": "param-uniqueness", "index": dataset_id},
                                                   options=[{"label": "Allow Duplicates", "value": "Allow Duplicates (Standard)"},
                                                            {"label": "Prefer Unique", "value": "Prefer Unique Values"}],
                                                   value='Allow Duplicates (Standard)', size="sm")
                                    ], width=4),
                                ], className="mb-2"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Temperature:", className="fw-bold text-muted small"),
                                        dcc.Slider(id={"type": "temperature-slider", "index": dataset_id}, min=0.0, max=1.0, step=0.1, value=0.3,
                                                   marks={0: {'label': 'Strict', 'style': {'color': '#aaa'}}, 1: {'label': 'Creative', 'style': {'color': '#aaa'}}})
                                    ], width=12)
                                ], className="mb-2"),

                                html.Hr(className="border-secondary my-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Chunk Size:", className="fw-bold text-light small"),
                                        dcc.Slider(id={"type": "gen-chunk-size", "index": dataset_id}, min=10, max=100, step=10, value=30)
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.Label("Max Context:", className="fw-bold text-light small me-auto"),
                                            html.Span("~0 Tok", id={"type": "estimated-tokens-lbl", "index": dataset_id}, className="badge bg-info ms-1")
                                        ], className="d-flex align-items-center justify-content-between"),
                                        dcc.Slider(id={"type": "gen-max-context", "index": dataset_id}, min=20, max=1000, step=20, value=60)
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
                                            value="random", size="sm", className="bg-dark text-light border-secondary"
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
                            rowData=new_data,
                            rowClassRules=row_class_rules,
                            defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
                            dashGridOptions={"rowSelection": "multiple", "undoRedoCellEditing": True},
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
                        html.Div(id={"type": "eda-dashboard-container", "index": dataset_id})
                    ], className="p-4")
                ], className="tabu-card mb-5 shadow border-secondary")
            ], className="px-4 py-3")
        ]
    )
