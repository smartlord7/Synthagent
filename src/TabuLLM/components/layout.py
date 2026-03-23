from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from utils.constants import initial_data, initial_cols, row_class_rules

def create_layout():
    return html.Div([
        # Stores for tracking Original vs Augmented state
        dcc.Store(id="original-data-store", data=initial_data),
        dcc.Store(id="active-col-store"),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Store(id="context-row-ids", data=[]),
        dcc.Store(id="target-row-ids", data=[]),
        dcc.Store(id="selection-mode", data="target"), # Default mode

        # Navbar
        dbc.NavbarSimple(
            brand=html.Div([
                html.I(className="bi bi-cpu me-2", style={"fontSize": "1.5rem"}), 
                html.Span("TabuLLM", className="brand-text ms-1", style={"fontSize": "1.5rem"})
            ]),
            brand_href="#",
            color="dark",
            dark=True,
            className="mb-4 shadow-sm border-bottom border-secondary p-3",
            children=[
                dcc.Upload(
                    id='upload-data',
                    children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Dataset"],
                                        color="primary", className="btn-sm rounded-pill px-3 shadow-sm"),
                    multiple=False
                ),
                html.Div(id="upload-file-info", className="ms-3 d-inline-block text-muted small")
            ]
        ),

        dbc.Container([
            # Control Panel Ribbon
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # Left Actions: Structure & Mode
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button([html.I(className="bi bi-plus-square me-1"), "Row"], id="btn-add-row", color="secondary", size="sm"),
                                dbc.Button([html.I(className="bi bi-layout-three-columns me-1"), "Column"], id="btn-add-col", color="secondary", size="sm"),
                            ], className="me-3"),
                            
                            html.Span("|", className="mx-2 text-muted fw-light d-none d-md-inline"),

                            dbc.ButtonGroup([
                                dbc.Button([html.I(className="bi bi-info-circle me-1"), "Set Context"], id="btn-mode-context", color="info", outline=True, size="sm"),
                                dbc.Button([html.I(className="bi bi-bullseye me-1"), "Set Targets"], id="btn-mode-target", color="danger", outline=True, size="sm"),
                            ], className="me-2"),
                            
                            dbc.Checklist(
                                options=[{"label": "Entire Dataset Context", "value": 1}],
                                value=[], id="check-full-context", switch=True, inline=True, className="small text-muted ms-1 align-middle"
                            ),
                        ], md=7, className="d-flex align-items-center mb-2 mb-md-0"),

                        # Right Actions: Strategy & Big Action
                        dbc.Col([
                            dbc.Select(
                                id='algo-selector',
                                options=[{'label': 'Bulk (Contextual)', 'value': 'bulk'},
                                         {'label': 'Independent Fill', 'value': 'independent'}],
                                value='bulk', style={'maxWidth': '180px'}, size="sm", className="me-3"
                            ),
                            
                            dbc.Button([html.I(className="bi bi-lightning-charge-fill me-2"), "Generate"], 
                                       id="btn-generate", color="success", className="px-4 fw-bold shadow", size="md",
                                       style={"borderRadius": "20px", "transition": "all 0.3s ease"}),
                            
                            html.Span("|", className="mx-2 text-muted fw-light"),
                            
                            dbc.Button([html.I(className="bi bi-file-earmark-pdf-fill me-1"), "Report"], 
                                       id="btn-export-pdf", color="danger", outline=True, size="sm", className="me-1"),

                            html.Span("|", className="mx-2 text-muted fw-light"),
                            
                            dbc.Button([html.I(className="bi bi-download")], id="btn-export", color="light", outline=True, size="sm", className="rounded-circle"),
                            dcc.Download(id="download-pdf-report")
                        ], md=5, className="d-flex align-items-center justify-content-md-end"),
                    ]),

                    
                    dbc.Collapse(
                        dbc.Progress(id="bench-progress-bar", value=0, striped=True, animated=True, style={"height": "12px", "borderRadius": "6px"}, color="success"),
                        id="bench-progress-collapse", is_open=False, className="mt-3"
                    ),
                    
                    dcc.Store(id="bench-state-store", data={"active": False, "k_total": 0, "k_current": 0, "fraction": 0.2, "results": []}),
                    dcc.Store(id="bench-rows-store", data=[]),
                    dcc.Store(id="dataset-description-store", data=""),

                ], className="p-3")
            ], className="tabu-card mb-4 shadow border-secondary"),

            # Dataset Metadata & Description Section
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        # Statistics Column
                        dbc.Col([
                            html.H6([html.I(className="bi bi-grid-fill me-2 text-info"), "Quick Statistics"], className="text-muted small fw-bold mb-3 d-flex align-items-center"),
                            html.Div(id="dataset-stats-grid", children=[
                                dbc.Alert("Upload a file to view statistics.", color="secondary", className="small")
                            ])
                        ], width=4, className="border-end border-secondary"),
                        
                        # Description Column
                        dbc.Col([
                            html.H6([html.I(className="bi bi-cpu-fill me-2 text-info"), "Structured Description"], className="text-muted small fw-bold mb-3 d-flex align-items-center"),
                            dbc.Button([html.I(className="bi bi-magic me-1"), "Generate Insights"], 
                                       id="btn-gen-desc", color="info", size="sm", outline=True, className="mb-2 rounded-pill"),
                            dbc.Spinner(
                                html.Div(id="dataset-description-output", className="text-light small mt-2", style={"maxHeight": "200px", "overflowY": "auto"}),
                                size="sm", color="info"
                            )
                        ], width=8)
                    ])
                ], title=[html.I(className="bi bi-bar-chart-fill me-2 text-info"), "Dataset Metadata & LLM Analysis"], item_id="meta-item")
            ], id="dataset-meta-accordion", className="mb-4 shadow-sm"),




            # Generation Parameters Accordion
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Style & Length:", className="fw-bold text-muted small"),
                            dbc.Select(id="param-tone",
                                       options=[{"label": "Concise / Standard", "value": "Concise / Standard"},
                                                {"label": "Descriptive", "value": "Descriptive"},
                                                {"label": "Professional", "value": "Professional"},
                                                {"label": "Casual", "value": "Casual"}],
                                       value='Concise / Standard', size="sm")
                        ], width=4),
                        dbc.Col([
                            html.Label("Data Consistency:", className="fw-bold text-muted small"),
                            dbc.Select(id="param-consistency",
                                       options=[{"label": "Strict (Perfectly uniform)", "value": "Strict (Perfectly uniform)"},
                                                {"label": "Realistic (Natural variations)", "value": "Realistic (Natural variations)"}],
                                       value='Strict (Perfectly uniform)', size="sm")
                        ], width=4),
                        dbc.Col([
                            html.Label("Value Uniqueness:", className="fw-bold text-muted small"),
                            dbc.Select(id="param-uniqueness",
                                       options=[{"label": "Allow Duplicates (Standard)", "value": "Allow Duplicates (Standard)"},
                                                {"label": "Prefer Unique Values", "value": "Prefer Unique Values"}],
                                       value='Allow Duplicates (Standard)', size="sm")
                        ], width=4),
                        dbc.Col([
                            html.Label("Creativity (Temperature):", className="fw-bold text-muted small"),
                            dcc.Slider(id="temperature-slider", min=0.0, max=1.0, step=0.1, value=0.3,
                                       marks={0: {'label': 'Strict', 'style': {'color': '#aaa'}}, 1: {'label': 'Creative', 'style': {'color': '#aaa'}}})
                        ], width=6, className="mt-4")
                    ], className="mb-3"),
                    html.Hr(className="border-secondary"),
                    html.H6([html.I(className="bi bi-funnel me-2"), "Custom Rules & Constraints"],
                            className="text-muted mb-3"),
                    html.Div(id="custom-params-container", children=[]),
                    dbc.Button([html.I(className="bi bi-plus-circle me-1"), "Add Custom Rule"], id="btn-add-param",
                               size="sm", color="secondary", outline=True, className="mt-2")
                ], title="⚙️ Generation Parameters & Rules", item_id="params-item", class_name="bg-dark text-light border-secondary")
            ], start_collapsed=True, className="mb-4 shadow-sm tabu-card"),

            # Quick Rename & Manage Columns inline with Grid
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="bi bi-pencil-square me-2 text-info"),
                        html.Span("Quick Rename:", className="text-muted me-2 fw-bold text-uppercase", style={"fontSize": "0.75rem", "letterSpacing": "1px"}),
                        dbc.Input(id="quick-rename-input", type="text", placeholder="Click cell to select column & type to rename...",
                                  size="sm", className="bg-dark text-light border-secondary", style={"width": "280px", "display": "inline-block"}),
                        dbc.Button([html.I(className="bi bi-trash-fill me-1"), "Remove Column"], 
                                   id="btn-remove-col", color="danger", size="sm", outline=True, className="ms-3 rounded-pill")
                    ], className="d-flex align-items-center mb-3")
                ], width=12)
            ]),


            # The Grid
            dbc.Card([
                dbc.CardBody([
                    dag.AgGrid(
                        id="data-grid",
                        className="ag-theme-alpine-dark",
                        columnDefs=initial_cols,
                        rowData=initial_data,
                        rowClassRules=row_class_rules,
                        defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
                        dashGridOptions={"rowSelection": "multiple", "undoRedoCellEditing": True},
                        style={"height": "500px", "width": "100%", "border": "none"},
                    )
                ], className="p-0 border-0 rounded overflow-hidden")
            ], className="tabu-card mb-5 shadow-lg border-secondary"),

            # Cross-Validation Benchmark 
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4([html.I(className="bi bi-speedometer2 me-2 text-primary"), "Cross-Validation Benchmark"], className="mb-2"),
                            html.P("Test the generation capabilities across multiple subsets (folds) to average metrics.", className="text-muted small mb-4"),
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Context Fraction", className="bg-dark text-light border-secondary"),
                                dbc.Input(id="bench-fraction", type="number", min=0.1, max=0.9, step=0.1, value=0.2, className="bg-dark text-light border-secondary"),
                            ], size="sm"),
                        ], width=3),
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Folds (K)", className="bg-dark text-light border-secondary"),
                                dbc.Input(id="bench-folds", type="number", min=1, max=10, step=1, value=3, className="bg-dark text-light border-secondary"),
                            ], size="sm"),
                        ], width=3),
                        dbc.Col([
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run Benchmark"], id="btn-run-benchmark", color="primary", size="sm", className="w-100 px-4 rounded-pill")
                        ], width=2)
                    ], className="mb-4 align-items-center"),
                    
                    dbc.Collapse(
                        dbc.Progress(id="bench-progress-bar", value=0, striped=True, animated=True, style={"height": "12px", "borderRadius": "6px"}, color="success"),
                        id="bench-progress-collapse", is_open=False, className="mb-4"
                    ),
                    
                    dcc.Store(id="bench-state-store", data={"active": False, "k_total": 0, "k_current": 0, "fraction": 0.2, "results": []}),
                    dcc.Store(id="bench-rows-store", data=[]),

                ], className="p-4")
            ], className="tabu-card mb-5 shadow border-secondary"),

            # Exploratory Data Analysis
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4([html.I(className="bi bi-bar-chart-fill me-2 text-warning"), "Exploratory Data Analysis"], className="mb-2"),
                            html.P(
                                "Compare the structural integrity and distribution of your Original Dataset vs. the AI-Augmented Dataset.",
                                className="text-muted small mb-4"),
                        ], width=8),
                        dbc.Col([
                            dbc.Button([html.I(className="bi bi-arrow-repeat me-2"), "Refresh Dashboard"], id="btn-refresh-eda",
                                       color="warning", outline=True, className="float-end rounded-pill mt-2")
                        ], width=4)
                    ]),
                    # Container for dynamic EDA graphs
                    html.Div(id="eda-dashboard-container", children=[
                        dbc.Alert([html.I(className="bi bi-info-circle me-2"), "Click 'Refresh Dashboard' to generate insights."], color="dark", className="border-secondary text-muted")
                    ])
                ], className="p-4")
            ], className="tabu-card mb-5 shadow border-secondary")

        ], fluid=True, className="px-4")
    ])
