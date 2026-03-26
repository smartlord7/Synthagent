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
        dcc.Store(id="dataset-description-store", data=""),

        # Navbar
        dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.I(className="bi bi-cpu text-info", style={"fontSize": "1.75rem"})),
                        dbc.Col(dbc.NavbarBrand("TabuLLM", className="ms-2 fw-bold text-light", style={"fontSize": "1.5rem"}))
                    ], align="center", className="g-0"),
                    href="#", style={"textDecoration": "none"}
                ),
                html.Div([
                    html.Span(id="upload-file-info", className="me-3 text-muted small align-middle"),
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Dataset"],
                                            color="info", className="btn-sm rounded-pill px-3 shadow-sm"),
                        multiple=False, className="d-inline-block"
                    )
                ], className="d-flex align-items-center")
            ], fluid=True),
            color="dark", dark=True, className="mb-4 shadow hover-shadow border-bottom border-secondary py-2"
        ),

        dbc.Container([



            # Dataset Metadata & Description Section
            # 2-Column Dashboard Widgets Row
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
                            html.Div(id="dataset-stats-grid", children=[
                                dbc.Alert("Upload file to view stats.", color="secondary", className="small py-2")
                            ], className="mb-3"),
                            
                            html.Hr(className="border-secondary my-2"),
                            
                            html.Div([
                                html.H6([html.I(className="bi bi-cpu-fill me-2 text-info"), "Structured Description"], className="text-muted small fw-bold mb-0 d-inline-block"),
                                dbc.Button([html.I(className="bi bi-magic me-1"), "Insights"], id="btn-gen-desc", color="info", size="sm", outline=True, className="ms-2 py-0 px-2 rounded-pill"),
                            ], className="d-flex align-items-center mb-2"),
                            
                            dbc.Spinner(
                                html.Div(id="dataset-description-output", className="text-light small mt-1", style={"maxHeight": "160px", "overflowY": "auto"}),
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
                                html.I(className="bi bi-gear-fill me-2 text-success"), "Generation Parameters & Rules"
                            ], className="fw-bold text-light d-flex align-items-center")
                        ], className="bg-dark border-secondary"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Style & Length:", className="fw-bold text-muted small"),
                                    dbc.Select(id="param-tone",
                                               options=[{"label": "Concise", "value": "Concise / Standard"},
                                                        {"label": "Descriptive", "value": "Descriptive"},
                                                        {"label": "Professional", "value": "Professional"}],
                                               value='Concise / Standard', size="sm")
                                ], width=4),
                                dbc.Col([
                                    html.Label("Consistency:", className="fw-bold text-muted small"),
                                    dbc.Select(id="param-consistency",
                                               options=[{"label": "Strict", "value": "Strict (Perfectly uniform)"},
                                                        {"label": "Realistic", "value": "Realistic (Natural variations)"}],
                                               value='Strict (Perfectly uniform)', size="sm")
                                ], width=4),
                                dbc.Col([
                                    html.Label("Uniqueness:", className="fw-bold text-muted small"),
                                    dbc.Select(id="param-uniqueness",
                                               options=[{"label": "Allow Duplicates", "value": "Allow Duplicates (Standard)"},
                                                        {"label": "Prefer Unique", "value": "Prefer Unique Values"}],
                                               value='Allow Duplicates (Standard)', size="sm")
                                ], width=4),
                            ], className="mb-2"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Creativity (Temperature):", className="fw-bold text-muted small"),
                                    dcc.Slider(id="temperature-slider", min=0.0, max=1.0, step=0.1, value=0.3,
                                               marks={0: {'label': 'Strict', 'style': {'color': '#aaa'}}, 1: {'label': 'Creative', 'style': {'color': '#aaa'}}})
                                ], width=12)
                            ], className="mb-2"),

                            html.Hr(className="border-secondary my-2"),
                            html.H6([html.I(className="bi bi-gear-wide-connected me-2 text-success"), "Local LLM Context Settings"], className="text-muted small fw-bold mb-2"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Chunk Size (Rows):", className="fw-bold text-light small"),
                                    dcc.Slider(id="gen-chunk-size", min=10, max=100, step=10, value=30,
                                               marks={10: {'label': '10', 'style': {'color': '#fff'}},
                                                      30: {'label': '30 (Def)', 'style': {'color': '#fff'}},
                                                      60: {'label': '60', 'style': {'color': '#fff'}},
                                                      100: {'label': '100', 'style': {'color': '#fff'}}})
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.Label("Max Context (Rows):", className="fw-bold text-light small me-auto"),
                                        html.Span("~0 Tok", id="estimated-tokens-lbl", className="badge bg-info ms-1")
                                    ], className="d-flex align-items-center justify-content-between"),
                                    dcc.Slider(id="gen-max-context", min=20, max=1000, step=20, value=60,
                                               marks={20: {'label': '20', 'style': {'color': '#fff'}},
                                                      100: {'label': '100', 'style': {'color': '#fff'}},
                                                      500: {'label': '500', 'style': {'color': '#fff'}},
                                                      1000: {'label': '1000', 'style': {'color': '#fff'}}})
                                ], width=6)
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Label("Context Selection Strategy:", className="fw-bold text-light small mb-0 me-1"),
                                        html.I(className="bi bi-info-circle text-info", id="strategy-info-icon", style={"cursor": "pointer", "fontSize": "0.75rem"})
                                    ], className="d-flex align-items-center mb-1"),
                                    dbc.Select(
                                        id="gen-context-strategy",
                                        options=[
                                            {"label": "Random Shuffle", "value": "random"},
                                            {"label": "Dense (Most Filled Rows)", "value": "dense"},
                                            {"label": "Diverse (Profile Variations)", "value": "diverse"}
                                        ],
                                        value="random",
                                        size="sm",
                                        className="bg-dark text-light border-secondary"
                                    ),
                                    dbc.Tooltip([
                                        html.B("Random Shuffle:"), " Standard arbitrary subsets.", html.Br(),
                                        html.B("Dense:"), " Picks rows containing the most non-empty values.", html.Br(),
                                        html.B("Diverse:"), " Uses K-Means tracking to isolate unique profile distributions."
                                    ], target="strategy-info-icon", placement="top")
                                ], width=12)
                            ], className="mb-2"),



                            html.Hr(className="border-secondary my-2"),
                            html.Div([
                                html.Span([html.I(className="bi bi-funnel me-1 text-success"), "Custom Rules"], className="text-muted small fw-bold me-auto"),
                                dbc.Button([html.I(className="bi bi-plus-circle me-1"), "Rule"], id="btn-add-param", size="sm", color="success", outline=True, className="py-0 px-2 rounded-pill")
                            ], className="d-flex align-items-center mb-1"),
                            html.Div(id="custom-params-container", children=[], style={"maxHeight": "100px", "overflowY": "auto"})
                        ], className="p-3 bg-dark text-light")
                    ], className="tabu-card shadow border-secondary h-100")
                ], md=7)
            ], className="mb-4 g-3"),

            # Integrated Workbench Grid
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        # Left actions header: Quick Rename & Table tools
                        dbc.Col([
                            html.Div([
                                html.I(className="bi bi-pencil-square me-1 text-info"),
                                dbc.Input(id="quick-rename-input", type="text", placeholder="Rename Column...", size="sm", className="bg-dark text-light border-secondary", style={"width": "160px", "display": "inline-block"}),
                                dbc.Button([html.I(className="bi bi-trash")], id="btn-remove-col", color="danger", size="sm", outline=True, className="ms-1 py-1 px-2 text-nowrap"),
                            ], className="d-flex align-items-center")
                        ], md=4, className="d-flex align-items-center mb-2 mb-md-0"),

                        # Right actions header: AI Completion Controls
                        dbc.Col([
                            html.Div([
                                dbc.ButtonGroup([
                                    dbc.Button([html.I(className="bi bi-info-circle me-1"), "Context"], id="btn-mode-context", color="info", outline=True, size="sm", className="text-nowrap"),
                                    dbc.Button([html.I(className="bi bi-bullseye me-1"), "Targets"], id="btn-mode-target", color="danger", outline=True, size="sm", className="text-nowrap"),
                                ], className="me-2"),
                                
                                dbc.Checklist(
                                    options=[{"label": "Entire", "value": 1}],
                                    value=[], id="check-full-context", switch=True, inline=True, className="small text-muted ms-1 me-2 align-middle text-nowrap"
                                ),
                                html.Span("|", className="mx-2 text-muted fw-light d-none d-md-inline"),

                                dbc.Select(
                                    id='algo-selector',
                                    options=[{'label': 'Bulk', 'value': 'bulk'}, {'label': 'Indep', 'value': 'independent'}],
                                    value='bulk', style={'maxWidth': '100px'}, size="sm", className="me-2"
                                ),
                                dbc.Button([html.I(className="bi bi-lightning-charge-fill me-1"), "Generate"], id="btn-generate", color="success", className="px-3 fw-bold shadow-sm text-nowrap", size="sm", style={"borderRadius": "20px"}),
                                html.Span("|", className="mx-2 text-muted fw-light"),
                                dbc.Button([html.I(className="bi bi-file-earmark-pdf-fill")], id="btn-export-pdf", color="danger", outline=True, size="sm", className="px-2 py-1 text-nowrap"),
                                dbc.Button([html.I(className="bi bi-download")], id="btn-export", color="light", outline=True, size="sm", className="px-2 py-1 text-nowrap"),
                                dcc.Download(id="download-pdf-report")
                            ], className="d-flex align-items-center")
                        ], md=8, className="d-flex align-items-center justify-content-md-end")
                    ])
                ], className="bg-dark border-bottom-0 p-2"),

                dbc.CardBody([
                    dbc.Row([
                        # Grid Body
                        dbc.Col([
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
                        ], md=11, className="pe-0"),
                        
                        # "Add Column" literal Column Slot
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-plus-lg fs-5 mb-1"),
                                html.Span("Add Col", className="small fw-bold text-uppercase")
                            ], id="btn-add-col", color="link", className="h-100 w-100 d-flex flex-column align-items-center justify-content-center text-decoration-none text-muted hover-bg-dark border-start border-secondary py-3", style={"borderRadius": "0"})
                        ], md=1, className="ps-0 d-flex")
                    ], className="g-0")
                ], className="p-0 border-0 rounded-bottom overflow-hidden d-flex flex-column"),

                # "Add Row" literal Row Slot
                dbc.CardFooter([
                    dbc.Button([
                        html.I(className="bi bi-plus-lg me-1"), "Add Row"
                    ], id="btn-add-row", color="link", className="w-100 text-decoration-none text-muted small py-1", style={"borderRadius": "0"})
                ], className="bg-dark p-0 border-top border-secondary")
            ], className="tabu-card mb-4 shadow-lg border-secondary"),





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
                                dbc.InputGroupText("Folds (K)", className="bg-dark text-light border-secondary"),
                                dbc.Input(id="bench-folds", type="number", min=1, max=10, step=1, value=3, className="bg-dark text-light border-secondary"),
                            ], size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run Benchmark"], id="btn-run-benchmark", color="primary", size="sm", className="w-100 px-4 rounded-pill")
                        ], width=3)
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
