from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from utils.constants import initial_data, initial_cols, row_class_rules

def create_layout():
    return dbc.Container([
        # Stores for tracking Original vs Augmented state
        dcc.Store(id="original-data-store", data=initial_data),
        dcc.Store(id="active-col-store"),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Store(id="context-row-ids", data=[]),
        dcc.Store(id="target-row-ids", data=[]),
        dcc.Store(id="selection-mode", data="target"), # Default mode

        dbc.Row([
            dbc.Col(html.H2([html.I(className="bi bi-cpu me-2"), "TabuLLM"], className="mt-4 mb-3"), width=8),
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Dataset"],
                                        color="primary", className="mt-4 float-end"),
                    multiple=False
                )
            ], width=4)
        ]),

        html.Hr(),

        # Control Panel
        dbc.Row([
            dbc.Col([
                dbc.Button([html.I(className="bi bi-plus-square me-1"), "Row"], id="btn-add-row", color="secondary",
                           className="me-2", size="sm"),
                dbc.Button([html.I(className="bi bi-layout-three-columns me-1"), "Column"], id="btn-add-col",
                           color="secondary", className="me-2", size="sm"),
                html.Span(" | ", className="mx-2 text-muted"),
                dbc.Button([html.I(className="bi bi-magic me-1"), "Generate"], id="btn-generate", color="success",
                           className="me-2", size="sm"),
                html.Span(" | ", className="mx-2 text-muted"),
                # NEW: Export Button
                dbc.Button([html.I(className="bi bi-download me-1"), "Export CSV"], id="btn-export", color="dark",
                           outline=True, size="sm"),
            ], width="auto"),
            dbc.Col([
                dbc.Select(
                    id='algo-selector',
                    options=[{'label': 'Contextual Bulk', 'value': 'bulk'},
                             {'label': 'Independent Fill', 'value': 'independent'}],
                    value='bulk', style={'width': '200px'}, size="sm"
                )
            ], width="auto"),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="bi bi-info-circle me-1"), "Set Context"], id="btn-mode-context",
                               color="info", outline=True, size="sm"),
                    dbc.Button([html.I(className="bi bi-bullseye me-1"), "Set Targets"], id="btn-mode-target",
                               color="danger", outline=True, size="sm"),
                ], className="me-2"),

                dbc.Checklist(
                    options=[{"label": "Use entire dataset as context", "value": 1}],
                    value=[], id="check-full-context", switch=True, inline=True, className="small text-muted"
                ),
            ], width="auto"),
        ], className="mb-3 align-items-center"),

        # Dynamic Generation Parameters
        dbc.Row([
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Style & Length:", className="fw-bold text-muted small"),
                                dcc.Dropdown(id="param-tone",
                                             options=['Concise / Standard', 'Descriptive', 'Professional', 'Casual'],
                                             value='Concise / Standard', clearable=False)
                            ], width=4),
                            dbc.Col([
                                html.Label("Data Consistency:", className="fw-bold text-muted small"),
                                dcc.Dropdown(id="param-consistency",
                                             options=['Strict (Perfectly uniform)', 'Realistic (Natural variations)'],
                                             value='Strict (Perfectly uniform)', clearable=False)
                            ], width=4),
                            dbc.Col([
                                html.Label("Value Uniqueness:", className="fw-bold text-muted small"),
                                dcc.Dropdown(id="param-uniqueness",
                                             options=['Allow Duplicates (Standard)', 'Prefer Unique Values'],
                                             value='Allow Duplicates (Standard)', clearable=False)
                            ], width=4),
                            dbc.Col([
                                html.Label("Creativity (Temperature):", className="fw-bold text-muted small"),
                                dcc.Slider(id="temperature-slider", min=0.0, max=1.0, step=0.1, value=0.3,
                                           marks={0: 'Strict', 1: 'Creative'})
                            ], width=6, className="mt-3")
                        ], className="mb-3"),
                        html.Hr(),
                        html.H6([html.I(className="bi bi-sliders me-2"), "Custom Rules & Constraints"],
                                className="text-muted mb-3"),
                        html.Div(id="custom-params-container", children=[]),
                        dbc.Button([html.I(className="bi bi-plus-circle me-1"), "Add Custom Rule"], id="btn-add-param",
                                   size="sm", color="secondary", outline=True, className="mt-2")
                    ], title="⚙️ Generation Parameters & Rules", item_id="params-item")
                ], start_collapsed=True, className="mb-3")
            ], width=12)
        ]),

        # Live In-Situ Renaming Bar
        dbc.Row([
            dbc.Col([
                html.I(className="bi bi-pencil-square me-2 text-muted"),
                html.Span("Quick Rename:", className="text-muted me-2", style={"fontSize": "0.9em"}),
                dbc.Input(id="quick-rename-input", type="text", placeholder="Click any cell to rename its column...",
                          size="sm", style={"width": "300px", "display": "inline-block"})
            ], width=12, className="mb-2 bg-light p-2 rounded border")
        ]),

        # The Grid
        dbc.Row([
            dbc.Col([
                dag.AgGrid(
                    id="data-grid",
                    columnDefs=initial_cols,
                    rowData=initial_data,
                    rowClassRules=row_class_rules,
                    defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
                    dashGridOptions={"rowSelection": "multiple", "undoRedoCellEditing": True},
                    style={"height": "400px", "width": "100%"},
                )
            ], width=12)
        ]),

        html.Hr(className="my-5"),

        # --- NEW: EXPLORATORY DATA ANALYSIS DASHBOARD ---
        dbc.Row([
            dbc.Col([
                html.H3([html.I(className="bi bi-bar-chart-fill me-2"), "Exploratory Data Analysis"], className="mb-3"),
                html.P(
                    "Compare the structural integrity and distribution of your Original Dataset vs. the AI-Augmented Dataset.",
                    className="text-muted"),
                dbc.Button([html.I(className="bi bi-arrow-repeat me-2"), "Refresh Dashboard"], id="btn-refresh-eda",
                           color="info", className="mb-4")
            ], width=8)
        ]),

        # Container for dynamic EDA graphs
        html.Div(id="eda-dashboard-container", children=[
            dbc.Alert("Click 'Refresh Dashboard' to generate insights.", color="light")
        ], className="mb-5")

    ], fluid=True)
