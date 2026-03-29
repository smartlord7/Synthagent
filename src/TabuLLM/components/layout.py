from dash import dcc, html
import dash_bootstrap_components as dbc
from components.dataset_tab import create_dataset_tab

def create_layout():
    # Initial state with one default workbench tab
    default_id = "default-dataset"
    default_tab = create_dataset_tab(default_id, "Workbench")
    
    return html.Div([
        # Master Store for tracking multiple datasets
        dcc.Store(id="datasets-metadata-store", data={
            default_id: {"filename": "Workbench", "status": "Ready", "rows": 10, "cols": 5}
        }),
        dcc.Store(id="active-tab-store"),

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
                    
                    # New Pipeline Toggle Button
                    dbc.Button([html.I(className="bi bi-diagram-3 me-2"), "Pipeline"],
                               id="btn-open-pipeline", color="primary", outline=True,
                               className="btn-sm rounded-pill px-3 shadow-sm me-2"),
                               
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Datasets"],
                                            color="info", className="btn-sm rounded-pill px-3 shadow-sm"),
                        multiple=True, className="d-inline-block"
                    )
                ], className="d-flex align-items-center")
            ], fluid=True),
            color="dark", dark=True, className="mb-0 shadow hover-shadow border-bottom border-secondary py-2"
        ),

        dbc.Container([
            # Pipeline Modal (replaces the Pipeline Tab)
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-cpu-fill me-2"), "Parallel Pipeline Configuration"])),
                dbc.ModalBody(id="pipeline-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="btn-close-pipeline", className="ms-auto", n_clicks=0)
                ),
            ], id="pipeline-modal", size="lg", centered=True, scrollable=True, backdrop="static", is_open=False, className="bg-dark text-light border-secondary shadow-lg"),
            
            # Delete Confirmation Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-exclamation-triangle-fill me-2 text-danger"), "Confirm Removal"])),
                dbc.ModalBody([
                    html.P("Are you sure you want to remove this dataset?", className="mb-1"),
                    html.P(id="delete-dataset-name", className="fw-bold text-info small"),
                    html.P("All associated configurations, synthetic data, and metrics will be permanently cleared.", className="text-muted small"),
                    dcc.Store(id="pending-delete-id-store")
                ]),
                dbc.ModalFooter([
                    dbc.Button("Cancel", id="btn-cancel-delete", color="secondary", outline=True, size="sm"),
                    dbc.Button("Remove", id="btn-confirm-delete", color="danger", size="sm", className="ms-2")
                ]),
            ], id="delete-dataset-modal", centered=True, backdrop="static"),

            dcc.Tabs(id="tabs-datasets", value=default_id, children=[
                default_tab,
            ], className="custom-tabs-container"),
        ], fluid=True, className="px-0")
    ])
