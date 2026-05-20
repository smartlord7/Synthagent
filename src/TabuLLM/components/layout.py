from dash import dcc, html
import dash_bootstrap_components as dbc
from components.dataset_tab import create_dataset_tab
from components.report_components import create_report_settings_modal

import os
import json

def create_layout():
    session_file = "workspace_session.json"
    metadata = {}
    pipeline_results = {}
    tabs_children = []
    
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            pipeline_results = data.get("pipeline_results", {})
            pipeline_config = data.get("pipeline_config", {})
            report_config = data.get("report_config", {})
            tab_states = data.get("tabs", {})
            
            for ds_id, ds_meta in metadata.items():
                state = tab_states.get(ds_id, {})
                tabs_children.append(create_dataset_tab(ds_id, ds_meta["filename"], saved_state=state, pipeline_res=pipeline_results.get(ds_id)))
        except Exception as e:
            print("Failed to load session:", e)
            
    if not tabs_children:
        default_id = "default-dataset"
        tabs_children = [create_dataset_tab(default_id, "Workbench")]
        metadata = {default_id: {"filename": "Workbench", "status": "Ready", "rows": 10, "cols": 5}}

    return html.Div([
        # Master Store for tracking multiple datasets
        dcc.Store(id="datasets-metadata-store", data=metadata),
        dcc.Store(id="active-tab-store"),
        dcc.Store(id="pipeline-results-store", data=pipeline_results),
        dcc.Store(id="pipeline-config-store", data=pipeline_config if 'pipeline_config' in locals() else {}),
        dcc.Store(id="pending-upload-hash-store"),
        dcc.Store(id="pending-upload-contents-store"),
        dcc.Store(id="pending-upload-filename-store"),

        # Global Report Settings Modal (Selection & Detail Config)
        create_report_settings_modal(report_config if 'report_config' in locals() else None),
        dcc.Download(id="global-download-report-pdf"),

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
                    
                    # New Global Report Button
                    dbc.Button([html.I(className="bi bi-file-earmark-pdf me-2"), "Global Report"],
                               id="btn-global-report", color="warning", outline=True,
                               className="btn-sm rounded-pill px-3 shadow-sm me-2"),

                    # Documentation Button
                    dbc.Button([html.I(className="bi bi-journal-text me-2"), "Docs"],
                               id="btn-open-docs", color="info", outline=True,
                               className="btn-sm rounded-pill px-3 shadow-sm me-2"),
                               
                    # New Pipeline Toggle Button
                    dbc.Button([html.I(className="bi bi-diagram-3 me-2"), "Pipeline"],
                               id="btn-open-pipeline", color="primary", outline=True,
                               className="btn-sm rounded-pill px-3 shadow-sm me-2"),
                               
                    # Session Context Actions
                    dbc.Button([html.I(className="bi bi-cloud-arrow-up me-2"), "Save Setup"],
                               id="btn-save-session", color="success", outline=True,
                               className="btn-sm rounded-pill px-3 shadow-sm me-2"),
                               
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button([html.I(className="bi bi-file-earmark-arrow-up me-2"), "Upload Datasets"],
                                            color="info", className="btn-sm rounded-pill px-3 shadow-sm"),
                        multiple=True, className="d-inline-block"
                    ),
                    
                    html.Div(id="session-toast-container", className="ms-2 align-middle d-inline-block"),

                    # Tooltips for Navbar Buttons
                    dbc.Tooltip("Download Global Benchmark Report (All Datasets & Metrics)", target="btn-global-report"),
                    dbc.Tooltip("View system architecture and user guides", target="btn-open-docs"),
                    dbc.Tooltip("Configure and run parallel data augmentation pipeline", target="btn-open-pipeline"),
                    dbc.Tooltip("Upload new CSV/Excel datasets to the workbench", target="upload-data"),
                    dbc.Tooltip("Save workspace layout and configurations to file", target="btn-save-session"),
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
            ], id="pipeline-modal", size="lg", is_open=False, className="text-light border-secondary shadow-lg"),
            
            # Documentation Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-journal-richtext me-2"), "System Documentation"])),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Document:", className="small text-muted fw-bold mb-2"),
                            dbc.Select(
                                id="docs-file-selector",
                                options=[
                                    {"label": "End-to-End Pipeline", "value": "end_to_end_pipeline.md"},
                                    {"label": "User Functionalities", "value": "user_functionalities.md"},
                                    {"label": "Pipeline Architecture", "value": "pipeline_architecture.md"},
                                    {"label": "Master Diagram", "value": "pipeline_master_diagram.md"},
                                ],
                                value="end_to_end_pipeline.md",
                                size="sm", className="bg-dark text-light border-secondary mb-3"
                            ),
                        ], width=12),
                    ]),
                    html.Hr(className="border-secondary"),
                    dbc.Spinner(html.Div(id="docs-modal-content", className="markdown-preview p-3 bg-black rounded border border-secondary", style={"minHeight": "400px", "maxHeight": "70vh", "overflowY": "auto"})),
                ], className="bg-dark text-light"),
                dbc.ModalFooter(
                    dbc.Button("Close Docs", id="btn-close-docs", color="secondary", outline=True, size="sm")
                ),
            ], id="docs-modal", size="xl", is_open=False),

            # Delete Confirmation Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-exclamation-triangle-fill me-2 text-danger"), "Confirm Removal"])),
                dbc.ModalBody([
                    html.P("Remove this dataset from the current workbench?", className="mb-1"),
                    html.P(id="delete-dataset-name", className="fw-bold text-info small"),
                    html.P("The session remains archived. You can resume it later by re-uploading the same file.", className="text-muted small"),
                    dcc.Store(id="pending-delete-id-store")
                ]),
                dbc.ModalFooter([
                    dbc.Button("Cancel", id="btn-cancel-delete", color="secondary", outline=True, size="sm"),
                    dbc.Button("Remove from View", id="btn-confirm-delete", color="danger", size="sm", className="ms-2")
                ]),
            ], id="delete-dataset-modal", centered=True, backdrop="static"),

            # Session Browser Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-folder2-open me-2 text-info"), "Resume or Branch Session"])),
                dbc.ModalBody([
                    html.P("We found previous work-sessions for this dataset. Would you like to resume an existing experiment or start a clean slate?", className="small text-muted mb-3"),
                    html.Div(id="session-browser-list", className="mb-3", style={"maxHeight": "400px", "overflowY": "auto"}),
                    html.Hr(className="border-secondary"),
                    dbc.Button([html.I(className="bi bi-plus-circle me-2"), "Start Clean / New Session"], 
                               id="btn-session-start-fresh", color="success", className="w-100 rounded-pill shadow-sm")
                ], className="bg-dark text-light p-4"),
                dbc.ModalFooter(
                    dbc.Button("Cancel Upload", id="btn-session-cancel", color="secondary", outline=True, size="sm")
                )
            ], id="session-browser-modal", centered=True, size="lg", backdrop="static"),

            dcc.Tabs(id="tabs-datasets", value=list(metadata.keys())[0] if metadata else "default-dataset", children=tabs_children, 
                     className="custom-tabs border-bottom border-secondary bg-dark text-light mb-0"),
            
            # Global ML Diagnostics Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle(id="ml-diag-modal-title")),
                dbc.ModalBody([
                    # Navigation Header
                    dbc.Row([
                        dbc.Col(dbc.Button([html.I(className="bi bi-arrow-left me-2"), "Prev Fold"], id="btn-diag-prev", color="secondary", outline=True, size="sm"), width=4),
                        dbc.Col(html.Div(id="ml-diag-fold-label", className="text-center fw-bold text-info"), width=4),
                        dbc.Col(dbc.Button(["Next Fold", html.I(className="bi bi-arrow-right ms-2")], id="btn-diag-next", color="secondary", outline=True, size="sm", className="float-end"), width=4),
                    ], align="center", className="mb-4 bg-dark p-2 rounded border border-secondary"),
                    
                    # Content
                    dbc.Spinner(html.Div(id="ml-diag-modal-content")),
                    
                    html.Hr(className="border-secondary my-4"),
                    html.Div([
                        html.H6([html.I(className="bi bi-gear-fill me-2"), "Model Hyperparameters & Training Config"], className="text-muted small fw-bold mb-3"),
                        dbc.Spinner(html.Div(id="ml-diag-params-content", className="bg-black p-3 rounded border border-secondary mb-4")),
                    ]),
                    
                    html.Hr(className="border-secondary my-4"),
                    html.H6("Aggregate Performance (Mean ± SD)", className="text-muted small fw-bold mb-3"),
                    dbc.Spinner(html.Div(id="ml-diag-aggregate-content")),
                ], className="bg-dark text-light"),
                dcc.Store(id="ml-diag-current-fold", data=0),
                dcc.Store(id="ml-diag-active-model", data=None), # {dataset_id, model_type}
                dbc.ModalFooter(
                    dbc.Button("Close Diagnostics", id="btn-diag-close", color="secondary", outline=True, size="sm", n_clicks=0)
                )
            ], id="ml-diag-modal", size="xl", is_open=False),
            
        ], fluid=True, className="px-0")
    ])
