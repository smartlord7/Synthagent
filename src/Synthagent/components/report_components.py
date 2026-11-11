from dash import dcc, html
import dash_bootstrap_components as dbc

def create_report_settings_modal(initial_data=None):
    """Returns the modal for configuring and confirming the global audit report."""
    default_config = {
        "include_plots": True,
        "include_appendix": True,
        "include_audit": True,
        "include_counter": True,
        "selected_datasets": []
    }
    config = initial_data if initial_data else default_config

    return html.Div([
        # Persistence store for report configurations
        dcc.Store(id="report-config-store", storage_type="session", data=config),
        
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-file-earmark-pdf-fill me-2 text-warning"), "Configure Global Audit Report"])),
            dbc.ModalBody([
                html.Div([
                    html.H6([html.I(className="bi bi-layers-fill me-2"), "Datasets to Include"], className="text-info small fw-bold mb-3 border-bottom border-secondary pb-2"),
                    html.P("Select the processed benchmarks to feature in the executive summary and detailed audit.", className="text-muted xx-small mb-3"),
                    dbc.Spinner(html.Div(id="report-dataset-list", className="bg-black p-3 rounded border border-secondary mb-4", style={"maxHeight": "300px", "overflowY": "auto"})),
                ]),
                
                html.Div([
                    html.H6([html.I(className="bi bi-gear-wide-connected me-2"), "Report Components & Depth"], className="text-info small fw-bold mb-3 border-bottom border-secondary pb-2"),
                    dbc.Stack([
                        dbc.Switch(id="report-opt-plots", label="Include Visual Coverage (PCA & Correlation)", value=True, className="small text-light"),
                        dbc.Switch(id="report-opt-audit", label="Include AI Audit Narratives & Insights", value=True, className="small text-light"),
                        dbc.Switch(id="report-opt-appendix", label="Include Full Data Trace Appendix (Fold Samples)", value=True, className="small text-light"),
                        dbc.Switch(id="report-opt-counter", label="Show Row Counter (#) in Data Tables", value=True, className="small text-light"),
                    ], gap=2)
                ], className="mb-2")
            ], className="bg-dark text-light"),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="btn-report-cancel", color="secondary", outline=True, size="sm"),
                dbc.Button([html.I(className="bi bi-check2-circle me-2"), "Generate PDF Audit"], id="btn-report-confirm", color="warning", size="sm", className="ms-2 shadow-sm")
            ], className="bg-dark border-secondary")
        ], id="report-settings-modal", centered=True, size="lg", backdrop="static", className="shadow-lg border-secondary")
    ])
