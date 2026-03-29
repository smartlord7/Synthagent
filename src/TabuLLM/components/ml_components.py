from dash import html, dcc
import dash_bootstrap_components as dbc

def get_model_config_ui(model_type, dataset_id, task_type="classification"):
    if model_type == "logistic" or model_type == "linear":
        return dbc.Card([
            dbc.CardHeader(f"Configure {'Logistic' if task_type == 'classification' else 'Linear/Ridge'} Regression", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Penalty/Regularization:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Select(id={"type": "ml-param", "index": f"{dataset_id}-{model_type}-penalty"},
                                   options=[{"label": "L2", "value": "l2"}, {"label": "None", "value": "none"}],
                                   value="l2", size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("C (Regularization):", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-{model_type}-C"}, type="number", value=1.0, size="sm")
                    ], width=6)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "rf":
        return dbc.Card([
            dbc.CardHeader("Configure Random Forest", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Estimators:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-rf-n_estimators"}, type="number", value=100, step=50, size="sm")
                    ], width=4),
                    dbc.Col([
                        html.Label("Max Depth:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-rf-max_depth"}, type="number", value=None, placeholder="Auto", size="sm")
                    ], width=4),
                    dbc.Col([
                        html.Label("Min Split:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-rf-min_samples_split"}, type="number", value=2, size="sm")
                    ], width=4)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "xgb":
        return dbc.Card([
            dbc.CardHeader("Configure XGBoost", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("LR / Eta:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-xgb-learning_rate"}, type="number", value=0.1, step=0.01, size="sm")
                    ], width=4),
                    dbc.Col([
                        html.Label("Max Depth:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-xgb-max_depth"}, type="number", value=6, size="sm")
                    ], width=4),
                    dbc.Col([
                        html.Label("Subsample:", className="x-small fw-bold text-muted mb-1"),
                        dcc.Slider(id={"type": "ml-param", "index": f"{dataset_id}-xgb-subsample"}, min=0.5, max=1.0, step=0.1, value=1.0)
                    ], width=4)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "mlp":
        return dbc.Card([
            dbc.CardHeader("Configure Neural Network (MLP)", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Hidden Layers:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-mlp-hidden_layer_sizes"}, type="text", value="(100,)", size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("Max Epochs:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-mlp-max_iter"}, type="number", value=200, size="sm")
                    ], width=6)
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Initial LR:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-mlp-learning_rate_init"}, type="number", value=0.001, step=0.0001, size="sm")
                    ], width=12)
                ])
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    return None
