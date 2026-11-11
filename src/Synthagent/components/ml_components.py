from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np

def get_model_config_ui(model_type, dataset_id, task_type="classification", n_samples=None, n_features=None):
    # Auto-adaptive logic for UI placeholders
    s_rf_est = min(300, max(50, int(n_samples / 5))) if n_samples else 100
    s_rf_depth = min(20, max(5, int(np.log2(n_samples + 2)))) if n_samples else "Auto"
    
    s_xgb_est = min(400, max(50, int(n_samples / 5))) if n_samples else 100
    s_xgb_depth = min(12, max(3, int(np.log2(n_features + 2)))) if n_features else 6
    
    s_mlp_l1 = min(512, max(64, n_features * 4)) if n_features else 100
    s_mlp_l2 = min(256, max(32, n_features * 2)) if n_features else 50
    s_mlp_layers = f"({s_mlp_l1}, {s_mlp_l2})" if n_samples and n_samples > 500 else (f"({s_mlp_l1 // 2},)" if n_samples else "(100,)")
    s_mlp_iter = max(1000, min(5000, 1000000 // (n_samples + 1))) if n_samples else 200
    
    s_knn_k = min(15, max(3, int(np.sqrt(n_samples)))) if n_samples else 5
    s_dt_depth = min(15, max(4, int(np.log2(n_samples)))) if n_samples else "Auto"
    s_gb_est = min(300, max(50, int(n_samples / 5))) if n_samples else 100
    s_ab_est = min(150, max(50, int(n_samples / 5))) if n_samples else 50

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
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-rf-n_estimators"}, type="number", value=None, placeholder=str(s_rf_est), step=10, size="sm")
                    ], width=4),
                    dbc.Col([
                        html.Label("Max Depth:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-rf-max_depth"}, type="number", value=None, placeholder=str(s_rf_depth), size="sm")
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
                        html.Label("Estimators:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-xgb-n_estimators"}, type="number", value=None, placeholder=str(s_xgb_est), size="sm")
                    ], width=4),
                    dbc.Col([
                        html.Label("Max Depth:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-xgb-max_depth"}, type="number", value=None, placeholder=str(s_xgb_depth), size="sm")
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
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-mlp-hidden_layer_sizes"}, type="text", value=None, placeholder=str(s_mlp_layers), size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("Max Iterations (Epochs):", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-mlp-max_iter"}, type="number", value=None, placeholder=str(s_mlp_iter), size="sm")
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
    if model_type == "svm":
        return dbc.Card([
            dbc.CardHeader("Configure Support Vector Machine", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Kernel:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Select(id={"type": "ml-param", "index": f"{dataset_id}-svm-kernel"},
                                   options=[{"label": "RBF", "value": "rbf"}, {"label": "Linear", "value": "linear"}, {"label": "Poly", "value": "poly"}],
                                   value="rbf", size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("C (Regularization):", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-svm-C"}, type="number", value=1.0, size="sm")
                    ], width=6)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "knn":
        return dbc.Card([
            dbc.CardHeader("Configure K-Nearest Neighbors", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Neighbors (k):", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-knn-n_neighbors"}, type="number", value=None, placeholder=str(s_knn_k), size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("Weights:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Select(id={"type": "ml-param", "index": f"{dataset_id}-knn-weights"},
                                   options=[{"label": "Uniform", "value": "uniform"}, {"label": "Distance", "value": "distance"}],
                                   value="uniform", size="sm")
                    ], width=6)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "nb":
        return dbc.Card([
            dbc.CardHeader("Configure Naive Bayes", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Var Smoothing:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-nb-var_smoothing"}, type="number", value=1e-9, step=1e-10, size="sm")
                    ], width=12)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "dt":
        return dbc.Card([
            dbc.CardHeader("Configure Decision Tree", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Max Depth:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-dt-max_depth"}, type="number", value=None, placeholder=str(s_dt_depth), size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("Min Split:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-dt-min_samples_split"}, type="number", value=2, size="sm")
                    ], width=6)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "gb":
        return dbc.Card([
            dbc.CardHeader("Configure Gradient Boosting", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Estimators:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-gb-n_estimators"}, type="number", value=None, placeholder=str(s_gb_est), step=10, size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("Learning Rate:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-gb-learning_rate"}, type="number", value=0.1, step=0.01, size="sm")
                    ], width=6)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    if model_type == "ab":
        return dbc.Card([
            dbc.CardHeader("Configure AdaBoost", className="py-1 small fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Estimators:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-ab-n_estimators"}, type="number", value=None, placeholder=str(s_ab_est), step=10, size="sm")
                    ], width=6),
                    dbc.Col([
                        html.Label("Learning Rate:", className="x-small fw-bold text-muted mb-1"),
                        dbc.Input(id={"type": "ml-param", "index": f"{dataset_id}-ab-learning_rate"}, type="number", value=1.0, step=0.1, size="sm")
                    ], width=6)
                ], className="g-2")
            ], className="px-3 py-2")
        ], className="mb-2 border-secondary bg-dark text-light")

    return None
