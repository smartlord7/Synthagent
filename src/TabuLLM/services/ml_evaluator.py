import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from utils.logging_config import logger

class MLEvaluator:
    @staticmethod
    def detect_task_type(df, targets):
        """Simple heuristic to detect if it's classification or regression."""
        if not targets: return "classification"
        target = targets[0]
        if target not in df.columns: return "classification"
        if df[target].dtype == 'object' or df[target].nunique() < 15:
            return "classification"
        return "regression"

    @staticmethod
    def train_and_evaluate(train_df, targets, model_type, params=None, task_type="classification", test_df=None):
        if not targets or train_df.empty:
            return None
        
        target = targets[0]
        if target not in train_df.columns:
            logger.warning(f"Target column '{target}' not found in training data.")
            return None
            
        target = targets[0]
        if target not in train_df.columns:
            logger.warning(f"Target column '{target}' not found in training data.")
            return None
            
        def apply_encoding(df_fit, df_transform, cols):
            df_out = df_transform.copy()
            for col in cols:
                if col in df_fit.columns:
                    # Capture unique values from fit set
                    codes, uniques = pd.factorize(df_fit[col])
                    mapping = {val: i for i, val in enumerate(uniques)}
                    # Apply to transform set
                    df_out[col] = df_transform[col].map(mapping).fillna(-1)
            # Fill numeric NaNs
            df_out.fillna(0, inplace=True)
            return df_out

        # Identify categorical columns (non-numeric)
        cat_cols = train_df.select_dtypes(exclude=['number']).columns.tolist()
        
        train_proc = apply_encoding(train_df, train_df, cat_cols)
        # Dropping target and any internal tracking columns
        internal_cols = ['_fold_id', '_row_id', 'is_context', 'is_target', 'ai_audit']
        X_train = train_proc.drop(columns=targets + internal_cols, errors='ignore')
        y_train = train_proc[target]
        
        # Safeguard: Classification requires at least 2 classes
        if task_type == "classification" and y_train.nunique() < 2:
            logger.warning(f"Single class detected in training set for {model_type}. Skipping.")
            return None

        if test_df is not None and not test_df.empty:
            test_proc = apply_encoding(train_df, test_df, cat_cols)
            # Ensure test set has same columns as train set
            for col in X_train.columns:
                if col not in test_proc.columns:
                    test_proc[col] = 0
            X_test = test_proc[X_train.columns]
            y_test = test_proc[target]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
        if X_train.empty:
            logger.warning(f"X_train is empty for {model_type}. Skipping.")
            return None

        # Cast target to int to avoid float class label issues after encoding
        try:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except (ValueError, TypeError):
            pass  # Keep original dtype if cast fails

        # Safeguard: test set must also have >=2 classes for classification scoring
        if task_type == "classification" and y_test.nunique() < 1:
            logger.warning(f"Empty test set for {model_type}. Skipping.")
            return None

        model = MLEvaluator.get_model(model_type, params, task_type, X_train, y_train)
        if model is None:
            return None

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == "classification":
                # Use 'binary' only when BOTH train and test have exactly 2 classes.
                # Otherwise use 'weighted' to safely handle multiclass targets.
                unique_train = np.unique(y_train)
                unique_test = np.unique(y_test)
                is_binary = len(unique_train) == 2 and len(unique_test) == 2
                avg = 'binary' if is_binary else 'weighted'
                f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1": f1,
                }
            else:
                metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "r2": float(r2_score(y_test, y_pred)),
                }
            return metrics
        except Exception as e:
            import traceback
            logger.error(f"Error training {model_type}: {e}")
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def get_model(model_type, params, task_type, X_train=None, y_train=None):
        p = params or {}
        # Clean params if necessary (e.g. string to int)
        clean_p = {}
        for k, v in p.items():
            if v is None or v == "": 
                continue
            if k == "hidden_layer_sizes" and isinstance(v, str):
                try: clean_p[k] = eval(v)
                except: clean_p[k] = (100,)
            elif k in ["n_estimators", "max_depth", "max_iter", "min_samples_split"]:
                try: clean_p[k] = int(v)
                except: clean_p[k] = v
            else:
                clean_p[k] = v

        # Automatically adapt architecture and parameters to dataset characteristics
        if X_train is not None and y_train is not None:
            n_samples, n_features = X_train.shape
            
            if model_type == "rf":
                if "n_estimators" not in clean_p:
                    clean_p["n_estimators"] = min(300, max(50, int(n_samples / 5)))
                if "max_depth" not in clean_p:
                    clean_p["max_depth"] = min(20, max(5, int(np.log2(n_samples + 2))))
            
            elif model_type == "mlp":
                if "hidden_layer_sizes" not in clean_p:
                    l1 = min(512, max(64, n_features * 4))
                    l2 = min(256, max(32, n_features * 2))
                    clean_p["hidden_layer_sizes"] = (l1, l2) if n_samples > 500 else (l1 // 2,)
                if "max_iter" not in clean_p:
                    clean_p["max_iter"] = max(1000, min(5000, 1000000 // (n_samples + 1)))

            elif model_type == "xgb":
                if "n_estimators" not in clean_p:
                    clean_p["n_estimators"] = min(400, max(50, int(n_samples / 5)))
                if "max_depth" not in clean_p:
                    clean_p["max_depth"] = min(12, max(3, int(np.log2(n_features + 2))))

            elif model_type in ["logistic", "linear"]:
                if "max_iter" not in clean_p:
                    clean_p["max_iter"] = min(10000, max(1000, 100000 // (n_samples + 1)))
                if model_type == "logistic" and n_samples > 10000 and "solver" not in clean_p:
                    clean_p["solver"] = "saga"

        if task_type == "classification":
            if model_type == "logistic": return LogisticRegression(**clean_p)
            if model_type == "rf": return RandomForestClassifier(**clean_p)
            if model_type == "xgb": return XGBClassifier(**clean_p)
            if model_type == "mlp": return MLPClassifier(**clean_p)
        else:
            if model_type == "linear" or model_type == "logistic":
                # Remove classification-specific params if switching to Ridge
                for k in ["penalty", "dual", "solver", "C", "multi_class", "l1_ratio"]:
                    clean_p.pop(k, None)
                return Ridge(**clean_p)
            if model_type == "rf": return RandomForestRegressor(**clean_p)
            if model_type == "xgb": return XGBRegressor(**clean_p)
            if model_type == "mlp": return MLPRegressor(**clean_p)
        return None
