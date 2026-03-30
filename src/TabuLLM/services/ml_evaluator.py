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
        X_train = train_proc.drop(columns=targets, errors='ignore')
        y_train = train_proc[target]
        
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
            
        model = MLEvaluator.get_model(model_type, params, task_type)
        if not model: return None
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task_type == "classification":
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred, average='weighted' if y_train.nunique() > 2 else 'binary'))
                }
            else:
                metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "r2": float(r2_score(y_test, y_pred))
                }
            return metrics
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            return None

    @staticmethod
    def get_model(model_type, params, task_type):
        p = params or {}
        # Clean params if necessary (e.g. string to int)
        clean_p = {}
        for k, v in p.items():
            if k == "hidden_layer_sizes" and isinstance(v, str):
                try: clean_p[k] = eval(v)
                except: clean_p[k] = (100,)
            elif k in ["n_estimators", "max_depth", "max_iter", "min_samples_split"] and v is not None:
                try: clean_p[k] = int(v)
                except: clean_p[k] = v
            else:
                clean_p[k] = v

        if task_type == "classification":
            if model_type == "logistic": return LogisticRegression(**clean_p)
            if model_type == "rf": return RandomForestClassifier(**clean_p)
            if model_type == "xgb": return XGBClassifier(**clean_p)
            if model_type == "mlp": return MLPClassifier(**clean_p)
        else:
            if model_type == "linear" or model_type == "logistic": return Ridge(**clean_p)
            if model_type == "rf": return RandomForestRegressor(**clean_p)
            if model_type == "xgb": return XGBRegressor(**clean_p)
            if model_type == "mlp": return MLPRegressor(**clean_p)
        return None
