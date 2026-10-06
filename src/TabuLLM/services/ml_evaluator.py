import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
    def preprocess_data(df, strategy="impute", targets=None):
        """Handle missing values before ML training."""
        if df.empty: return df
        df_clean = df.copy()
        
        # Internal tracking columns to ignore
        internal = ['_fold_id', '_row_id', 'is_context', 'is_target', 'ai_audit']
        cols_to_process = [c for c in df_clean.columns if c not in internal]

        if strategy == "drop":
            # Drop rows with NAs in features or targets
            df_clean = df_clean.dropna(subset=cols_to_process)
        elif strategy == "impute":
            # Impute Mean for numeric, Mode for categorical
            for col in cols_to_process:
                if df_clean[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    else:
                        mode_vals = df_clean[col].mode()
                        if not mode_vals.empty:
                            val = mode_vals[0]
                            if isinstance(val, (list, dict, tuple)):
                                val = str(val)
                            df_clean[col] = df_clean[col].fillna(val)
                        else:
                            df_clean[col] = df_clean[col].fillna("N/A")
        
        # Fallback filler for any remaining edge cases
        df_clean = df_clean.fillna(0) if strategy == "impute" else df_clean
        return df_clean

    @staticmethod
    def train_and_evaluate(train_df, targets, model_type, params=None, task_type="classification", test_df=None, na_strategy="impute"):
        if not targets or train_df.empty:
            return None, None, "Empty dataset or no targets"
        
        # 0. Preprocess Missing Values
        train_df = MLEvaluator.preprocess_data(train_df, na_strategy, targets)
        if test_df is not None and not test_df.empty:
            test_df = MLEvaluator.preprocess_data(test_df, na_strategy, targets)

        target = targets[0]
        if target not in train_df.columns:
            logger.warning(f"Target column '{target}' not found in training data.")
            return None, None, f"Target '{target}' missing"
            
        def apply_encoding(df_fit, df_transform, cols):
            if not cols: return df_transform.copy()
            df_out = df_transform.copy()
            # Optimization: Pre-calculate mappings to avoid redundant factorize calls
            for col in cols:
                if col in df_fit.columns:
                    # Capture unique values from fit set - ensure string cast to avoid unhashable lists
                    _, uniques = pd.factorize(df_fit[col].astype(str))
                    mapping = {val: i for i, val in enumerate(uniques)}
                    # Apply to transform set
                    df_out[col] = df_transform[col].astype(str).map(mapping).fillna(-1).astype(int)
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
            return None, None, "Single class in train"

        if test_df is not None and not test_df.empty:
            if target not in test_df.columns:
                logger.warning(f"Target column '{target}' not found in test data.")
                return None, None, f"Target '{target}' missing from test data"
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
            return None, None, "No features remaining"

        # Cast target to int to avoid float class label issues after encoding
        try:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except (ValueError, TypeError):
            pass  # Keep original dtype if cast fails

        # Safeguard: test set must also have >=2 classes for classification scoring
        if task_type == "classification" and y_test.nunique() < 1:
            logger.warning(f"Empty test set for {model_type}. Skipping.")
            return None, None, "Empty test targets"

        # --- SCALE SAFEGUARDS ---
        n_samples, n_features = X_train.shape
        if model_type == "svm" and (n_samples > 5000 or n_features > 200):
             return None, None, f"Skipped: Scale ({n_samples}R/{n_features}C too large for SVM)"
        if model_type == "mlp" and n_features > 500:
             return None, None, f"Skipped: Scale ({n_features}C too complex for MLP)"

        model = MLEvaluator.get_model(model_type, params, task_type, X_train, y_train)
        if model is None:
            return None, None, f"Invalid model type '{model_type}'"

        # XGBClassifier requires class labels 0..k-1 and rejects e.g. {1, 2}. Encode for fit and
        # decode predictions back, so metrics stay in the original label space -- f1 with
        # average='binary' scores pos_label=1, so scoring in encoded space would silently
        # change which class is "positive" versus every other model.
        xgb_label_encoder = None
        y_train_fit = y_train
        if task_type == "classification" and model_type == "xgb":
            xgb_label_encoder = LabelEncoder().fit(y_train)
            y_train_fit = xgb_label_encoder.transform(y_train)

        try:
            logger.info(f"[ML] Training {model_type.upper()} Model ({task_type}) on {n_samples} samples x {n_features} features...")
            model.fit(X_train, y_train_fit)
            y_pred = model.predict(X_test)
            if xgb_label_encoder is not None:
                y_pred = xgb_label_encoder.inverse_transform(np.asarray(y_pred).astype(int))

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

            logger.info(f"[ML] {model_type.upper()} Evaluation Complete. Metrics: {metrics}")

            # --- EXTRACT METADATA ---
            metadata = {"model_type": model_type}
            if hasattr(model, "get_params"):
                 metadata["params"] = model.get_params()
            
            try:
                if model_type == "mlp" and hasattr(model, "loss_curve_"):
                     metadata["loss_curve"] = [float(x) for x in model.loss_curve_]
                
                # Models with feature_importances_ attribute
                if model_type in ["rf", "xgb", "gb", "dt", "ab"] and hasattr(model, "feature_importances_"):
                     metadata["feature_importances"] = [float(x) for x in model.feature_importances_]
                     metadata["feature_names"] = X_train.columns.tolist()
                
                # Models with coef_ attribute (Linear models, Linear SVM)
                if model_type in ["logistic", "linear", "svm"] and hasattr(model, "coef_"):
                     try:
                         # SVM coef_ can be a sparse matrix OR a dense array depending on kernel
                         c = model.coef_.toarray() if hasattr(model.coef_, "toarray") else model.coef_
                         metadata["coef"] = c.tolist()
                         metadata["feature_names"] = X_train.columns.tolist()
                         metadata["intercept"] = model.intercept_.tolist() if hasattr(model, "intercept_") else 0
                     except Exception as e:
                         logger.debug(f"Could not extract coefficients for {model_type}: {e}")
            except Exception as meta_e:
                logger.warning(f"Failed to extract metadata for {model_type}: {meta_e}")

            return metrics, metadata, None
        except Exception as e:
            import traceback
            err_msg = str(e)
            if "MemoryError" in err_msg: err_msg = "Memory Exhausted"
            logger.error(f"Error training {model_type}: {e}")
            logger.debug(traceback.format_exc())
            return None, None, err_msg

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
            elif k in ["n_estimators", "max_depth", "max_iter", "min_samples_split", "n_neighbors"]:
                try: clean_p[k] = int(v)
                except: clean_p[k] = v
            elif k in ["C", "learning_rate", "var_smoothing"]:
                try: clean_p[k] = float(v)
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

            elif model_type == "knn":
                if "n_neighbors" not in clean_p:
                    clean_p["n_neighbors"] = min(15, max(3, int(np.sqrt(n_samples))))
            elif model_type == "dt":
                if "max_depth" not in clean_p:
                    clean_p["max_depth"] = min(12, max(4, int(np.log2(n_samples + 2))))
            elif model_type == "gb":
                if "n_estimators" not in clean_p:
                    clean_p["n_estimators"] = min(300, max(50, int(n_samples / 5)))
            elif model_type == "ab":
                if "n_estimators" not in clean_p:
                    clean_p["n_estimators"] = min(150, max(50, int(n_samples / 5)))

        # Normalise "lr" alias → "logistic" so both spellings work
        if model_type == "lr":
            model_type = "logistic"

        if task_type == "classification":
            if model_type == "logistic": return LogisticRegression(**clean_p)
            if model_type == "rf": return RandomForestClassifier(**clean_p)
            if model_type == "xgb": return XGBClassifier(**clean_p)
            if model_type == "mlp": return MLPClassifier(**clean_p)

            if model_type == "svm": return SVC(**clean_p)
            if model_type == "knn": return KNeighborsClassifier(**clean_p)
            if model_type == "nb": return GaussianNB(**clean_p)
            if model_type == "dt": return DecisionTreeClassifier(**clean_p)
            if model_type == "gb": return GradientBoostingClassifier(**clean_p)
            if model_type == "ab": return AdaBoostClassifier(**clean_p)
        else:
            if model_type == "linear" or model_type == "logistic":
                for k in ["penalty", "dual", "solver", "C", "multi_class", "l1_ratio"]:
                    clean_p.pop(k, None)
                return Ridge(**clean_p)
            if model_type == "rf": return RandomForestRegressor(**clean_p)
            if model_type == "xgb": return XGBRegressor(**clean_p)
            if model_type == "mlp": return MLPRegressor(**clean_p)
            
            if model_type == "svm": return SVR(**clean_p)
            if model_type == "knn": return KNeighborsRegressor(**clean_p)
            if model_type == "nb": 
                clean_p.pop("var_smoothing", None)
                return Ridge(**clean_p) # Naive Bayes doesn't support regression
            if model_type == "dt": return DecisionTreeRegressor(**clean_p)
            if model_type == "gb": return GradientBoostingRegressor(**clean_p)
            if model_type == "ab": return AdaBoostRegressor(**clean_p)
            
        return None
