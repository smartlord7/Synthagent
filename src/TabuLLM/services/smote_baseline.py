import numpy as np
import pandas as pd

from utils.logging_config import logger


def impute_for_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-imputes continuous columns, median-imputes integer-coded ones and
    mode-imputes the rest. SMOTE, CTGAN and TVAE cannot fit on NaN, so the
    baseline generators fit on this imputed copy of the training split.
    """
    out = df.copy()
    for col in out.columns:
        if not out[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            observed = out[col].dropna()
            # Integer-coded columns (counts, category codes) keep an integer fill so
            # imputation does not invent a value outside the column's domain.
            if len(observed) and np.allclose(observed, observed.round()):
                out[col] = out[col].fillna(observed.median().round())
            else:
                out[col] = out[col].fillna(observed.mean())
        else:
            mode = out[col].mode(dropna=True)
            out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "missing")
    return out


def generate_smote_rows(df_train: pd.DataFrame, target_col: str, categorical_cols: list, random_state: int, method: str = "smote") -> pd.DataFrame:
    """
    Runs SMOTENC (when `categorical_cols` is non-empty) or plain SMOTE on df_train,
    balancing the minority class(es) of `target_col` up to the majority count, and
    returns ONLY the newly synthesized rows -- the SMOTE analogue of the LLM pipeline's
    `generated_df`, letting the same fidelity/ML evaluation code run on either.

    Returns an empty DataFrame (with a logged warning) whenever SMOTE cannot run: the
    target has fewer than 2 classes, or the smallest class has too few rows to support
    any k_neighbors >= 1. Missing feature values are imputed first (impute_for_baseline),
    since imblearn rejects NaN.
    """
    from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, BorderlineSMOTE

    if target_col not in df_train.columns or df_train.empty:
        return pd.DataFrame(columns=df_train.columns)
    df_train = impute_for_baseline(df_train.dropna(subset=[target_col]))

    feature_cols = [c for c in df_train.columns if c != target_col]
    class_counts = df_train[target_col].value_counts()
    if len(class_counts) < 2:
        logger.warning("SMOTE: target has fewer than 2 classes; skipping.")
        return pd.DataFrame(columns=df_train.columns)

    min_class_count = int(class_counts.min())
    k_neighbors = min(5, min_class_count - 1)
    if k_neighbors < 1:
        logger.warning(f"SMOTE: smallest class has only {min_class_count} row(s); cannot compute neighbours, skipping.")
        return pd.DataFrame(columns=df_train.columns)

    # Encode categoricals to integer codes; remember the mapping to decode afterwards.
    encoders: dict = {}
    X = df_train[feature_cols].copy()
    for col in categorical_cols:
        if col in X.columns:
            codes, uniques = pd.factorize(X[col])
            X[col] = codes
            encoders[col] = uniques

    y = df_train[target_col]
    cat_indices = [feature_cols.index(c) for c in categorical_cols if c in feature_cols]

    try:
        if method == "adasyn":
            # ADASYN and Borderline-SMOTE have no categorical variant; the categoricals are
            # integer codes here and are rounded back to valid codes below.
            sampler = ADASYN(n_neighbors=k_neighbors, random_state=random_state)
        elif method == "borderline_smote":
            sampler = BorderlineSMOTE(k_neighbors=k_neighbors, random_state=random_state)
        elif cat_indices:
            sampler = SMOTENC(categorical_features=cat_indices, k_neighbors=k_neighbors, random_state=random_state)
        else:
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as e:
        logger.warning(f"SMOTE failed ({e}); skipping.")
        return pd.DataFrame(columns=df_train.columns)

    # imblearn preserves the original rows first, appending synthetic ones after --
    # slice off just the synthetic tail.
    n_original = len(df_train)
    if len(X_res) <= n_original:
        return pd.DataFrame(columns=df_train.columns)

    synth_X = pd.DataFrame(X_res).iloc[n_original:].reset_index(drop=True)
    synth_y = pd.Series(y_res).iloc[n_original:].reset_index(drop=True)

    # Decode categoricals back to their original labels.
    for col, uniques in encoders.items():
        n_uniques = len(uniques)
        synth_X[col] = (
            synth_X[col].round().astype(int).clip(0, n_uniques - 1).map(lambda i: uniques[i])
        )

    synth_df = synth_X.copy()
    synth_df[target_col] = synth_y.values
    return synth_df[list(df_train.columns)]
