import numpy as np
import pandas as pd

from services.smote_baseline import impute_for_baseline
from utils.logging_config import logger

# Integer columns with at most this many distinct values are modelled as discrete.
MAX_DISCRETE_LEVELS = 10


def _discrete_columns(df: pd.DataFrame, target_col: str, categorical_cols: list) -> list:
    discrete = [c for c in categorical_cols if c in df.columns]
    for col in df.columns:
        if col in discrete:
            continue
        values = df[col].dropna()
        is_integral = pd.api.types.is_integer_dtype(df[col]) or (
            pd.api.types.is_float_dtype(df[col]) and np.allclose(values, values.round())
        )
        if col == target_col or (is_integral and values.nunique() <= MAX_DISCRETE_LEVELS):
            discrete.append(col)
    return discrete


def generate_deep_rows(
    df_train: pd.DataFrame,
    target_col: str,
    categorical_cols: list,
    random_state: int,
    method: str = "ctgan",
    n_rows: "int | None" = None,
    epochs: int = 300,
) -> pd.DataFrame:
    """
    Fits CTGAN or TVAE (the `ctgan` package, library defaults apart from `epochs`) on
    the imputed training split and samples `n_rows` rows (default: len(df_train), the
    same volume the LLM arms request). Returns an empty DataFrame if fitting fails.
    """
    import torch
    from ctgan import CTGAN, TVAE

    if df_train.empty:
        return pd.DataFrame(columns=df_train.columns)

    data = impute_for_baseline(df_train.dropna(subset=[target_col]) if target_col else df_train)
    discrete = _discrete_columns(data, target_col, categorical_cols)
    n_rows = int(n_rows or len(df_train))

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    try:
        model = CTGAN(epochs=epochs) if method == "ctgan" else TVAE(epochs=epochs)
        model.set_random_state(random_state)
        model.fit(data, discrete_columns=discrete)
        synth = model.sample(n_rows)
    except Exception as e:
        logger.warning(f"{method.upper()} failed ({e}); skipping.")
        return pd.DataFrame(columns=df_train.columns)

    # Restore integer dtypes so discrete codes compare equal to the real values.
    for col in data.columns:
        if pd.api.types.is_integer_dtype(data[col]):
            synth[col] = pd.to_numeric(synth[col], errors="coerce").round().astype("Int64").astype(data[col].dtype, errors="ignore")
    return synth[list(df_train.columns)]
