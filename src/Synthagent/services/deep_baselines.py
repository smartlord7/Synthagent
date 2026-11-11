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

    # Restore integer-valued columns (int dtypes, and float columns such as codes that
    # only hold whole numbers) so discrete codes compare equal to the real values.
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        observed = data[col].dropna()
        if pd.api.types.is_integer_dtype(data[col]):
            synth[col] = pd.to_numeric(synth[col], errors="coerce").round().astype("Int64").astype(data[col].dtype, errors="ignore")
        elif len(observed) and np.allclose(observed, observed.round()):
            synth[col] = pd.to_numeric(synth[col], errors="coerce").round()
    return synth[list(df_train.columns)]


def generate_gaussian_copula_rows(
    df_train: pd.DataFrame,
    target_col: str,
    categorical_cols: list,
    random_state: int,
    n_rows: "int | None" = None,
) -> pd.DataFrame:
    """
    Gaussian copula baseline: every column is mapped to normal scores through its
    empirical distribution, the correlation of the scores is estimated, new scores are
    sampled from that multivariate normal and mapped back through the empirical
    quantiles of each column. Discrete and integer columns keep their observed values.
    """
    from scipy.stats import norm, rankdata

    if df_train.empty:
        return pd.DataFrame(columns=df_train.columns)
    from services.other_baselines import _encode, _decode
    data, codes = _encode(df_train, target_col)
    n_rows = int(n_rows or len(df_train))
    rng = np.random.default_rng(random_state)

    scores = np.column_stack([norm.ppf(rankdata(data[c]) / (len(data) + 1)) for c in data.columns])
    corr = np.corrcoef(scores, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)                       # repair to positive semi-definite
    corr = (v * np.clip(w, 1e-6, None)) @ v.T
    z = rng.multivariate_normal(np.zeros(len(data.columns)), corr, size=n_rows)
    u = norm.cdf(z)

    discrete = set(_discrete_columns(data, target_col, categorical_cols))
    synth = {}
    for j, col in enumerate(data.columns):
        values = np.sort(data[col].to_numpy())
        if col in discrete:
            synth[col] = values[np.minimum((u[:, j] * len(values)).astype(int), len(values) - 1)]
        else:
            synth[col] = np.quantile(values, u[:, j])
            if np.allclose(values, values.round()):
                synth[col] = np.round(synth[col])
    return _decode(pd.DataFrame(synth), codes)[list(df_train.columns)]
