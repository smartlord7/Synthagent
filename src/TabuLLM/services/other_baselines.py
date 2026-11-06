"""
Further generators that need no LLM and no GPU (CART, ARF), and two that are meant to
run on a GPU machine (TabDDPM via synthcity, GReaT via be_great). All fit on the imputed
training split of one fold and sample len(train) records, like CTGAN and TVAE.
"""
import numpy as np
import pandas as pd

from services.smote_baseline import impute_for_baseline
from services.deep_baselines import _discrete_columns
from utils.logging_config import logger


def _encode(df_train, target_col):
    """Imputes, then codes every text column as integers; returns the data and the codes."""
    data = impute_for_baseline(df_train.dropna(subset=[target_col]) if target_col else df_train)
    codes = {}
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            values, uniques = pd.factorize(data[col])
            data[col] = values
            codes[col] = uniques
    return data.apply(pd.to_numeric, errors="coerce").fillna(0), codes


def _decode(synth, codes):
    """Maps integer codes of text columns back to their labels."""
    for col, uniques in codes.items():
        idx = pd.to_numeric(synth[col], errors="coerce").round().clip(0, len(uniques) - 1)
        synth[col] = [uniques[int(i)] if pd.notna(i) else uniques[0] for i in idx]
    return synth


def _prepare(df_train, target_col):
    return _encode(df_train, target_col)[0]


def _restore_integers(synth, data):
    for col in data.columns:
        observed = data[col].dropna()
        if len(observed) and np.allclose(observed, observed.round()):
            synth[col] = pd.to_numeric(synth[col], errors="coerce").round()
    return synth


def generate_cart_rows(df_train, target_col, categorical_cols, random_state, n_rows=None, min_leaf=5):
    """
    Sequential CART synthesis, as in the R package synthpop: the first column is drawn
    from its observed values, and each following column is predicted from the columns
    already synthesized by a decision tree fitted on the real records; the synthetic value
    is drawn at random from the real values that fall in the same leaf.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if df_train.empty:
        return pd.DataFrame(columns=df_train.columns)
    data, codes = _encode(df_train, target_col)
    n_rows = int(n_rows or len(data))
    rng = np.random.default_rng(random_state)
    discrete = set(_discrete_columns(data, target_col, categorical_cols))
    order = [target_col] + [c for c in data.columns if c != target_col] if target_col else list(data.columns)

    synth = pd.DataFrame(index=range(n_rows))
    synth[order[0]] = rng.choice(data[order[0]].to_numpy(), size=n_rows, replace=True)
    for i, col in enumerate(order[1:], start=1):
        prev = order[:i]
        Tree = DecisionTreeClassifier if col in discrete else DecisionTreeRegressor
        tree = Tree(min_samples_leaf=min_leaf, random_state=random_state)
        tree.fit(data[prev].to_numpy(), data[col].to_numpy())
        real_leaf = tree.apply(data[prev].to_numpy())
        synth_leaf = tree.apply(synth[prev].to_numpy())
        pools = {leaf: data[col].to_numpy()[real_leaf == leaf] for leaf in np.unique(real_leaf)}
        synth[col] = [rng.choice(pools[leaf]) for leaf in synth_leaf]
    return _decode(synth, codes)[list(df_train.columns)]


def generate_arf_rows(df_train, target_col, categorical_cols, random_state, n_rows=None):
    """Adversarial random forests (Watson et al., AISTATS 2023), arfpy defaults."""
    if not hasattr(np, "in1d"):  # arfpy still calls np.in1d, removed in NumPy 2
        np.in1d = np.isin
    from arfpy import arf

    if df_train.empty:
        return pd.DataFrame(columns=df_train.columns)
    data, codes = _encode(df_train, target_col)
    discrete = _discrete_columns(data, target_col, categorical_cols)
    frame = data.copy()
    for col in discrete:
        frame[col] = frame[col].astype("category")
    n_rows = int(n_rows or len(data))
    np.random.seed(random_state)
    try:
        model = arf.arf(x=frame, random_state=random_state)
        model.forde()
        synth = model.forge(n=n_rows)
    except Exception as e:
        logger.warning(f"ARF failed ({e}); skipping.")
        return pd.DataFrame(columns=df_train.columns)
    for col in discrete:
        synth[col] = pd.to_numeric(synth[col].astype(str), errors="coerce")
    return _decode(_restore_integers(synth, data), codes)[list(df_train.columns)]


def generate_tabddpm_rows(df_train, target_col, categorical_cols, random_state, n_rows=None):
    """TabDDPM through synthcity's "ddpm" plugin, library defaults. Meant for a GPU."""
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    data, codes = _encode(df_train, target_col)
    n_rows = int(n_rows or len(data))
    try:
        model = Plugins().get("ddpm", random_state=random_state)
        model.fit(GenericDataLoader(data, target_column=target_col))
        synth = model.generate(count=n_rows).dataframe()
    except Exception as e:
        logger.warning(f"TabDDPM failed ({e}); skipping.")
        return pd.DataFrame(columns=df_train.columns)
    return _decode(_restore_integers(synth, data), codes)[list(df_train.columns)]


def generate_great_rows(df_train, target_col, categorical_cols, random_state, n_rows=None, epochs=100):
    """GReaT (Borisov et al., ICLR 2023) with distilgpt2, be_great defaults apart from epochs. Meant for a GPU."""
    from be_great import GReaT

    data, codes = _encode(df_train, target_col)
    n_rows = int(n_rows or len(data))
    try:
        model = GReaT(llm="distilgpt2", epochs=epochs, batch_size=32)
        model.fit(data)
        synth = model.sample(n_samples=n_rows)
    except Exception as e:
        logger.warning(f"GReaT failed ({e}); skipping.")
        return pd.DataFrame(columns=df_train.columns)
    return _decode(_restore_integers(synth.apply(pd.to_numeric, errors="coerce"), data), codes)[list(df_train.columns)]


def generate_bayesian_network_rows(df_train, target_col, categorical_cols, random_state, n_rows=None):
    """Bayesian network through synthcity's "bayesian_network" plugin, library defaults."""
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    data, codes = _encode(df_train, target_col)
    n_rows = int(n_rows or len(data))
    try:
        model = Plugins().get("bayesian_network", random_state=random_state)
        model.fit(GenericDataLoader(data, target_column=target_col))
        synth = model.generate(count=n_rows).dataframe()
    except Exception as e:
        logger.warning(f"Bayesian network failed ({e}); skipping.")
        return pd.DataFrame(columns=df_train.columns)
    return _decode(_restore_integers(synth, data), codes)[list(df_train.columns)]


GENERATORS = {
    "cart": generate_cart_rows,
    "arf": generate_arf_rows,
    "tabddpm": generate_tabddpm_rows,
    "great": generate_great_rows,
    "bayesian_network": generate_bayesian_network_rows,
}
