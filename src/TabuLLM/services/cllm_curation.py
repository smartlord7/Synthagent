"""
CLLM-style curation (Seedat et al., ICML 2024) of LLM-generated records.

CLLM keeps only the synthetic records that a classifier trained on the real data learns
easily, using the learning dynamics of Data-IQ: an XGBoost model is trained on the real
training records, and for every synthetic record (x, y) the probability of its own label
y is read at several points of training. Records with a high average probability
(confidence) and a low variability across training (aleatoric uncertainty) are kept.
"""
import numpy as np
import pandas as pd

from utils.logging_config import logger

CONFIDENCE_MIN = 0.75     # Data-IQ upper confidence threshold
ALEATORIC_MAX = 0.2       # of a maximum of 0.25 for a probability
N_ESTIMATORS = 100
CHECKPOINTS = 10


def curate(generated_rows: list, df_train: pd.DataFrame, target_col: str, random_state: int = 0) -> "tuple[list, dict]":
    """Returns the kept rows and a small stats dict (kept, discarded)."""
    from xgboost import XGBClassifier

    if not generated_rows or target_col not in df_train.columns:
        return generated_rows, {"cllm_kept": len(generated_rows or []), "cllm_discarded": 0}
    gen = pd.DataFrame(generated_rows)
    feats = [c for c in df_train.columns if c != target_col]
    classes = sorted(df_train[target_col].dropna().unique().tolist())
    index = {c: i for i, c in enumerate(classes)}

    y_gen = pd.to_numeric(gen.get(target_col), errors="coerce")
    known = y_gen.isin(classes)
    X_tr = df_train[feats].apply(pd.to_numeric, errors="coerce")
    y_tr = df_train[target_col].map(index)
    X_gen = gen.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")

    try:
        model = XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3, learning_rate=0.1,
                              random_state=random_state, n_jobs=1, verbosity=0)
        model.fit(X_tr, y_tr)
        probs = []
        for k in np.linspace(N_ESTIMATORS / CHECKPOINTS, N_ESTIMATORS, CHECKPOINTS).astype(int):
            p = model.predict_proba(X_gen, iteration_range=(0, int(k)))
            lab = y_gen.map(index).fillna(0).astype(int).to_numpy()
            probs.append(p[np.arange(len(p)), lab])
        probs = np.vstack(probs)
        confidence = probs.mean(axis=0)
        aleatoric = (probs * (1 - probs)).mean(axis=0)
        keep = known.to_numpy() & (confidence >= CONFIDENCE_MIN) & (aleatoric <= ALEATORIC_MAX)
    except Exception as e:  # curation must never break a fold
        logger.warning(f"CLLM curation failed ({e}); keeping all records.")
        return generated_rows, {"cllm_kept": len(generated_rows), "cllm_discarded": 0}

    kept = [r for r, k in zip(generated_rows, keep) if k]
    return kept, {"cllm_kept": int(keep.sum()), "cllm_discarded": int((~keep).sum())}
