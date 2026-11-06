"""
Downloads the four additional UCI datasets with ucimlrepo and writes them to
data/medical/ exactly as used in the paper. Run from src/TabuLLM:

    pip install ucimlrepo
    python scripts/reproduce/prepare_uci_datasets.py

Targets are recoded to 0/1 (1 = disease or death) so that every classifier, XGBoost
included, accepts them; binary categorical features are coded 0/1; ordinal codes of the
thoracic-surgery data (DGN, PRE6, PRE14) keep their number. Missing values are kept and
imputed by the pipeline.
"""
import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "medical")


def fetch(uci_id):
    d = fetch_ucirepo(id=uci_id)
    df = d.data.features.copy()
    for col in d.data.targets.columns:
        df[col] = d.data.targets[col]
    return df


def main():
    c = fetch(451)                                   # Breast Cancer Coimbra
    c["Classification"] = (c["Classification"] == 2).astype(int)
    c.to_csv(os.path.join(OUT, "breast_cancer_coimbra.csv"), index=False)

    h = fetch(46)                                    # Hepatitis
    h["Class"] = (h["Class"] == 1).astype(int)       # 1 = died
    h.to_csv(os.path.join(OUT, "hepatitis.csv"), index=False)

    k = fetch(336)                                   # Chronic Kidney Disease
    for col in k.columns:
        if k[col].dtype == object:
            k[col] = k[col].astype(str).str.strip().replace({"nan": np.nan})
    k["class"] = (k["class"] == "ckd").astype(int)
    codes = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1, "no": 0, "yes": 1, "good": 0, "poor": 1}
    for col in ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]:
        k[col] = k[col].map(codes)
    k.to_csv(os.path.join(OUT, "chronic_kidney_disease.csv"), index=False)

    t = fetch(277)                                   # Thoracic Surgery
    for col in ["DGN", "PRE6", "PRE14"]:
        t[col] = t[col].str.extract(r"(\d+)").astype(int)
    for col in t.columns:
        if t[col].dtype == object:
            t[col] = (t[col] == "T").astype(int)
    t.to_csv(os.path.join(OUT, "thoracic_surgery.csv"), index=False)


if __name__ == "__main__":
    main()
