"""
Merges the separate run folders of the paper into results/ablation_20260918_114707, the
folder that generate_paper_assets.py reads. Idempotent: rerunning replaces what it merged
before. Run from src/Synthagent:

    python scripts/reproduce/merge_runs.py

Each source is (folder, arm in that folder, arm name in the merged results). The arm
folders are copied, the summary rows and checkpoint units are added under the new name.
"""
import json
import os
import shutil

import pandas as pd

MAIN = os.environ.get("TABULLM_MERGE_TARGET", "results/ablation_20260918_114707")
SOURCES = [
    # sensitivity of the reference configuration (original six datasets)
    *[(f"results/sens_{a}", a, a) for a in ("ctx_10", "ctx_40", "batch_15", "batch_60", "temp_0", "temp_10")],
    # the four new datasets, same arm names as on the original six
    ("results/new_full", "full", "full"),
    ("results/new_raw", "raw_prompting", "raw_prompting"),
    ("results/new_structured", "structured_output", "structured_output"),
    ("results/new_gpt41", "full", "backend_gpt_4_1"),
    ("results/new_baselines", "smote", "smote"),
    ("results/new_baselines", "ctgan", "ctgan"),
    ("results/new_baselines", "tvae", "tvae"),
    ("results/claude/sonnet46_newdata", "full", "backend_claude_sonnet_4_6"),
    # further baselines on all ten datasets
    ("results/cllm_old", "cllm", "cllm"),
    ("results/cllm_new", "cllm", "cllm"),
    ("results/copula_old", "gaussian_copula", "gaussian_copula"),
    ("results/copula_new", "gaussian_copula", "gaussian_copula"),
    *[(f"results/cpu_{p}", a, a) for p in ("old", "new") for a in ("adasyn", "borderline_smote", "cart", "arf")],
    # Bayesian network (pgmpy, CPU) on all ten datasets
    ("results/bn_all", "bayesian_network", "bayesian_network"),
    # GPU machine (merged only once pushed)
    ("results/gpu_baselines", "tabddpm", "tabddpm"),
]


def main():
    summary = pd.read_csv(os.path.join(MAIN, "ablation_summary.csv"))
    checkpoint = json.load(open(os.path.join(MAIN, "checkpoint.json")))
    added = []
    for folder, arm, new in SOURCES:
        ck_path = os.path.join(folder, "checkpoint.json")
        if not os.path.exists(ck_path):
            continue
        ck = json.load(open(ck_path))
        units = {k.split("::", 1)[1]: v for k, v in ck.get("completed", {}).items() if k.startswith(arm + "::")}
        if not units:
            continue
        # summary rows: replace this arm's rows for these datasets
        rows = pd.DataFrame([v["summary_row"] for v in units.values()])
        rows["arm"] = new
        summary = summary[~((summary.arm == new) & summary.dataset.isin(units))]
        summary = pd.concat([summary, rows], ignore_index=True)
        for dataset, v in units.items():
            checkpoint["completed"][f"{new}::{dataset}"] = v
        # per-repetition files and synthetic records
        src_arm = os.path.join(folder, arm)
        dst_arm = os.path.join(MAIN, new)
        os.makedirs(dst_arm, exist_ok=True)
        for name in os.listdir(src_arm) if os.path.isdir(src_arm) else []:
            if any(name.startswith(os.path.splitext(d)[0] + "_") for d in units):
                shutil.copy2(os.path.join(src_arm, name), os.path.join(dst_arm, name))
        syn_src = os.path.join(folder, "synthetic", arm)
        if os.path.isdir(syn_src):
            syn_dst = os.path.join(MAIN, "synthetic", new)
            os.makedirs(syn_dst, exist_ok=True)
            for name in os.listdir(syn_src):
                shutil.copy2(os.path.join(syn_src, name), os.path.join(syn_dst, name))
        added.append(f"{new} <- {folder}/{arm}: {len(units)} datasets")
    summary.to_csv(os.path.join(MAIN, "ablation_summary.csv"), index=False)
    json.dump(checkpoint, open(os.path.join(MAIN, "checkpoint.json"), "w"), indent=2)
    print("\n".join(added))


if __name__ == "__main__":
    main()
