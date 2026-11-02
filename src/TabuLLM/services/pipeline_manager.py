import multiprocessing
import time
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, false_discovery_control
from sklearn.model_selection import train_test_split
from services.llm_client import generate_bulk_data
from services.data_manager import select_context_rows
from utils.metrics import calculate_metrics, calculate_metrics_by_class, compute_c2st_accuracy, compute_mia_attack_auc, compute_schema_validity
from utils.logging_config import logger
from services.ml_evaluator import MLEvaluator
from services.rule_inference import RuleInference
from services.auditor import DatasetAuditor


def _paired_pvalue(a: list, b: list):
    """Wilcoxon signed-rank p-value for paired fold measurements a vs b, or None if not computable."""
    diffs = [y - x for x, y in zip(a, b)]
    if len(diffs) < 3 or all(d == 0 for d in diffs):
        return None
    try:
        return float(wilcoxon(diffs).pvalue)
    except Exception:
        return None


def apply_fdr_correction(pvalue_refs: list, get_fn, set_fn) -> None:
    """
    Benjamini-Hochberg FDR correction applied across one family of hypothesis tests.

    Running many Wilcoxon tests (one per model x metric x comparison) and reading each
    raw p-value in isolation inflates the false-discovery rate -- some "significant"
    deltas are expected by chance alone once dozens of tests are run together. This
    collects every raw p-value the caller points at via `get_fn`, corrects them jointly,
    and writes the corrected q-values back via `set_fn`.

    Args:
        pvalue_refs: opaque keys identifying each hypothesis test (whatever get_fn/set_fn expect).
        get_fn: callable(ref) -> raw p-value or None.
        set_fn: callable(ref, qvalue) -> None, invoked only for refs with a non-None p-value.
    """
    raw = [(ref, get_fn(ref)) for ref in pvalue_refs]
    raw = [(ref, p) for ref, p in raw if p is not None]
    if not raw:
        return
    refs, pvalues = zip(*raw)
    qvalues = false_discovery_control(list(pvalues), method="bh")
    for ref, q in zip(refs, qvalues):
        set_fn(ref, float(q))


def _compute_dataset_audit(df_orig_all: pd.DataFrame, targets: list, filename: str) -> dict:
    """
    One-time Health & Ethics audit for a dataset (structural/fairness/privacy/quality/
    proxies/quasi-identifiers) -- computed once per dataset and reused across every fold.
    Shared by the LLM benchmark path and the SMOTE baseline path so both report an
    identical audit for the same dataset.
    """
    try:
        auditor_service = DatasetAuditor()
        target_col = targets[0] if targets else None
        sensitive_cols = DatasetAuditor.detect_sensitive_columns(df_orig_all)
        return {
            "structural": auditor_service.get_structural_metrics(df_orig_all),
            "outliers": auditor_service.get_outlier_metrics(df_orig_all),
            "fairness": auditor_service.get_fairness_metrics(df_orig_all, target_col, sensitive_cols),  # type: ignore[arg-type]
            "privacy": auditor_service.get_privacy_metrics(df_orig_all),
            "proxies": auditor_service.get_proxy_risks(df_orig_all, sensitive_cols),
            "quality": auditor_service.get_quality_metrics(df_orig_all, target_col),  # type: ignore[arg-type]
            "quasi_identifiers": auditor_service.get_quasi_identifiers(df_orig_all),
        }
    except Exception as e:
        logger.warning(f"Dataset audit failed for {filename}: {e}")
        return {}


def _aggregate_overall_ml(results: list) -> dict:
    """
    Builds the overall_ml summary (mean/std per arm, deltas, paired Wilcoxon p-values,
    Benjamini-Hochberg-corrected q-values) from a dataset's per-fold `results` list.
    Shared by the LLM benchmark path and the SMOTE baseline path so both arms aggregate
    identically and stay directly comparable.
    """
    overall_ml: dict = {}
    if not results:
        return overall_ml

    all_models = set()
    for r in results: all_models.update(r.get("ml_eval", {}).keys())
    for m_type in all_models:
        overall_ml[m_type] = {}
        for m_name in ["accuracy", "f1", "rmse", "r2"]:
            # Original/generated/augmented are paired independently per comparison rather
            # than requiring all three to succeed together in the same fold: e.g. SMOTE's
            # "generated" arm is inherently single-class (SMOTE only ever creates
            # minority-class synthetic rows), so MLEvaluator correctly refuses to train a
            # classifier on it -- that must not blank out the still-valid orig-vs-aug
            # comparison. Each delta below is computed from its own fold-index-paired subset.
            f_orig_any, f_gen_any, f_aug_any = [], [], []
            paired_orig_aug: tuple = ([], [])
            paired_orig_gen: tuple = ([], [])
            for r in results:
                me = r.get("ml_eval", {}).get(m_type)
                if not me: continue
                o = (me.get("original") or {}).get(m_name)
                g = (me.get("generated") or {}).get(m_name)
                a = (me.get("augmented") or {}).get(m_name)
                if o is not None: f_orig_any.append(o)
                if g is not None: f_gen_any.append(g)
                if a is not None: f_aug_any.append(a)
                if o is not None and a is not None:
                    paired_orig_aug[0].append(o); paired_orig_aug[1].append(a)
                if o is not None and g is not None:
                    paired_orig_gen[0].append(o); paired_orig_gen[1].append(g)

            if f_orig_any:
                entry: dict = {}
                m_o, s_o = np.mean(f_orig_any), np.std(f_orig_any)
                entry["real"], entry["real_sd"] = m_o, s_o
                entry["display_real"] = f"{m_o:.3f} ± {s_o:.2f}"

                if f_gen_any:
                    m_g, s_g = np.mean(f_gen_any), np.std(f_gen_any)
                    entry["gen"], entry["gen_sd"] = m_g, s_g
                    entry["display_gen"] = f"{m_g:.3f} ± {s_g:.2f}"
                if f_aug_any:
                    m_a, s_a = np.mean(f_aug_any), np.std(f_aug_any)
                    entry["aug"], entry["aug_sd"] = m_a, s_a
                    entry["display_aug"] = f"{m_a:.3f} ± {s_a:.2f}"

                po_aug, pa = paired_orig_aug
                if po_aug:
                    delta_aug = [a - o for a, o in zip(pa, po_aug)]
                    m_d, s_d = np.mean(delta_aug), np.std(delta_aug)
                    entry["delta"], entry["delta_sd"] = m_d, s_d
                    entry["display_delta"] = f"{m_d:+.3f} ± {s_d:.2f}"
                    entry["pvalue_aug_vs_real"] = _paired_pvalue(po_aug, pa)

                po_gen, pg = paired_orig_gen
                if po_gen:
                    delta_gen = [g - o for g, o in zip(pg, po_gen)]
                    m_dg, s_dg = np.mean(delta_gen), np.std(delta_gen)
                    entry["delta_gen"], entry["delta_gen_sd"] = m_dg, s_dg
                    entry["display_delta_gen"] = f"{m_dg:+.3f} ± {s_dg:.2f}"
                    entry["pvalue_gen_vs_real"] = _paired_pvalue(po_gen, pg)

                overall_ml[m_type][m_name] = entry

            overall_ml[m_type]["folds_metadata"] = [r["ml_eval"][m_type].get("augmented_metadata") for r in results if m_type in r.get("ml_eval", {}) and r["ml_eval"][m_type].get("augmented_metadata") is not None]
            if not f_orig_any:
                errors = [r["ml_eval"][m_type].get("error") for r in results if m_type in r.get("ml_eval", {}) and r["ml_eval"][m_type].get("error")]
                if errors: overall_ml[m_type]["error"] = errors[0]

    # Multiple-comparison correction (Benjamini-Hochberg FDR): every model x metric x
    # arm-comparison above is a separate hypothesis test for this dataset. Correcting
    # them jointly avoids inflating the false-discovery rate that comes from reading
    # dozens of raw p-values in isolation. Adds qvalue_* alongside the raw pvalue_*.
    _pvalue_keys = [
        (m_type, m_name, key)
        for m_type, metrics in overall_ml.items()
        for m_name, m in metrics.items()
        if isinstance(m, dict)
        for key in ("pvalue_aug_vs_real", "pvalue_gen_vs_real")
    ]
    apply_fdr_correction(
        _pvalue_keys,
        get_fn=lambda ref: overall_ml[ref[0]][ref[1]].get(ref[2]),
        set_fn=lambda ref, q: overall_ml[ref[0]][ref[1]].__setitem__(ref[2].replace("pvalue", "qvalue"), q),
    )
    return overall_ml


def _structured_output_format(df_real: pd.DataFrame) -> "tuple[dict, dict]":
    """
    JSON schema for the structured-output baseline: an object {"records": [...]} whose
    items have exactly the real columns, numerical columns as numbers and other columns
    restricted to their observed values. Property names are sanitized because some
    providers only accept [A-Za-z0-9_-]; the second return value maps them back.
    """
    import re
    props, back, used = {}, {}, set()
    for col in df_real.columns:
        key = re.sub(r"[^A-Za-z0-9_-]", "_", str(col)) or "col"
        base, i = key, 1
        while key in used:
            i += 1
            key = f"{base}_{i}"
        used.add(key)
        back[key] = col
        if pd.api.types.is_numeric_dtype(df_real[col]):
            props[key] = {"type": "number"}
        else:
            values = sorted(df_real[col].dropna().astype(str).unique().tolist())
            props[key] = {"type": "string", "enum": values} if 0 < len(values) <= 500 else {"type": "string"}
    schema = {
        "type": "object",
        "properties": {"records": {"type": "array", "items": {
            "type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}}},
        "required": ["records"], "additionalProperties": False,
    }
    fmt = {"type": "json_schema", "json_schema": {"name": "synthetic_records", "strict": True, "schema": schema}}
    return fmt, back


def _align_raw_rows(rows: list, df_real: pd.DataFrame) -> "tuple[list, int]":
    """
    Evaluation-side alignment for the no-controller arm: keeps only columns whose name
    matches the real table exactly (others become missing) and coerces numerical columns
    with pd.to_numeric. Returns the rows and the number of non-empty cells that failed to
    coerce, which the controlled pipeline would have repaired or flagged.
    """
    if not rows:
        return [], 0
    df = pd.DataFrame(rows).reindex(columns=df_real.columns)
    type_errors = 0
    for col in df_real.select_dtypes(include=[np.number]).columns:
        before = df[col].notna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        type_errors += int((before & df[col].isna()).sum())
    return df.to_dict("records"), type_errors


def run_single_dataset_benchmark(args):
    """
    Worker function for multiprocessing. 
    args: (dataset_id, filename, orig_data, col_defs, config)
    """
    dataset_id, filename, orig_data, col_defs, config = args
    logger.info(f"Starting pipeline benchmark: {filename}")
    
    folds = config.get("folds", 3)
    strategy = config.get("strategy", "random")
    algorithm = config.get("algorithm", "google/gemma-3-4b")
    temperature = config.get("temperature", 0.3)
    chunk_size = config.get("chunk_size", 30)
    instructions = config.get("instructions", "")
    desc_metadata = config.get("desc_metadata", "")
    ml_config = config.get("ml_config", {})
    targets = config.get("targets", [])
    na_strategy = config.get("na_strategy", "impute")
    rules_metadata = config.get("rules_metadata", None)
    enforce_rules = config.get("enforce_rules", True)
    enforce_boundaries = config.get("enforce_boundaries", False)
    inject_inferred_rules = bool(config.get("inject_inferred_rules", True))
    structured_output = bool(config.get("structured_output", False))
    # The structured-output baseline is the no-controller path with schema-constrained decoding.
    raw_prompting = bool(config.get("raw_prompting", False)) or structured_output
    apply_post_processing = bool(config.get("apply_post_processing", True)) and not raw_prompting

    progress_dict = config.get("_progress_dict")
    dataset_id_key = config.get("_dataset_id")

    if progress_dict is not None and dataset_id_key:
        try:
            p = dict(progress_dict[dataset_id_key])
            p["status"] = "Initializing..."
            p["done"] = 0
            progress_dict[dataset_id_key] = p
        except: pass

    df_orig_all = pd.DataFrame(orig_data)
    if '_row_id' in df_orig_all.columns:
        df_orig_all.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
    
    field_map = {c['field']: c.get('headerName', c['field']) for c in col_defs if c['field'] not in ['ai_audit', '_row_id']}
    df_orig_all = df_orig_all.rename(columns=field_map)

    # --- GENERATION PLAN (computed once per dataset, reused across every fold) ---
    # Both the inferred rules and the audit only depend on df_orig_all, which is
    # identical across folds -- recomputing them per fold was pure waste.
    inferred_rules = RuleInference.infer_dataset_rules(df_orig_all) if inject_inferred_rules else []
    rules_hierarchy = {
        "integrity": [r for r in inferred_rules if r['level'] == 1],
        "boundaries": [r for r in inferred_rules if r['level'] == 2],
        "user": rules_metadata if isinstance(rules_metadata, list) else [],
        "enforce_user": bool(enforce_rules),
        "enforce_boundaries": bool(enforce_boundaries)
    }

    dataset_audit = _compute_dataset_audit(df_orig_all, targets, filename)

    # Fold-level resume support: `_resume_fold_results` (int fold_num -> that fold's
    # already-computed result dict, from a prior interrupted run's checkpoint) lets folds
    # already on disk be reused verbatim instead of re-generating from the LLM. `_on_fold_done`
    # (fold_num, result_dict) -> None is invoked right after each NEWLY computed fold, so a
    # caller can persist it immediately -- rather than only after every fold in this dataset
    # finishes -- so a crash/interrupt mid-dataset only loses the ONE fold in flight. Both are
    # no-ops (empty dict / None) for a normal run with no resumption in play.
    resume_fold_results = config.get("_resume_fold_results") or {}
    on_fold_done = config.get("_on_fold_done")

    results = []
    all_synthetic_data = []
    all_augmented_data = []

    for fold_num in range(1, folds + 1):
        if fold_num in resume_fold_results:
            m = resume_fold_results[fold_num]
            results.append(m)
            all_synthetic_data.extend(m.get("synthetic_data", []) or [])
            all_augmented_data.extend(
                (m.get("context_data", []) or []) + (m.get("synthetic_data", []) or []) + (m.get("test_data", []) or [])
            )
            logger.info(f"[{filename} | Fold {fold_num}] Reusing cached result from checkpoint -- skipping regeneration.")
            continue

        if progress_dict is not None:
             p = progress_dict[dataset_id_key]
             p["status"] = f"Fold {fold_num}/{folds} - Data Prep"
             progress_dict[dataset_id_key] = p

        try:
            fold_start_time = time.time()
            # 1. Split Original into Train/Test (80/20) for this fold
            df_train_orig, df_test_orig = train_test_split(df_orig_all, test_size=0.2, random_state=42 + fold_num)
            logger.info(f"[DATA] [{filename} | Fold {fold_num}] Split Complete: Train={len(df_train_orig)}, Test={len(df_test_orig)}")
            
            # 2. Setup context/augmentation split
            join_context = config.get("join_context", True)
            ctx_size_pct = config.get("ctx_size_pct", 20)
            gen_size_pct = config.get("gen_size_pct", 80)

            # Context is sampled from the train split (80% of original)
            n_context_rows = max(1, int(len(df_train_orig) * (ctx_size_pct / 100)))
            # Generated size is also relative to the train split size to maintain scale
            num_empty = max(1, int(len(df_train_orig) * (gen_size_pct / 100)))

            # Honour explicit num_empty if caller passed a positive value (overrides % default)
            explicit_num_empty = config.get("num_empty", 0)
            if explicit_num_empty and int(explicit_num_empty) > 0:
                num_empty = int(explicit_num_empty)
            
            logger.info(f"[DATA] [{filename} | Fold {fold_num}] Strategy: {strategy} | Join: {join_context} | Context: {n_context_rows} | Target Gen: {num_empty}")
            context_data_list = select_context_rows(df_train_orig, strategy=strategy, n_rows=n_context_rows) or []
            context_dict = {"metadata": desc_metadata, "manual_ctx": "[]", "global_ctx": context_data_list}
            
            # Isolate rows that are in the training split but NOT sent to the LLM as context
            # This provides a 100% complete trace of the original dataset's role per fold.
            try:
                # If _row_id is available (standard), we use it to differntiate
                if '_row_id' in df_train_orig.columns:
                    ctx_ids = [row.get('_row_id') for row in context_data_list]
                    df_non_ctx = df_train_orig[~df_train_orig['_row_id'].isin(ctx_ids)]
                    train_non_context_data = df_non_ctx.to_dict('records')
                else:
                    train_non_context_data = []
            except:
                train_non_context_data = []

            # 3. Generate Augmented Data
            if progress_dict is not None and dataset_id_key:
                 try:
                     # Refresh dict to force sync
                     p = dict(progress_dict[dataset_id_key])
                     p["status"] = f"Fold {fold_num}/{folds}: LLM Gen..."
                     progress_dict[dataset_id_key] = p
                 except: pass
                 
            logger.info(f"[{filename} | Fold {fold_num}] Pre-Generation Geometry -> Train: {len(df_train_orig)} | Context: {n_context_rows} | Target Gen: {num_empty}")

            # Rule hierarchy + audit are computed once per dataset (see above the fold loop)
            # and reused unchanged across every fold.
            all_fields = list(df_orig_all.columns)
            response_format, key_map = _structured_output_format(df_orig_all) if structured_output else (None, {})
            gen_start_time = time.time()
            result_bundle = generate_bulk_data(
                [], context_dict, all_fields, all_fields, algorithm, instructions, temperature,
                num_empty=num_empty, chunk_size=chunk_size,
                rules_metadata=rules_hierarchy, enforce_rules=True, raw_mode=raw_prompting,
                response_format=response_format
            )
            if result_bundle and key_map:
                result_bundle["data"] = [{key_map.get(k, k): v for k, v in row.items()}
                                         for row in result_bundle["data"]]
            generation_seconds = time.time() - gen_start_time

            if result_bundle:
                from services.post_processor import PostProcessor

                # Extract and then apply active rules (noise, rounding, etc.) via code post-processing
                generated_rows = result_bundle["data"]
                if apply_post_processing:
                    generated_rows = PostProcessor.apply_active_rules(generated_rows, rules_hierarchy.get("user", []))
                if raw_prompting:
                    # A fold in which no request returned a completion measures the provider
                    # (quota, rate limit, network), not the model's output. Fail it so the unit
                    # is not checkpointed as complete and --resume retries it.
                    raw_stats = result_bundle.get("stats", {})
                    if raw_stats.get("total_tokens", 0) == 0:
                        raise RuntimeError(
                            f"raw_prompting fold {fold_num}: no LLM request succeeded "
                            f"(request_errors={raw_stats.get('request_errors', 0)})"
                        )
                    generated_rows, type_errors = _align_raw_rows(generated_rows, df_orig_all)
                    result_bundle.setdefault("stats", {})["type_errors"] = type_errors
                generation_stats = {**result_bundle.get("stats", {}), "generation_seconds": generation_seconds}
                test_data_list = df_test_orig.to_dict('records')
                
                for row in generated_rows: row["_fold_id"] = fold_num
                for row in context_data_list: row["_fold_id"] = fold_num
                for row in test_data_list: row["_fold_id"] = fold_num
                
                all_synthetic_data.extend(generated_rows)
                all_augmented_data.extend(context_data_list + generated_rows + test_data_list)

                generated_df = pd.DataFrame(generated_rows)

                if join_context:
                    df_aug_train = pd.concat([pd.DataFrame(context_data_list), generated_df], ignore_index=True)
                    test_df_eval = df_test_orig # Standard 20% test
                else:
                    df_aug_train = generated_df
                    test_df_eval = df_orig_all # Evaluate against whole real dataset

                logger.info(f"[{filename} | Fold {fold_num}] Post-Generation Geometry -> Target: {num_empty} | Generated: {len(generated_rows)} | Final Train Size: {len(df_aug_train)} | Test Size: {len(test_df_eval)}")

                # Fidelity Metrics — blended (context + generated) against the full original,
                # kept for backward compatibility with the appendix/report sampling.
                m = calculate_metrics(df_orig_all, df_aug_train)
                # Fidelity Metrics — generated rows only, isolated from real context blending.
                # This is the clean signal used by ablation studies to compare modules.
                gen_fidelity = calculate_metrics(df_orig_all, generated_df)
                m.update({f"Gen {k}": v for k, v in gen_fidelity.items()})

                target_col = targets[0] if targets else None
                task_type = MLEvaluator.detect_task_type(df_orig_all, targets) if targets else None

                # Per-class fidelity (minority vs majority), on the generated-only corpus --
                # isolates whether synthetic fidelity holds up on the minority class
                # specifically, the class an augmentation pipeline is meant to help on an
                # imbalanced dataset, rather than only the pooled/aggregate signal above.
                if target_col and task_type == "classification":
                    m["fidelity_by_class"] = calculate_metrics_by_class(df_orig_all, generated_df, target_col)

                # Privacy/discriminability attacks against the generated-only corpus:
                #   C2ST   -- can a classifier tell real (df_train_orig) from synthetic
                #             rows apart? ~0.5 accuracy = indistinguishable (good fidelity).
                #   MIA    -- distance-based membership-inference attack: does the
                #             synthetic corpus sit closer to the real rows it was
                #             generated from (members) than to held-out rows it never
                #             saw (non-members)? ~0.5 AUC = no membership leakage signal.
                c2st_accuracy = compute_c2st_accuracy(df_train_orig, generated_df)
                mia_auc = compute_mia_attack_auc(df_train_orig, df_test_orig, generated_df)
                if c2st_accuracy is not None:
                    m["Gen C2ST Accuracy"] = c2st_accuracy
                if mia_auc is not None:
                    m["Gen MIA Attack AUC"] = mia_auc
                validity = compute_schema_validity(df_orig_all, generated_df)
                if validity is not None:
                    m["Gen Valid Rows"] = validity

                # ML Downstream Metrics: three arms, always evaluated on the fold's fixed
                # 20% holdout (df_test_orig) so ablation arms that toggle join_context
                # remain comparable to the baseline on the same test set.
                #   - original:  trained on the real train split
                #   - generated: trained on the synthetic rows only
                #   - augmented: trained on real train split + synthetic rows
                ml_fold_results = {}
                if ml_config and targets:
                    df_full_aug = pd.concat([df_train_orig, generated_df], ignore_index=True)
                    for m_type in ml_config.get("selected_models", []):
                        if progress_dict is not None:
                             p = progress_dict[dataset_id_key]
                             p["status"] = f"Fold {fold_num}/{folds} - Eval {m_type.upper()}"
                             progress_dict[dataset_id_key] = p

                        m_params = ml_config.get("params", {}).get(m_type, {})
                        logger.info(f"[ML] [{filename} | Fold {fold_num}] Evaluating {m_type.upper()} (NA Strategy: {na_strategy})")

                        orig_perf, _,        orig_err = MLEvaluator.train_and_evaluate(df_train_orig, targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                        gen_perf,  gen_meta,  gen_err  = MLEvaluator.train_and_evaluate(generated_df,  targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                        aug_perf,  aug_meta,  aug_err  = MLEvaluator.train_and_evaluate(df_full_aug,   targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)

                        ml_fold_results[m_type] = {
                            "original": orig_perf,
                            "generated": gen_perf, "generated_metadata": gen_meta,
                            "augmented": aug_perf, "augmented_metadata": aug_meta,
                            "task_type": task_type, "error": aug_err or gen_err or orig_err,
                        }

                m["ml_eval"] = ml_fold_results
                m["synthetic_data"] = generated_rows
                m.update({
                    "context_data": context_data_list,
                    "test_data": test_df_eval.to_dict('records') if not test_df_eval.empty else [],
                    "train_non_ctx_data": train_non_context_data,
                    "ctx_size_pct": ctx_size_pct,
                    "gen_size_pct": gen_size_pct,
                    "join_context": join_context,
                    "generation_stats": generation_stats,
                    # Total wall-clock time for this fold (data split + generation +
                    # fidelity/privacy-attack metrics + all ML arms) -- tracked for every
                    # fold in every arm/combo, not just LLM generation time.
                    "fold_seconds": time.time() - fold_start_time,
                })
                results.append(m)

                if on_fold_done:
                    try:
                        on_fold_done(fold_num, m)
                    except Exception as cb_err:
                        logger.warning(f"on_fold_done callback failed for {filename} fold {fold_num}: {cb_err}")

                if progress_dict is not None and dataset_id_key:
                    # Explicitly update nested manager dict to ensure cross-process visibility
                    try:
                        p = dict(progress_dict[dataset_id_key])
                        p["done"] = fold_num
                        p["status"] = f"Fold {fold_num}/{folds} complete..."
                        progress_dict[dataset_id_key] = p
                    except Exception as pe:
                        logger.warning(f"Metadata progress update failed for {filename}: {pe}")
        except Exception as e:
            logger.error(f"Pipeline error in {filename} fold {fold_num}: {e}")
            continue

    overall_ml = _aggregate_overall_ml(results)

    return {
        "dataset_id": dataset_id, "filename": filename, "results": results, "synthetic_data": all_synthetic_data,
        "augmented_data": all_augmented_data, "overall_ml": overall_ml, "dataset_audit": dataset_audit,
        "avg_fidelity": sum(r.get("Synthetic Fidelity (JSD)", 0) for r in results) / len(results) if results else 0
    }

def run_smote_dataset_benchmark(args):
    """
    SMOTE baseline arm: the same per-fold protocol as run_single_dataset_benchmark
    (identical train_test_split seeds, identical fidelity/privacy-attack/3-arm-ML
    evaluation), but generates synthetic rows via SMOTE/SMOTENC
    (services.smote_baseline.generate_smote_rows) instead of the LLM. Reusing the exact
    same fold seeds is what makes SMOTE folds directly paired/comparable with the LLM
    arms' folds for apply_fdr_correction/_paired_pvalue.

    args: (dataset_id, filename, orig_data, col_defs, config) -- same shape as
    run_single_dataset_benchmark, so it slots into the same call convention.

    config["baseline_method"] selects the generator: "smote" (default), or "ctgan" /
    "tvae" (services.deep_baselines.generate_deep_rows, sampling len(train) rows).
    """
    from services.smote_baseline import generate_smote_rows
    from services.deep_baselines import generate_deep_rows

    dataset_id, filename, orig_data, col_defs, config = args
    logger.info(f"Starting {config.get('baseline_method', 'smote').upper()} baseline benchmark: {filename}")

    folds = config.get("folds", 3)
    ml_config = config.get("ml_config", {})
    targets = config.get("targets", [])
    na_strategy = config.get("na_strategy", "impute")
    method = config.get("baseline_method", "smote")
    label = method.upper()

    df_orig_all = pd.DataFrame(orig_data)
    if '_row_id' in df_orig_all.columns:
        df_orig_all.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')

    field_map = {c['field']: c.get('headerName', c['field']) for c in col_defs if c['field'] not in ['ai_audit', '_row_id']}
    df_orig_all = df_orig_all.rename(columns=field_map)

    dataset_audit = _compute_dataset_audit(df_orig_all, targets, filename)

    target_col = targets[0] if targets else None
    task_type = MLEvaluator.detect_task_type(df_orig_all, targets) if targets else None
    categorical_cols = [c for c in df_orig_all.select_dtypes(exclude=[np.number]).columns if c != target_col]

    # See the matching block in run_single_dataset_benchmark() for what these do.
    resume_fold_results = config.get("_resume_fold_results") or {}
    on_fold_done = config.get("_on_fold_done")

    results = []
    all_synthetic_data = []

    for fold_num in range(1, folds + 1):
        if fold_num in resume_fold_results:
            m = resume_fold_results[fold_num]
            results.append(m)
            all_synthetic_data.extend(m.get("synthetic_data", []) or [])
            logger.info(f"[{filename} | Fold {fold_num}] ({label}) Reusing cached result from checkpoint -- skipping regeneration.")
            continue

        try:
            fold_start_time = time.time()
            df_train_orig, df_test_orig = train_test_split(df_orig_all, test_size=0.2, random_state=42 + fold_num)
            logger.info(f"[DATA] [{filename} | Fold {fold_num}] ({label}) Split Complete: Train={len(df_train_orig)}, Test={len(df_test_orig)}")

            gen_start_time = time.time()
            if target_col and method == "smote":
                generated_df = generate_smote_rows(df_train_orig, target_col, categorical_cols, random_state=42 + fold_num)
            elif target_col:
                generated_df = generate_deep_rows(df_train_orig, target_col, categorical_cols, random_state=42 + fold_num, method=method)
            else:
                generated_df = pd.DataFrame(columns=df_train_orig.columns)
            generation_seconds = time.time() - gen_start_time
            logger.info(f"[{filename} | Fold {fold_num}] ({label}) Generated: {len(generated_df)} rows in {generation_seconds:.3f}s")

            generated_rows = generated_df.to_dict('records')
            for row in generated_rows: row["_fold_id"] = fold_num
            all_synthetic_data.extend(generated_rows)

            # SMOTE has no LLM cost at all -- near-zero generation_stats is itself a
            # meaningful comparison point against the LLM arm's token/time cost.
            generation_stats = {
                "prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0,
                "rule_violations": 0, "repair_attempts": 0,
                "generation_seconds": generation_seconds,
            }

            # SMOTE has no separate real-context blend -- the generated corpus IS what
            # gets scored, so the "blended" and "Gen "-prefixed fidelity are identical.
            # calculate_metrics guards internally for an empty generated_df (returns 0s).
            m = calculate_metrics(df_orig_all, generated_df)
            gen_fidelity = dict(m)
            m.update({f"Gen {k}": v for k, v in gen_fidelity.items()})

            if target_col and task_type == "classification":
                m["fidelity_by_class"] = calculate_metrics_by_class(df_orig_all, generated_df, target_col)

            c2st_accuracy = compute_c2st_accuracy(df_train_orig, generated_df)
            mia_auc = compute_mia_attack_auc(df_train_orig, df_test_orig, generated_df)
            if c2st_accuracy is not None:
                m["Gen C2ST Accuracy"] = c2st_accuracy
            if mia_auc is not None:
                m["Gen MIA Attack AUC"] = mia_auc
            validity = compute_schema_validity(df_orig_all, generated_df)
            if validity is not None:
                m["Gen Valid Rows"] = validity

            ml_fold_results = {}
            if ml_config and targets:
                # Concatenating an empty frame would recast every column to object dtype
                # and silently change the downstream preprocessing.
                df_full_aug = pd.concat([df_train_orig, generated_df], ignore_index=True) if len(generated_df) else df_train_orig.copy()
                for m_type in ml_config.get("selected_models", []):
                    m_params = ml_config.get("params", {}).get(m_type, {})
                    logger.info(f"[ML] [{filename} | Fold {fold_num}] ({label}) Evaluating {m_type.upper()} (NA Strategy: {na_strategy})")

                    orig_perf, _,        orig_err = MLEvaluator.train_and_evaluate(df_train_orig, targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                    gen_perf,  gen_meta,  gen_err  = MLEvaluator.train_and_evaluate(generated_df,  targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                    aug_perf,  aug_meta,  aug_err  = MLEvaluator.train_and_evaluate(df_full_aug,   targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)

                    ml_fold_results[m_type] = {
                        "original": orig_perf,
                        "generated": gen_perf, "generated_metadata": gen_meta,
                        "augmented": aug_perf, "augmented_metadata": aug_meta,
                        "task_type": task_type, "error": aug_err or gen_err or orig_err,
                    }

            m["ml_eval"] = ml_fold_results
            m["synthetic_data"] = generated_rows
            m.update({
                "context_data": [],
                "test_data": df_test_orig.to_dict('records'),
                "train_non_ctx_data": [],
                "ctx_size_pct": 0,
                "gen_size_pct": 100,
                "join_context": False,
                "generation_stats": generation_stats,
                "fold_seconds": time.time() - fold_start_time,
            })
            results.append(m)

            if on_fold_done:
                try:
                    on_fold_done(fold_num, m)
                except Exception as cb_err:
                    logger.warning(f"on_fold_done callback failed for {filename} fold {fold_num}: {cb_err}")
        except Exception as e:
            logger.error(f"{label} pipeline error in {filename} fold {fold_num}: {e}")
            continue

    overall_ml = _aggregate_overall_ml(results)

    return {
        "dataset_id": dataset_id, "filename": filename, "results": results,
        "synthetic_data": all_synthetic_data, "augmented_data": all_synthetic_data,
        "overall_ml": overall_ml, "dataset_audit": dataset_audit,
        "avg_fidelity": sum(r.get("Synthetic Fidelity (JSD)", 0) for r in results) / len(results) if results else 0
    }


def run_stochasticity_check(args, repeats: int = 5) -> dict:
    """
    Isolates pure LLM generation stochasticity from data-split and context-sampling
    variance. A normal fold-to-fold run changes the data split, the context sample, AND
    triggers a fresh LLM generation every fold, so its std conflates all three sources.
    Here, ONE train/test split (fold 1's seed) and ONE context sample are fixed once,
    then generate_bulk_data is called `repeats` times on those identical inputs -- any
    variance across repeats is attributable to LLM sampling noise alone. Comparing this
    std against the same metric's per-fold std in a normal batch run shows how much of
    the reported fold-to-fold spread is actually generation noise vs. data/context noise.

    args: (dataset_id, filename, orig_data, col_defs, config) -- same shape as
    run_single_dataset_benchmark. config["repeats"] overrides the `repeats` argument.
    """
    dataset_id, filename, orig_data, col_defs, config = args
    logger.info(f"Starting stochasticity check: {filename}")

    repeats = int(config.get("repeats", repeats))
    strategy = config.get("strategy", "random")
    algorithm = config.get("algorithm", "google/gemma-3-4b")
    temperature = config.get("temperature", 0.3)
    chunk_size = config.get("chunk_size", 30)
    instructions = config.get("instructions", "")
    desc_metadata = config.get("desc_metadata", "")
    ml_config = config.get("ml_config", {})
    targets = config.get("targets", [])
    na_strategy = config.get("na_strategy", "impute")
    rules_metadata = config.get("rules_metadata", None)
    enforce_rules = config.get("enforce_rules", True)
    enforce_boundaries = config.get("enforce_boundaries", False)
    inject_inferred_rules = bool(config.get("inject_inferred_rules", True))
    apply_post_processing = bool(config.get("apply_post_processing", True))
    ctx_size_pct = config.get("ctx_size_pct", 20)
    gen_size_pct = config.get("gen_size_pct", 100)

    df_orig_all = pd.DataFrame(orig_data)
    if '_row_id' in df_orig_all.columns:
        df_orig_all.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
    field_map = {c['field']: c.get('headerName', c['field']) for c in col_defs if c['field'] not in ['ai_audit', '_row_id']}
    df_orig_all = df_orig_all.rename(columns=field_map)

    inferred_rules = RuleInference.infer_dataset_rules(df_orig_all) if inject_inferred_rules else []
    rules_hierarchy = {
        "integrity": [r for r in inferred_rules if r['level'] == 1],
        "boundaries": [r for r in inferred_rules if r['level'] == 2],
        "user": rules_metadata if isinstance(rules_metadata, list) else [],
        "enforce_user": bool(enforce_rules),
        "enforce_boundaries": bool(enforce_boundaries),
    }

    # ONE fixed split (fold 1's seed) -- held constant across every repeat.
    df_train_orig, df_test_orig = train_test_split(df_orig_all, test_size=0.2, random_state=43)
    n_context_rows = max(1, int(len(df_train_orig) * (ctx_size_pct / 100)))
    num_empty = max(1, int(len(df_train_orig) * (gen_size_pct / 100)))

    # ONE fixed context sample. select_context_rows' "random" branch calls df.sample()
    # with no random_state (verified by reading services/data_manager.py), so it is NOT
    # deterministic between calls -- holding it fixed here, rather than resampling every
    # repeat, is what isolates pure LLM sampling noise from context-sampling noise.
    context_data_list = select_context_rows(df_train_orig, strategy=strategy, n_rows=n_context_rows) or []
    context_dict = {"metadata": desc_metadata, "manual_ctx": "[]", "global_ctx": context_data_list}
    all_fields = list(df_orig_all.columns)

    target_col = targets[0] if targets else None
    task_type = MLEvaluator.detect_task_type(df_orig_all, targets) if targets else None

    repeat_results = []
    for repeat_num in range(1, repeats + 1):
        try:
            repeat_start_time = time.time()
            gen_start_time = time.time()
            result_bundle = generate_bulk_data(
                [], context_dict, all_fields, all_fields, algorithm, instructions, temperature,
                num_empty=num_empty, chunk_size=chunk_size,
                rules_metadata=rules_hierarchy, enforce_rules=True
            )
            generation_seconds = time.time() - gen_start_time

            if not result_bundle:
                logger.warning(f"Stochasticity check: empty result_bundle for {filename} repeat {repeat_num}")
                continue

            from services.post_processor import PostProcessor
            generated_rows = result_bundle["data"]
            if apply_post_processing:
                generated_rows = PostProcessor.apply_active_rules(generated_rows, rules_hierarchy.get("user", []))
            generation_stats = {**result_bundle.get("stats", {}), "generation_seconds": generation_seconds}
            for row in generated_rows: row["_repeat_id"] = repeat_num
            generated_df = pd.DataFrame(generated_rows)

            m = calculate_metrics(df_orig_all, generated_df)
            gen_fidelity = dict(m)
            m.update({f"Gen {k}": v for k, v in gen_fidelity.items()})

            if target_col and task_type == "classification":
                m["fidelity_by_class"] = calculate_metrics_by_class(df_orig_all, generated_df, target_col)

            c2st_accuracy = compute_c2st_accuracy(df_train_orig, generated_df)
            mia_auc = compute_mia_attack_auc(df_train_orig, df_test_orig, generated_df)
            if c2st_accuracy is not None:
                m["Gen C2ST Accuracy"] = c2st_accuracy
            if mia_auc is not None:
                m["Gen MIA Attack AUC"] = mia_auc
            validity = compute_schema_validity(df_orig_all, generated_df)
            if validity is not None:
                m["Gen Valid Rows"] = validity

            ml_repeat_results = {}
            if ml_config and targets:
                df_full_aug = pd.concat([df_train_orig, generated_df], ignore_index=True)
                for m_type in ml_config.get("selected_models", []):
                    m_params = ml_config.get("params", {}).get(m_type, {})
                    orig_perf, _,       orig_err = MLEvaluator.train_and_evaluate(df_train_orig, targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                    gen_perf,  gen_meta, gen_err = MLEvaluator.train_and_evaluate(generated_df,  targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                    aug_perf,  aug_meta, aug_err = MLEvaluator.train_and_evaluate(df_full_aug,   targets, m_type, m_params, task_type, test_df=df_test_orig, na_strategy=na_strategy)
                    ml_repeat_results[m_type] = {
                        "original": orig_perf,
                        "generated": gen_perf, "generated_metadata": gen_meta,
                        "augmented": aug_perf, "augmented_metadata": aug_meta,
                        "task_type": task_type, "error": aug_err or gen_err or orig_err,
                    }

            m["ml_eval"] = ml_repeat_results
            m["synthetic_data"] = generated_rows
            m.update({
                "repeat": repeat_num,
                "generation_stats": generation_stats,
                "fold_seconds": time.time() - repeat_start_time,
            })
            repeat_results.append(m)
        except Exception as e:
            logger.error(f"Stochasticity check error in {filename} repeat {repeat_num}: {e}")
            continue

    return {
        "dataset_id": dataset_id, "filename": filename, "repeat_results": repeat_results,
        "split_sizes": {"train": len(df_train_orig), "test": len(df_test_orig), "context": len(context_data_list)},
        "overall_ml": _aggregate_overall_ml(repeat_results),
    }


class PipelineManager:
    @staticmethod
    def run_pipeline(tasks):
        # Support synchronous execution for testing (prevents mock loss across processes)
        import os
        if len(tasks) == 1 or os.environ.get("TABULLM_SYNC_TEST") == "1":
            return [run_single_dataset_benchmark(t) for t in tasks]

        cpus = min(len(tasks), multiprocessing.cpu_count())
        if cpus == 0: return []
        with multiprocessing.Pool(processes=cpus) as pool:
            results = pool.map(run_single_dataset_benchmark, tasks)
        return results
