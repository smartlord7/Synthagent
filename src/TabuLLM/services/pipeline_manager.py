import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from services.llm_client import generate_bulk_data
from services.data_manager import select_context_rows
from utils.metrics import calculate_metrics
from utils.logging_config import logger
from services.ml_evaluator import MLEvaluator
from services.rule_inference import RuleInference

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
    
    results = []
    all_synthetic_data = []
    all_augmented_data = [] 

    for fold_num in range(1, folds + 1):
        if progress_dict is not None:
             p = progress_dict[dataset_id_key]
             p["status"] = f"Fold {fold_num}/{folds} - Data Prep"
             progress_dict[dataset_id_key] = p

        try:
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
            
            # Infer hierarchical rules
            inferred_rules = RuleInference.infer_dataset_rules(df_orig_all)
            
            rules_hierarchy = {
                "integrity": [r for r in inferred_rules if r['level'] == 1],
                "boundaries": [r for r in inferred_rules if r['level'] == 2],
                "user": rules_metadata if isinstance(rules_metadata, list) else [],
                "enforce_user": bool(enforce_rules),
                "enforce_boundaries": bool(enforce_boundaries)
            }

            result_bundle = generate_bulk_data(
                [], context_dict, list(df_orig_all.columns), algorithm, instructions, temperature,
                num_empty=num_empty, chunk_size=chunk_size,
                rules_metadata=rules_hierarchy, enforce_rules=True
            )

            if result_bundle:
                generated_rows = result_bundle["data"]
                test_data_list = df_test_orig.to_dict('records')
                
                for row in generated_rows: row["_fold_id"] = fold_num
                for row in context_data_list: row["_fold_id"] = fold_num
                for row in test_data_list: row["_fold_id"] = fold_num
                
                all_synthetic_data.extend(generated_rows)
                all_augmented_data.extend(context_data_list + generated_rows + test_data_list)
                
                if join_context:
                    df_aug_train = pd.concat([pd.DataFrame(context_data_list), pd.DataFrame(generated_rows)], ignore_index=True)
                    test_df_eval = df_test_orig # Standard 20% test
                else:
                    df_aug_train = pd.DataFrame(generated_rows)
                    test_df_eval = df_orig_all # Evaluate against whole real dataset
                
                logger.info(f"[{filename} | Fold {fold_num}] Post-Generation Geometry -> Target: {num_empty} | Generated: {len(generated_rows)} | Final Train Size: {len(df_aug_train)} | Test Size: {len(test_df_eval)}")
                
                # Fidelity Metrics
                m = calculate_metrics(df_orig_all, df_aug_train)
                
                # ML Downstream Metrics
                ml_fold_results = {}
                if ml_config and targets:
                    task_type = MLEvaluator.detect_task_type(df_orig_all, targets)
                    for m_type in ml_config.get("selected_models", []):
                        if progress_dict is not None:
                             p = progress_dict[dataset_id_key]
                             p["status"] = f"Fold {fold_num}/{folds} - Eval {m_type.upper()}"
                             progress_dict[dataset_id_key] = p
                             
                        m_params = ml_config.get("params", {}).get(m_type, {})
                        # Apply NA Handling Strategy to both original and augmented
                        logger.info(f"[ML] [{filename} | Fold {fold_num}] Evaluating {m_type.upper()} (NA Strategy: {na_strategy})")
                        
                        # Reference model is always tested on the same test set as the augmented model for comparison
                        orig_perf, _, orig_err = MLEvaluator.train_and_evaluate(df_train_orig, targets, m_type, m_params, task_type, test_df=test_df_eval, na_strategy=na_strategy)
                        aug_perf, aug_meta, aug_err = MLEvaluator.train_and_evaluate(df_aug_train, targets, m_type, m_params, task_type, test_df=test_df_eval, na_strategy=na_strategy)
                        
                        ml_fold_results[m_type] = {"original": orig_perf, "augmented": aug_perf, "augmented_metadata": aug_meta, "task_type": task_type, "error": aug_err or orig_err}
                
                m["ml_eval"] = ml_fold_results
                m["synthetic_data"] = generated_rows
                m.update({
                    "context_data": context_data_list, 
                    "test_data": test_df_eval.to_dict('records') if not test_df_eval.empty else [],
                    "train_non_ctx_data": train_non_context_data,
                    "ctx_size_pct": ctx_size_pct, 
                    "gen_size_pct": gen_size_pct, 
                    "join_context": join_context
                })
                results.append(m)
                
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

    # --- AGGREGATE ML RESULTS FOR STORE ---
    overall_ml = {}
    if results:
        all_models = set()
        for r in results: all_models.update(r.get("ml_eval", {}).keys())
        for m_type in all_models:
            overall_ml[m_type] = {}
            for m_name in ["accuracy", "f1", "rmse", "r2"]:
                f_orig = [r["ml_eval"][m_type]["original"][m_name] for r in results if m_type in r.get("ml_eval", {}) and r["ml_eval"][m_type].get("original") is not None and m_name in r["ml_eval"][m_type]["original"]]
                f_aug = [r["ml_eval"][m_type]["augmented"][m_name] for r in results if m_type in r.get("ml_eval", {}) and r["ml_eval"][m_type].get("augmented") is not None and m_name in r["ml_eval"][m_type]["augmented"]]
                
                if f_orig and f_aug:
                    m_o, s_o = np.mean(f_orig), np.std(f_orig)
                    m_a, s_a = np.mean(f_aug), np.std(f_aug)
                    m_d, s_d = np.mean([a - o for a, o in zip(f_aug, f_orig)]), np.std([a - o for a, o in zip(f_aug, f_orig)])
                    
                    overall_ml[m_type][m_name] = {
                        "real": m_o, "real_sd": s_o, "aug": m_a, "aug_sd": s_a, "delta": m_d, "delta_sd": s_d,
                        "display_real": f"{m_o:.3f} ± {s_o:.2f}", "display_aug": f"{m_a:.3f} ± {s_a:.2f}", "display_delta": f"{m_d:+.3f} ± {s_d:.2f}"
                    }
                
                overall_ml[m_type]["folds_metadata"] = [r["ml_eval"][m_type].get("augmented_metadata") for r in results if m_type in r.get("ml_eval", {}) and r["ml_eval"][m_type].get("augmented_metadata") is not None]
                if not f_aug:
                    errors = [r["ml_eval"][m_type].get("error") for r in results if m_type in r.get("ml_eval", {}) and r["ml_eval"][m_type].get("error")]
                    if errors: overall_ml[m_type]["error"] = errors[0]

    return {
        "dataset_id": dataset_id, "filename": filename, "results": results, "synthetic_data": all_synthetic_data,
        "augmented_data": all_augmented_data, "overall_ml": overall_ml,
        "avg_fidelity": sum(r.get("Synthetic Fidelity (JSD)", 0) for r in results) / len(results) if results else 0
    }

class PipelineManager:
    @staticmethod
    def run_pipeline(tasks):
        cpus = min(len(tasks), multiprocessing.cpu_count())
        if cpus == 0: return []
        with multiprocessing.Pool(processes=cpus) as pool:
            results = pool.map(run_single_dataset_benchmark, tasks)
        return results
