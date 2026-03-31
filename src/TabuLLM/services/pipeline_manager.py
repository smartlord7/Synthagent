import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from services.llm_client import generate_bulk_data
from services.data_manager import select_context_rows
from utils.metrics import calculate_metrics
from utils.logging_config import logger
from services.ml_evaluator import MLEvaluator

def run_single_dataset_benchmark(args):
    """
    Worker function for multiprocessing. 
    args: (dataset_id, filename, orig_data, col_defs, config)
    """
    dataset_id, filename, orig_data, col_defs, config = args
    logger.info(f"Starting pipeline benchmark: {filename}")
    
    folds = config.get("folds", 3)
    max_context = config.get("max_context", 60)
    strategy = config.get("strategy", "random")
    algorithm = config.get("algorithm", "google/gemma-3-4b")
    temperature = config.get("temperature", 0.3)
    chunk_size = config.get("chunk_size", 30)
    instructions = config.get("instructions", "")
    desc_metadata = config.get("desc_metadata", "")
    ml_config = config.get("ml_config", {})
    targets = config.get("targets", [])
    
    progress_dict = config.get("_progress_dict")
    dataset_id_key = config.get("_dataset_id")

    df_orig_all = pd.DataFrame(orig_data)
    if '_row_id' in df_orig_all.columns:
        df_orig_all.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
    
    field_map = {c['field']: c.get('headerName', c['field']) for c in col_defs if c['field'] not in ['ai_audit', '_row_id']}
    df_orig_all = df_orig_all.rename(columns=field_map)
    
    results = []
    all_synthetic_data = []

    for fold_num in range(1, folds + 1):
        if progress_dict is not None:
             p = progress_dict[dataset_id_key]
             p["status"] = f"Fold {fold_num}/{folds} - Data Prep"
             progress_dict[dataset_id_key] = p

        try:
            # 1. Split Original into Train/Test (80/20) for this fold
            # This ensures we always test on "Real" data
            df_train_orig, df_test_orig = train_test_split(df_orig_all, test_size=0.2, random_state=42 + fold_num)
            
            # 2. Setup context for synthesis from the training fraction
            n_context_rows = min(len(df_train_orig), max_context)
            context_data_list = select_context_rows(df_train_orig, strategy=strategy, n_rows=n_context_rows)
            num_empty = len(df_orig_all) - len(context_data_list)
            
            context_dict = {"metadata": desc_metadata, "manual_ctx": "[]", "global_ctx": context_data_list}

            # 3. Generate Augmented Data
            if progress_dict is not None:
                 p = progress_dict[dataset_id_key]
                 p["status"] = f"Fold {fold_num}/{folds} - LLM Gen"
                 progress_dict[dataset_id_key] = p
                 
            result_bundle = generate_bulk_data(
                [], context_dict, list(df_orig_all.columns),
                algorithm, instructions, temperature, num_empty=num_empty,
                chunk_size=chunk_size
            )

            if result_bundle:
                generated_rows = result_bundle["data"]
                # Add fold ID for grouping in EDA aggregation
                for row in generated_rows:
                    row["_fold_id"] = fold_num
                all_synthetic_data.extend(generated_rows)
                
                generated_df = pd.DataFrame(generated_rows)
                context_df = pd.DataFrame(context_data_list)
                df_aug_train = pd.concat([context_df, generated_df], ignore_index=True)
                
                # Fidelity Metrics
                m = calculate_metrics(df_orig_all, df_aug_train)
                
                # ML Downstream Metrics (Train on Synthetic, Test on Real)
                ml_fold_results = {}
                if ml_config and targets:
                    task_type = MLEvaluator.detect_task_type(df_orig_all, targets)
                    for m_type in ml_config.get("selected_models", []):
                        if progress_dict is not None:
                             p = progress_dict[dataset_id_key]
                             p["status"] = f"Fold {fold_num}/{folds} - Eval {m_type.upper()}"
                             progress_dict[dataset_id_key] = p
                             
                        m_params = ml_config.get("params", {}).get(m_type, {})
                        
                        # Performance on Original (Train on 80% Real, Test on 20% Real)
                        orig_perf, _ = MLEvaluator.train_and_evaluate(df_train_orig, targets, m_type, m_params, task_type, test_df=df_test_orig)
                        # Performance on Augmented (Train on Synthetic+Context, Test on 20% Real)
                        aug_perf, aug_meta = MLEvaluator.train_and_evaluate(df_aug_train, targets, m_type, m_params, task_type, test_df=df_test_orig)
                        
                        ml_fold_results[m_type] = {
                            "original": orig_perf,
                            "augmented": aug_perf,
                            "augmented_metadata": aug_meta,
                            "task_type": task_type
                        }
                
                m["ml_eval"] = ml_fold_results
                m["synthetic_data"] = generated_rows
                results.append(m)
                
                if progress_dict is not None:
                    p = progress_dict[dataset_id_key]
                    p["done"] = fold_num
                    progress_dict[dataset_id_key] = p
        except Exception as e:
            import traceback
            logger.error(f"Pipeline error in {filename} fold {fold_num}: {e}")
            logger.error(traceback.format_exc())
            continue

    # --- AGGREGATE ML RESULTS FOR STORE ---
    overall_ml = {}
    if results:
        # Collect all models across folds
        all_models = set()
        for r in results:
            all_models.update(r.get("ml_eval", {}).keys())
            
        for m_type in all_models:
            overall_ml[m_type] = {}
            for m_name in ["accuracy", "f1", "rmse", "r2"]:
                f_orig = [
                    r["ml_eval"][m_type]["original"][m_name]
                    for r in results
                    if m_type in r.get("ml_eval", {})
                    and r["ml_eval"][m_type].get("original") is not None
                    and m_name in r["ml_eval"][m_type]["original"]
                ]
                f_aug = [
                    r["ml_eval"][m_type]["augmented"][m_name]
                    for r in results
                    if m_type in r.get("ml_eval", {})
                    and r["ml_eval"][m_type].get("augmented") is not None
                    and m_name in r["ml_eval"][m_type]["augmented"]
                ]
                
                if f_orig and f_aug:
                    m_o, s_o = np.mean(f_orig), np.std(f_orig)
                    m_a, s_a = np.mean(f_aug), np.std(f_aug)
                    
                    deltas = [a - o for a, o in zip(f_aug, f_orig)]
                    m_d, s_d = np.mean(deltas), np.std(deltas)
                    
                    # Store aggregated metadata list for full fold inspection
                    overall_ml[m_type][m_name] = {
                        "real": f"{m_o:.3f} ± {s_o:.2f}",
                        "aug": f"{m_a:.3f} ± {s_a:.2f}",
                        "delta": f"{m_d:+.3f} ± {s_d:.2f}"
                    }
                
                # Capture all fold metadata for this model type
                overall_ml[m_type]["folds_metadata"] = [
                    r["ml_eval"][m_type].get("augmented_metadata")
                    for r in results
                    if m_type in r.get("ml_eval", {})
                    and r["ml_eval"][m_type].get("augmented_metadata") is not None
                ]

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "results": results,
        "synthetic_data": all_synthetic_data,
        "overall_ml": overall_ml,
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
