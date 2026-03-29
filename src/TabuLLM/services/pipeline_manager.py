import multiprocessing
import pandas as pd
from services.llm_client import generate_bulk_data
from services.data_manager import select_context_rows
from utils.metrics import calculate_metrics
from utils.logging_config import logger

def run_single_dataset_benchmark(args):
    """
    Worker function for multiprocessing. 
    args: (dataset_id, filename, orig_data, col_defs, config)
    """
    dataset_id, filename, orig_data, col_defs, config = args
    logger.info(f"Starting pipeline benchmark for: {filename}")
    
    folds = config.get("folds", 3)
    max_context = config.get("max_context", 60)
    strategy = config.get("strategy", "random")
    algorithm = config.get("algorithm", "bulk")
    temperature = config.get("temperature", 0.3)
    chunk_size = config.get("chunk_size", 30)
    instructions = config.get("instructions", "")
    desc_metadata = config.get("desc_metadata", "")

    df_orig = pd.DataFrame(orig_data)
    if '_row_id' in df_orig.columns:
        df_orig.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], inplace=True, errors='ignore')
    
    field_to_header = {
        col['field']: col.get('headerName', col['field'])
        for col in col_defs
        if col['field'] not in ['ai_audit', '_row_id'] and not col['field'].startswith('is_ai_')
    }
    df_orig = df_orig.rename(columns=field_to_header)
    
    fraction = min(0.8, max_context / len(df_orig)) if not df_orig.empty else 0.2
    results = []
    all_synthetic_data = []

    for fold_num in range(1, folds + 1):
        try:
            n_target = int(len(df_orig) * fraction)
            n_context_rows = min(n_target, max_context)
            
            context_data_list = select_context_rows(df_orig, strategy=strategy, n_rows=n_context_rows)
            context_df = pd.DataFrame(context_data_list)
            num_empty = len(df_orig) - len(context_df)
            
            context_dict = {
                "metadata": desc_metadata,
                "manual_ctx": "[]",
                "global_ctx": context_data_list
            }

            result_bundle = generate_bulk_data(
                [], context_dict, list(field_to_header.values()),
                algorithm, instructions, temperature, num_empty=num_empty,
                chunk_size=chunk_size
            )

            if result_bundle:
                generated_rows = result_bundle["data"]
                all_synthetic_data.extend(generated_rows)
                
                generated_df = pd.DataFrame(generated_rows)
                aug_df = pd.concat([context_df, generated_df], ignore_index=True)
                m = calculate_metrics(df_orig, aug_df)
                results.append(m)
        except Exception as e:
            logger.error(f"Pipeline error in {filename} fold {fold_num}: {e}")
            continue

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "results": results,
        "synthetic_data": all_synthetic_data,
        "avg_fidelity": sum(r["fidelity_score"] for r in results) / len(results) if results else 0
    }

class PipelineManager:
    @staticmethod
    def run_pipeline(tasks):
        """
        Runs multiple benchmarks in parallel.
        tasks: list of (dataset_id, filename, orig_data, col_defs, config)
        """
        cpus = min(len(tasks), multiprocessing.cpu_count())
        if cpus == 0: return []
        
        with multiprocessing.Pool(processes=cpus) as pool:
            results = pool.map(run_single_dataset_benchmark, tasks)
        
        return results
