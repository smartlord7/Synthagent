#!/bin/bash
# Runs every experiment of the paper in sequence (see README.md). From src/Synthagent:
#   bash scripts/reproduce/run_all.sh
# Needs OPENAI_API_KEY and TABULLM_ANTHROPIC_API_KEY. Each run resumes if repeated.
set -e
cd "$(dirname "$0")/../.."
OLD=heart_disease.csv,breast_cancer_wisconsin_diagnostic.csv,heart_failure_clinical_records.csv,indian_diabetes.csv,indian_liver_patient_dataset.csv,parkinsons.csv
NEW=breast_cancer_coimbra.csv,hepatitis.csv,chronic_kidney_disease.csv,thoracic_surgery.csv
run() {  # run PROVIDER MODEL ARMS DATASETS DIR [extra env]
  env TABULLM_PROVIDER=$1 TABULLM_MODEL=$2 TABULLM_ABLATION_ARMS=$3 TABULLM_ABLATION_DATASETS=$4 \
      LOCAL_LLM_TIMEOUT=180 "${@:6}" python run_batch_pipeline.py --ablation --resume "$5"
}
MAIN=results/ablation_20260918_114707
run openai gpt-4o-mini full,no_rule_injection,no_post_processing,no_context_join,strategy_dense,strategy_diverse,ponder_strategy,raw_prompting,structured_output,full_logged,boundaries,smote,ctgan,tvae $OLD $MAIN
run openai gpt-4o-mini backend_gpt_4_1_mini,backend_gpt_4o,backend_gpt_4_1 $OLD $MAIN TABULLM_OPENAI_SWEEP_MODELS=gpt-4.1-mini,gpt-4o,gpt-4.1
for m in haiku45:claude-haiku-4-5 sonnet46:claude-sonnet-4-6 sonnet55:claude-sonnet-5-5 haiku55:claude-haiku-5-5; do
  run anthropic ${m#*:} full $OLD results/claude/${m%%:*}_full
done
run anthropic claude-haiku-5-5 raw_prompting $OLD results/claude/haiku55_raw
run anthropic claude-haiku-5-5 structured_output $OLD results/claude/haiku55_structured
for a in ctx_10 ctx_40 batch_15 batch_60 temp_0 temp_10; do run openai gpt-4o-mini $a $OLD results/sens_$a; done
run openai gpt-4o-mini full $NEW results/new_full
run openai gpt-4o-mini raw_prompting $NEW results/new_raw
run openai gpt-4o-mini structured_output $NEW results/new_structured
run openai gpt-4.1 full $NEW results/new_gpt41
run openai gpt-4o-mini smote,ctgan,tvae $NEW results/new_baselines
run openai gpt-4o-mini cllm $OLD results/cllm_old
run openai gpt-4o-mini cllm $NEW results/cllm_new
run openai gpt-4o-mini gaussian_copula $OLD results/copula_old
run openai gpt-4o-mini gaussian_copula $NEW results/copula_new
run openai gpt-4o-mini adasyn,borderline_smote,cart,arf $OLD results/cpu_old
run openai gpt-4o-mini adasyn,borderline_smote,cart,arf $NEW results/cpu_new
run anthropic claude-sonnet-4-6 full $NEW results/claude/sonnet46_newdata
run openai gpt-4o-mini bayesian_network $OLD,$NEW results/bn_all
# GPU machine only: run openai gpt-4o-mini tabddpm,great $OLD,$NEW results/gpu_baselines
