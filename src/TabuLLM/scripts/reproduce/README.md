# Reproducing the Synthagent experiments

Everything in the paper comes from the runs below and from
`Synthagent__An_Agentic_Framework_for_Synthetic_Data_Generation/generate_paper_assets.py`,
which rebuilds every table and figure from the committed results.

## Environment

- Python 3.13, `pip install -r requirements-lock.txt` (exact versions used; `requirements.txt`
  lists the same packages without pins).
- API keys as environment variables: `OPENAI_API_KEY` for the OpenAI models and
  `TABULLM_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY` outside Claude Code environments) for
  the Claude models. Never commit them.
- Datasets: the six original CSVs are in `data/medical/`. The four further UCI datasets are
  also committed, and `python scripts/reproduce/prepare_uci_datasets.py` regenerates them
  byte for byte from the UCI repository.

## What is and is not deterministic

- The ten train/test divisions of every dataset are fixed (`random_state = 42 + repetition`),
  so every configuration sees the same divisions.
- SMOTE, ADASYN, Borderline-SMOTE, CTGAN, TVAE, the Gaussian copula, CART, ARF, TabDDPM,
  GReaT and the Bayesian network are seeded with the same value.
- LLM responses are not deterministic, even at temperature 0, and model aliases such as
  `gpt-4o-mini` can be updated by the provider. A rerun therefore reproduces the procedure
  and the statistics within the run-to-run variation the paper reports (the noise floor),
  not the identical records. The runs were made between 18 September and 6 November 2026.
- To make the reported numbers checkable without calling an LLM again, every run saves the
  synthetic records of each repetition in `<run>/synthetic/<arm>/<dataset>_fold<k>.csv`
  (added on 6 November 2026; earlier runs kept only the per-repetition metrics, in
  `<run>/<arm>/<dataset>_per_fold.csv`, which are what the paper's statistics use).

## Runs

All commands run from `src/TabuLLM`. A run resumes from its `checkpoint.json` when the same
command is repeated. Common settings: ten repetitions, N equal to the training size.

| Results folder | Provider / model | `TABULLM_ABLATION_ARMS` | Datasets |
|---|---|---|---|
| `results/ablation_20260918_114707` | openai / gpt-4o-mini | all control-layer variants, `raw_prompting`, `structured_output`, `full_logged`, `boundaries`, `smote`, `ctgan`, `tvae`; `TABULLM_OPENAI_SWEEP_MODELS=gpt-4.1-mini,gpt-4o,gpt-4.1` for the LLM variants | original six |
| `results/claude/<run>` | anthropic / claude-haiku-4-5, claude-sonnet-4-6, claude-sonnet-5-5, claude-haiku-5-5 | `full`; `raw_prompting` and `structured_output` with Haiku 5.5 | original six (see `scripts/claude_runs/`) |
| `results/sens_<arm>` | openai / gpt-4o-mini | `ctx_10`, `ctx_40`, `batch_15`, `batch_60`, `temp_0`, `temp_10` (one per folder) | original six |
| `results/new_full`, `new_raw`, `new_structured` | openai / gpt-4o-mini | `full`, `raw_prompting`, `structured_output` | four new |
| `results/new_gpt41` | openai / gpt-4.1 | `full` | four new |
| `results/new_baselines` | (no LLM) | `smote,ctgan,tvae` | four new |
| `results/cllm_old`, `cllm_new` | openai / gpt-4o-mini | `cllm` | original six / four new |
| `results/copula_old`, `copula_new` | (no LLM) | `gaussian_copula` | original six / four new |
| `results/cpu_old`, `cpu_new` | (no LLM) | `adasyn,borderline_smote,cart,arf` | original six / four new |
| `results/claude/sonnet46_newdata` | anthropic / claude-sonnet-4-6 | `full` | four new |
| `results/bn_all` | (no LLM, CPU) | `bayesian_network` | all ten |
| `results/gpu_baselines` | (no LLM, GPU) | `tabddpm,great` | all ten (see `scripts/gpu_baselines/`) |

When several runs share a machine, set `OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) so that the classifiers and generators of one run do not take every core.

Example (one sensitivity run):

```bash
TABULLM_PROVIDER=openai TABULLM_MODEL=gpt-4o-mini TABULLM_ABLATION_ARMS=ctx_10 \
TABULLM_ABLATION_DATASETS=heart_disease.csv,breast_cancer_wisconsin_diagnostic.csv,heart_failure_clinical_records.csv,indian_diabetes.csv,indian_liver_patient_dataset.csv,parkinsons.csv \
LOCAL_LLM_TIMEOUT=120 python run_batch_pipeline.py --ablation --resume results/sens_ctx_10
```

`run_all.sh` runs every row of the table in sequence.

## Paper assets

```bash
cd ../../Synthagent__An_Agentic_Framework_for_Synthetic_Data_Generation
python generate_paper_assets.py      # writes generated/ (tables, figures, tests)
latexmk -pdf main.tex
```
