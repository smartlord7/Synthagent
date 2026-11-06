# GPU baselines: TabDDPM, GReaT and a Bayesian network

TabDDPM and GReaT need a GPU, so they run on your own machine; the Bayesian network needs no GPU but uses the same synthcity install, so it runs there too. Everything else
(datasets, folds, classifiers, metrics) is the same as for the other baselines, and the
results go to `src/TabuLLM/results/gpu_baselines/`, which the paper scripts merge.

## 1. Install (once)

From the repository root, in a Python 3.10–3.12 environment with a CUDA build of PyTorch:

```powershell
cd src\TabuLLM
pip install -r requirements.txt
pip install synthcity be_great
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## 2. Run (the three generators, all ten datasets, ten repetitions)

PowerShell:

```powershell
cd src\TabuLLM
$env:TABULLM_PROVIDER = "openai"          # no LLM is called by these arms
$env:TABULLM_MODEL = "gpt-4o-mini"
$env:TABULLM_ABLATION_ARMS = "tabddpm,great,bayesian_network"
$env:TABULLM_ABLATION_DATASETS = "heart_disease.csv,breast_cancer_wisconsin_diagnostic.csv,heart_failure_clinical_records.csv,indian_diabetes.csv,indian_liver_patient_dataset.csv,parkinsons.csv,breast_cancer_coimbra.csv,hepatitis.csv,chronic_kidney_disease.csv,thoracic_surgery.csv"
python run_batch_pipeline.py --ablation --resume results/gpu_baselines
```

bash:

```bash
cd src/TabuLLM
TABULLM_PROVIDER=openai TABULLM_MODEL=gpt-4o-mini TABULLM_ABLATION_ARMS=tabddpm,great,bayesian_network \
TABULLM_ABLATION_DATASETS=heart_disease.csv,breast_cancer_wisconsin_diagnostic.csv,heart_failure_clinical_records.csv,indian_diabetes.csv,indian_liver_patient_dataset.csv,parkinsons.csv,breast_cancer_coimbra.csv,hepatitis.csv,chronic_kidney_disease.csv,thoracic_surgery.csv \
python run_batch_pipeline.py --ablation --resume results/gpu_baselines
```

The run checkpoints every repetition; if it stops, run the same command again and it
resumes. When it prints `ABLATION COMPLETE`, commit and push `src/TabuLLM/results/gpu_baselines`.

## 3. Settings

- TabDDPM: synthcity plugin `ddpm`, library defaults.
- GReaT: `be_great` with `distilgpt2`, 100 epochs, batch size 32.
- Bayesian network: synthcity plugin `bayesian_network`, library defaults.
- Both fit on the training split of each repetition (missing values imputed as for the
  other baselines) and sample as many records as the training split has.
