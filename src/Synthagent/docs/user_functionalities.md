# Synthagent: User Functionality Guide

Synthagent is an enterprise-grade framework designed to procedurally augment undersized, imbalanced, or privacy-restricted tabular datasets. It utilizes Large Language Models (LLMs) to construct high-fidelity synthetic data, rigorously benchmarking its predictive utility via integrated Machine Learning engines.

Below is an exhaustive index of all tools, controls, and functionalities available to the end-user through the Synthagent interactive interface.

---

## 1. Interactive Data Workspace

The baseline entry-point for data orchestration, allowing users to safely import, inspect, and schema-define their tables before routing them into the LLM pipelines.

### Data Ingestion & State
* **Multi-File Uploading:** Drag-and-drop support for CSV and JSON datasets.
### Data Profiling & Quick Statistics
* **Live Telemetry:** Instantly computes dataset dimensions (Rows × Columns), total Missing values (NaNs), and active Memory footprint in KB.
* **Master Workspace Persistence:** Instantly save all uploaded datasets, UI targets, and generated benchmarks into a `workspace_session.json` state file to resume work on application reboot.
* **Workspace Viewer:** A high-performance AG-Grid displays the uploaded datasets, handling thousands of rows efficiently with cell-level editing, adding/removing rows/columns, and quick renaming. 

### Pre-Generation Auditing & Strategy
* **Health & Ethics Audit Module:** A comprehensive 5-pillar pre-generation analysis pane:
    * **Structural:** Detects sparsity issues and data outliers.
    * **Fairness:** Scans categorical schemas for potential group biases.
    * **Privacy:** Checks baseline raw data for re-identification risks.
    * **Quality:** Monitors basic data cleanliness bounds.
    * **Strategy (Ponder Strategy):** Feeds the dataset signature to an LLM to generate a custom mathematical "Strategic Augmentation Plan" in Markdown, advising you on exactly how to tune the engine.
* **Custom Mathematical Rules:** A flexible rule engine allowing users to forcefully inject bespoke logical rules prior to generation (e.g., `If Age < 18, Status = Minor`).

### AI-Assisted Schema Definition
* **Smart Define Tool (Insights):** Automatically probes a dataset using an LLM to generate a physical metadata summary in Markdown. It understands semantic column meanings and sets up the strict background prompt. Users can toggle freely between viewing the Markdown and directly editing the LLM's conclusions.
* **Target Auto-Detection:** Uses correlation and AI heuristics to automatically identify and flag the target classification/regression column (e.g., `Patient_Outcome`).
* **Column Role Toggles:** Users can manually force columns to be defined as **Numeric**, **Categorical**, or **Ignored**, directly steering how the ML Evaluation engine will encode them.

---

## 2. LLM Synthetic Generation Engine

The core generative reactor. Users can granularly control the creativity, bounds, and volume of the datasets manufactured.

### Strategy & Prompting Controls
* **Algorithmic Selectors:** Switch seamlessly between **Local Models** (e.g., `gemma-3`) and **Cloud APIs** (e.g., `Gemini 1.5 Pro`).
* **Generation Tone:** Enforce stylistic generation patterns (e.g., 'Academic', 'Casual', 'Medical').
* **Consistency Enforcers:** Toggle between 'Strict' bounds (LLM must mathematically adhere to min/max outliers) or 'Lenient' bounds (LLM can generate novel edge cases).
* **Context Allocation:** A slider dictating what percentage (5% to 95%) of the true dataset should be "fed" into the LLM as structural context vs purely generated.

### Execution Safeguards
* **Token Estimation Engine:** Real-time computation predicting how many tokens a generation task will cost before running.
* **Algebraic "Top-Up" Guarantee:** An autonomous engine that forces the LLM into a bounded while-loop. If the LLM returns 48 rows instead of 50, the system loops back and forces it to generate the remaining 2 rows until a 100% volume match is achieved.
* **Syntax Self-Healing:** Aggressive internal regex and Python AST parsers that automatically fix LLM hallucinations (missing commas, abruptly truncated brackets, Python True/False mappings) without terminating the pipeline.

---

## 3. Parallel Benchmark Pipeline

Rather than generating data blindly, Synthagent runs K-Fold Cross Validation testing to prove whether the synthetic data actually improves downstream predictive models.

### Pipeline Configurator
* **K-Fold Multiplier:** Define how many statistical folds to execute (1-10) to validate variance stability.
* **Missing Value Handlers:** Tell the pipeline to explicitly "Impute" NaNs (Mean/Mode) or "Drop" them prior to ML training.
* **Parallel Execution:** Toggle multiprocessing to run multiple datasets and folds concurrently across your computer's CPU cores.

### Sub-Engine ML Evaluator
Users can select up to **11 distinct Machine Learning Models** to pit against the generated data:
* **Tree Based:** Random Forest, Extreme Gradient Boosting (XGBoost), Gradient Boosting, AdaBoost, Decision Trees.
* **Linear/Distance:** Logistic Regression, Linear Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN).
* **Probabilistic / Deep:** Naive Bayes, Multi-Layer Perceptron (MLP Neural Networks).

---

## 4. Interactive EDA Auditing (Exploratory Data Analysis)

Once data is generated, a unified Dashboard explicitly compares the `Original vs. Augmented` metrics.

### AI Verification
* **AI Audit Flags:** Synthetically generated rows are actively scanned for domain impossibilities (e.g., Negative ages, misspelled categorical enums). Bad rows are explicitly visually flagged in the AG-Grid.

### Statistical Metrics Module
A readout computing the mathematical fidelity of the generated data:
* **Synthetic Fidelity (JSD):** Jensen-Shannon Divergence measuring statistical overlap.
* **Correlation Stability & Shift:** Checks whether the internal relationships between variables (e.g., `BMI` and `Diabetes`) survived generation.
* **Privacy Risk (Memorization):** Scans the synthetic outputs to violently guarantee the LLM didn't just copy-paste real dataset rows. 
* **Novelty Score:** Measures the exact ratio of completely newly fabricated domain data.

### Visual Distributions
* **Overlay Plots:** Users can click on any feature column to spawn a distribution graph overlaying the *Real Data spread (Blue)* against the *Synthetic Data spread (Orange)* to visually verify distribution logic.

---

## 5. Machine Learning Diagnostics Modal

Users can "Inspect" any of the 11 ML models that were run via an interactive modal to understand *why* the model improved or degraded with synthetic data.

* **MLP Loss Curves:** If Neural Networks were tested, users can view the epoch-by-epoch Mean Training Loss curves across all K-Folds.
* **Tree Feature Importances:** For RF and XGBoost, a live Bar Chart ranking exactly which columns the model used to make its decisions. 
* **Linear Coefficients:** For LogR and LinR, a chart mapping exact directional coordinate weights (positive/negative impacts).
* **Fold Skimming:** "Next" and "Previous" cycle buttons allow the user to skim through the individual fold-level diagnostics sequentially.

---

## 6. Executive Reporting Engine

For academic and enterprise sharing, Synthagent features an offline, mathematically formatted PDF Generation engine.

* **One-Click Generation:** Compiles all runs, parameters, and datasets active in the UI into a single scientific document.
* **Executive Summary:** Aggregates a global, cross-dataset statistical summary on the very first page.
* **Geometric PCA Visualization:** Renders 2D Principal Component Analysis (PCA) maps explicitly showing the geometric clustering of the AI's synthetic rows directly layered on top of the original bounds.
* **Trace Appendices:** Exhaustive printouts of the Fold data featuring color-coded row markers distinguishing LLM Context arrays from Raw LLM Synthetic generations.
