# TabuLLM Pipeline Architecture (Complete System Design)

This document visualizes the exact, exhaustive data flow, token compression, generation algorithms, error repairing, and machine learning benchmarking steps in the TabuLLM codebase.

---

## 0. High-Level Generation Overview
A simplified view of how data travels from your source file to the final verified synthetic output.

```mermaid
graph LR
    classDef input fill:#2c3e50,stroke:#34495e,color:#fff,stroke-width:2px;
    classDef process fill:#2980b9,stroke:#1f6391,color:#fff,stroke-width:2px;
    classDef ai fill:#8e44ad,stroke:#71368a,color:#fff,stroke-width:2px;
    classDef safety fill:#c0392b,stroke:#962d22,color:#fff,stroke-width:2px;
    classDef result fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px;

    Source[(Source Dataset)]:::input --> Rules[Rule Inference & User Rules]:::process
    Rules --> Context[Context Sampling<br/>Selection of reference rows]:::process
    
    Context --> LLM[Generative Synthesis<br/>LLM-powered dataset creation]:::ai
    
    LLM --> Guard[Hierarchical Guardrails<br/>Pydantic self-correction loop]:::safety
    Guard -.-> |"Automatic Fix Turn"| LLM
    
    Guard --> Final[(Verified Synthetic Data)]:::result
```

---

## 1. Global Pipeline Control & Parallel Processing
How datasets are ingested, evaluated synchronously via multiprocessing, and heavily pre-processed to minimize token expenditure before interacting with the LLM API.

```mermaid
graph TD
    classDef ds fill:#1e1e24,stroke:#444,stroke-width:2px;
    classDef pool fill:#4b184f,stroke:#26052c,color:#fff,stroke-width:2px;
    classDef compress fill:#804000,stroke:#3b1f02,color:#fff,stroke-width:2px;

    T[Incoming Datasets List] --> M(Python Multiprocessing Pool):::pool
    M --> Worker1[Process Node: e.g., dataset_1.csv]
    M --> Worker2[Process Node: e.g., dataset_2.csv]
    
    Worker1 --> KF1{Stratified K-Fold Validation Loop<br/>Folds 1 to K}
    
    KF1 --> Stage1[Split Fold Data into Train/Test]:::ds
    Stage1 --> Stage2[Select Context Seed Rows]:::ds
    
    Stage2 --> MemOpt[Token Compression: 'Compact JSON' Format<br/>Extracts keys into headers, reduces data to strict 2D array]:::compress
    
    MemOpt --> Stage3[Generative LLM Broker Layer]:::ds
    Stage3 --> Stage4[Statistical Distance & Downstream ML Benchmarking]:::ds
    Stage4 --> Store[(Save Complete Fold Object to Session Store)]
```

---

## 2. Train/Test Constraints & Context Sampling Strategies
Visualizing how original data is split algebraically. The system utilizes configurable unsupervised sampling strategies to intelligently pick the `Context Rows` sent to the LLM.

```mermaid
graph TD
    classDef algo fill:#0a3d62,stroke:#032030,color:#fff,stroke-width:2px;
    classDef data fill:#1e3d59,stroke:#0f202a,color:#fff,stroke-width:2px;
    classDef target fill:#1a4a38,stroke:#0a2a1a,color:#fff,stroke-width:2px;
    classDef strat fill:#351a4a,stroke:#1a0820,color:#fff,stroke-width:2px;

    MainDS[Original Fold Data<br/>100% Size]:::data --> Split[Scikit-Learn Train_Test_Split]:::algo
    
    Split --> Train[Train Split<br/>e.g. 80%]:::data
    Split --> Test[Hidden Test Split<br/>e.g. 20%]:::data
    
    Train --> Sampler{Context Selection Strategy}:::strat
    Sampler -.-> |"Dense: K-Means Clustering Core Centers"| Context
    Sampler -.-> |"Diverse: K-Means Max-Distance Outliers"| Context
    Sampler -.-> |"Random: Shuffled Index Sub-selection"| Context
    
    Context[Context Data<br/>Few-Shot Seed Rows]:::data --> Task[Row Generation Target: 'num_empty'<br/>Calculation: Train Split Size - Context Array Size]:::target
    Train --> Task
    
    Context -.-> |"Math Constraint: Context Rows + Generated 'num_empty' Rows = Exact Train Split Volume"| MathNote((Match Train))
    MathNote -.-> |"Output Restitution: Recombined Train Volume + Filtered Test Split = Precise Original CSV Row Count"| FinalMath((100% Reconstruction))
```

---

## 3. The LLM Target Chunking & Error Recovery Loops
Because deep learning models suffer context collapse, `generate_bulk_data` slices the requirement into fixed `chunk_size` windows, forces autonomous text repairs via `robust_json_loads`, and runs a deterministic top-up recursion if the LLM halts generation prematurely.

```mermaid
sequenceDiagram
    autonumber
    
    participant PM as Pipeline Manager
    participant LC as Chunking Scheduler
    participant AI as Local API / Gemini Client
    participant Out as Verified Data Output

    Note over PM, LC: Task parameters: generate exactly 64 empty rows.
    PM->>LC: call generate_bulk_data(num_empty=64, chunk_size=30)
    
    Note right of LC: 1. Calculate Request Blocks: ceil(64 / 30) = 3 calls
    LC->>LC: Request Queue: [Array Size 30], [Array Size 30], [Array Size 4]
    
    LC->>AI: Chunk 1 Prompt: "Synthesize 30 rows"
    AI-->>LC: JSON [Returns Missing Bracket Syntax Error]
    
    rect rgb(70, 30, 30)
    Note over LC, AI: robust_json_loads fails to parse. Autonomous Repair triggers natively.
    LC->>AI: Repair Prompt: "Fix trailing characters at specific JSON index"
    AI-->>LC: JSON [Returns 28 valid rows] (Halted two rows early)
    end
    
    LC->>AI: Chunk 2 Prompt: "Synthesize 30 rows"
    AI-->>LC: JSON [Returns exact 30 valid rows]
    
    LC->>AI: Chunk 3 Prompt: "Synthesize 4 rows"
    AI-->>LC: JSON [Returns 2 valid rows]
    
    Note right of LC: Current Array Length = 60 rows.<br/>Shortfall Deficit = Target(64) - Current(60) = 4 rows.
    
    rect rgb(20, 50, 40)
    Note over LC, AI: 2. Mandatory Top-Up Generation Loop (Maximum 5 Retries)
    LC->>LC: Calculate missing arrays exact value: 4
    LC->>AI: Shortfall Prompt: "Synthesize exactly 4 missing deficit rows"
    AI-->>LC: JSON [Returns 5 rows]
    end
    
    LC->>LC: Post-Processing Truncation check: Trim 65 rows down to exactly 64 limit.
    LC->>Out: from_compact_json() (Decodes 2D list of lists back to list of dictionaries)
    Out-->>PM: Deliver exactly 64 arrays effectively
```

---

## 4. The Hierarchical Generation & Self-Correction Pipeline
How the system orchestrates automated rule inference, hierarchical merging, and the iterative Pydantic-based repair loop.

```mermaid
sequenceDiagram
    autonumber
    
    participant UI as Dash Components (UI)
    participant INF as RuleInference Service
    participant CB as Generation Callback
    participant LC as LLM Client Broker
    participant VAL as RuleValidator (Pydantic)
    participant AI as Generative LLM (Gemini/Gemma)

    UI->>CB: Click Generate (Toggles: UserRules, Boundaries)
    
    rect rgb(30, 30, 50)
    Note over CB, INF: Static Analysis Phase
    CB->>INF: infer_dataset_rules(original_df)
    INF-->>CB: Hierarchy: Integrity (L1) + Boundaries (L2)
    end
    
    CB->>LC: call generate_bulk_data(rules_hierarchy)
    
    loop Per Chunk (size=30)
        LC->>AI: Initial Generation (System Instructions + Context)
        AI-->>LC: Raw JSON Chunk (e.g., 30 rows)
        
        rect rgb(50, 40, 20)
        Note over LC, VAL: Validation Turn
        LC->>VAL: validate_batch(chunk, active_pydantic_model)
        VAL-->>LC: Found N Violations (e.g., Age > Max OR String in Num field)
        end
        
        break No Violations or Max Retries met
            Note over LC, VAL: Proceed to next chunk
        end
        
        rect rgb(60, 20, 20)
        Note over LC, AI: Self-Correction Turn
        LC->>AI: Fix Prompt (Template: generation_fix.j2 + Detailed Violations)
        AI-->>LC: Corrected JSON Chunk
        end
    end
    
    LC-->>CB: Consolidated Validated Data Records
    CB-->>UI: Update Grid (Mark successful rows in Green)
```

---

## 5. Pydantic Hierarchical Guardrail Layer
Instead of simple heuristics, TabuLLM uses dynamic Pydantic models to categorize and enforce constraints across three layers of strictness.

```mermaid
graph LR
    classDef integrity fill:#b33939,stroke:#4d1313,color:#fff,stroke-width:2px;
    classDef boundaries fill:#227093,stroke:#113a4d,color:#fff,stroke-width:2px;
    classDef user fill:#218c74,stroke:#10473a,color:#fff,stroke-width:2px;
    classDef validator fill:#40407a,stroke:#20203d,color:#fff,stroke-width:2px;

    L1[Level 1: Integrity<br/>Datatypes & Structure]:::integrity --> Model{Pydantic Model Generator}:::validator
    L2[Level 2: Boundaries<br/>Min/Max & Domain Sets]:::boundaries --> |Toggle Controlled| Model
    L3[Level 3: User Rules<br/>Custom UI Definitions]:::user --> |Toggle Controlled| Model
    
    Model --> Filter[Validation Runner]:::validator
    Raw[Raw LLM Output] --> Filter
    
    Filter -.-> |"Violation Found!"| Repair[LLM Correction Feedback Turn]:::validator
    Filter -.-> |Passes all Active Levels| Sink[Accepted Synthetic Row]
    
    Repair --> Filter
```

---

## 5. Downstream Machine Learning & Missing Value Engines
The `MLEvaluator` proves the systemic efficacy of the generated data. It cleans unhashable lists/NaNs (via drop or Mode/Mean imputation), trains XGBoost/Scikit-Learn estimators, extracts structural feature importances, and ultimately runs predictions solely against the unseen Test layer.

```mermaid
graph TD
    classDef metrics fill:#2f4f4f,stroke:#1a2b2b,color:#fff,stroke-width:2px;
    classDef ml fill:#0a3d62,stroke:#032030,color:#fff,stroke-width:2px;
    classDef test fill:#214436,stroke:#0d1d17,color:#fff,stroke-width:2px;
    classDef feature fill:#4a1c1c,stroke:#260505,color:#fff,stroke-width:2px;

    Ctx[Context Row Sub-sample] --> Aug[Augmented Train Split Construction]
    Syn[Synthesized Data Grid] --> Aug

    Aug --> Fid{Fidelity Processor<br/>Continuous Data: KS-Test<br/>Categorical: Hellinger Distance}:::metrics
    OrigTrain[Original Train Split] --> Fid
    
    Fid -.-> |"Calculated Divergence Probabilities"| FoldResult[Push to Fold Results Object]

    Aug --> MLEval{MLEvaluator Benchmarking Module}:::ml
    OrigTrain --> MLEval
    
    MLEval --> ML1[Fit Random Forest / Gradient Boost Models on Augmented Data]:::ml
    MLEval --> ML2[Fit Random Forest / Gradient Boost Models on Baseline Original Data]:::ml
    
    ML1 --> NA[Apply missing value 'na_strategy'<br/>Drop / Mean Imputation / Mode Imputation<br/>Cast Dicts/Lists to str scalars for safety]
    ML2 --> NA
    
    Test[Strictly Hidden Test Split]:::test --> Bench
    NA --> Bench[Benchmark Dual Model Predictions actively onto Test Split Output]:::ml
    
    Bench -.-> |"Scoring (F1-score, RMSE, Accuracy, R² Deltas)"| FoldResult
    
    ML1 --> Extract[Model Structure Parser<br/>Extract instance.coef_ or instance.feature_importances_]:::feature
    ML2 --> Extract
    Extract -.-> |"Importance Weights Float Array"| FoldResult
```

---

## 6. Global Artifact Assembly & Visual Rendering Nexus
Decentralized fold outputs are captured by the manager, averaged, and concatenated to form the master output arrays that populate the front-end DASH data grids, scatter-plots, and final PDF audit generator.

```mermaid
graph TD
    classDef store fill:#103942,stroke:#091d21,color:#fff,stroke-width:2px;
    classDef inter fill:#636e72,stroke:#2d3436,color:#fff,stroke-width:2px;

    F1[Evaluation Output: Fold 1] --> Agg[Thread Pipeline Aggregator]
    F2[Evaluation Output: Fold 2] --> Agg
    F3[Evaluation Output: Fold 3] --> Agg
    
    Agg --> SynList[all_synthetic_data: Raw Generation Array]:::store
    Agg --> AugList[all_augmented_data: Master Fold-Tracked Grid]:::store
    Agg --> Stats[overall_ml: Score Averages and Standard Deviations calculated]:::store
    
    Stats -.-> |"Populates Plotly Figures & Statistical Reference Tables"| Report[DASH React UI Component Layout]
    AugList -.-> |"Paginates into Frontend Ag-Grid Layouts"| Report
    
    SynList --> Audit(PDF Global Report Processor):::inter
    Stats --> Audit
```
