import pandas as pd
import os
import json
from synthagent import Synthagent

def main():
    print("--- Synthagent Library: Full-Feature Demo ---")
    
    # 1. Initialize the Library
    # You can specify 'local' (e.g., LM Studio) or 'gemini'
    lib = Synthagent(provider="local")
    
    # 2. Check Available Models
    try:
        models = lib.get_models()
        print(f"\n[1] Connected Backend Models: {[m['id'] for m in models]}")
    except:
        print("\n[1] Models: Backend not reachable, using defaults.")

    # 3. Load Sample Dataset
    data = {
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "attrition": [0, 0, 1, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    target = ["attrition"]
    print(f"\n[2] Loaded Dataset: {len(df)} rows, {len(df.columns)} columns.")

    # 4. Get AI Insights (Description)
    print("\n[3] Generating AI Dataset Insights...")
    # (In real use, this calls the LLM)
    # insights = lib.get_insights(df)
    # print(f"    Insights: {insights[:100]}...")
    print("    -> Logic: calls LocalOpenAIProvider.get_dataset_description")

    # 5. Run Comprehensive Audit (Health & Ethics)
    print("\n[4] Running Deep Health & Ethics Audit...")
    audit = lib.run_audit(df, targets=target)
    print(f"    - Sparsity: {audit['structural']['overall_sparsity']*100:.1f}%")
    print(f"    - Privacy Risk: {audit['privacy']['risk_level']} (Uniqueness: {audit['privacy']['uniqueness_ratio']*100:.1f}%)")
    print(f"    - Fairness (Disparate Impact): {list(audit['fairness'].keys())}")

    # 6. Strategic AI Pondering (The "Plan")
    print("\n[5] AI Strategic Pondering (Planning)...")
    # strategy = lib.ponder_strategy(df, targets=target)
    # print(f"    - Guidance: {strategy[:150]}...")
    print("    -> Logic: calls ponder_audit.j2 template with full metrics context")

    # 7. High-Reliability Augmentation (K-Fold + ML)
    print("\n[6] Starting High-Reliability Augmentation (2-Fold CV)...")
    print("    - Evaluating Models: ['rf', 'lr']")
    
    # Note: To run this for real, you need the LLM server active.
    # results = lib.augment(df, targets=target, num_rows=20, folds=2, ml_models=["rf", "lr"])
    
    # For the sake of a clean demo that always prints something:
    fake_results = {
        "dataset_id": "demo-ds",
        "avg_fidelity": 0.88,
        "overall_ml": {
            "rf": {"accuracy": {"real": 0.82, "aug": 0.84, "delta": 0.02}},
            "lr": {"accuracy": {"real": 0.78, "aug": 0.79, "delta": 0.01}}
        }
    }
    print(f"    - Avg Fidelity: {fake_results['avg_fidelity']}")
    print(f"    - ML Benchmarks: {list(fake_results['overall_ml'].keys())}")

    # 8. Generate Final Audit Report
    output_pdf = "synthagent_full_feature_demo.pdf"
    print(f"\n[7] Exporting Professional Audit Report to: {output_pdf}")
    lib.generate_pdf_report(fake_results, output_pdf)
    
    print("\n--- Demo Complete! ---")

if __name__ == "__main__":
    main()
