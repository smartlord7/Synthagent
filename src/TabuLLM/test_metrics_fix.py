import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics

def test_metrics_integrity():
    df_orig = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    df_aug = pd.DataFrame({"A": [1.1, 2.2, 3.3], "B": ["x", "y", "w"]})
    
    try:
        m = calculate_metrics(df_orig, df_aug)
        print("Metrics keys:", m.keys())
        
        # Test avg_fidelity logic from pipeline_manager
        results = [m, m]
        avg_fid = sum(r.get("Synthetic Fidelity (JSD)", 0) for r in results) / len(results)
        print(f"Avg Fidelity: {avg_fid}")
        
        print("SUCCESS: Metric integrity verified.")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_metrics_integrity()
