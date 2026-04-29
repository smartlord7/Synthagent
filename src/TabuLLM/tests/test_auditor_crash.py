import pandas as pd
import numpy as np
import sys
import os

# Add src/TabuLLM to path if running from within test dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.auditor import auditor

def test_auditor_no_crash():
    print("Creating test data with 'string' dtype...")
    data = {
        'ID': ['ES001', 'ES002', 'ES003', 'ES004', 'ES005'],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Age': [25, 30, 35, 40, 45],
        'Income': [50000, 60000, 70000, 80000, 90000]
    }
    df = pd.DataFrame(data)
    # Force ID to be 'string' dtype (not just object)
    df['ID'] = df['ID'].astype('string')
    
    print(f"Dtypes:\n{df.dtypes}")
    
    sensitive_cols = ['Gender']
    target_col = 'Income'
    
    print("\nTesting get_proxy_risks...")
    try:
        proxies = auditor.get_proxy_risks(df, sensitive_cols)
        print(f"Proxies found: {proxies}")
    except Exception as e:
        print(f"FAILED: get_proxy_risks crashed: {e}")
        sys.exit(1)
        
    print("\nTesting get_structural_metrics...")
    try:
        struct = auditor.get_structural_metrics(df)
        print(f"Structural metrics calculated. Stability: {struct.get('stability')}")
        # ID should be in stability now because it's non-numeric
        if 'ID' not in struct.get('stability', {}):
             print("WARNING: 'ID' not found in stability metrics (it should be as a non-numeric column)")
    except Exception as e:
        print(f"FAILED: get_structural_metrics crashed: {e}")
        sys.exit(1)
        
    print("\nTesting get_fairness_metrics...")
    try:
        fairness = auditor.get_fairness_metrics(df, target_col, sensitive_cols)
        print(f"Fairness metrics calculated.")
    except Exception as e:
        print(f"FAILED: get_fairness_metrics crashed: {e}")
        sys.exit(1)

    print("\n--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_auditor_no_crash()
