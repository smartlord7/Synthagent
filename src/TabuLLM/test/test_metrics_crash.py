import pandas as pd
from utils.metrics import calculate_metrics

print("Creating dummy original data...")
df_orig = pd.DataFrame({
    'patient_id': ['S001', 'S002'],
    'age': [30, 45],
    'sex': ['Male', 'Female'],
    'sarcoma_type': ['TypeA', 'TypeB'],
    'tumor_size_cm': [3.5, 4.2],
    'stage': ['I', 'II'],
    'metastasis': ['No', 'No'],
    'treatment': ['Surg', 'Rad'],
    'follow_up_months': [12, 24],
    'status': ['Alive', 'Alive']
})

print("Creating dummy augmented data...")
df_aug = pd.DataFrame({
    'patient_id': ['S001', 'S002', 'S003', 'S004'],
    'age': [30, 45, 32, 44],
    'sex': ['Male', 'Female', 'Male', 'Female'],
    'sarcoma_type': ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
    'tumor_size_cm': [3.5, 4.2, 3.6, 4.1],
    'stage': ['I', 'II', 'I', 'II'],
    'metastasis': ['No', 'No', 'No', 'No'],
    'treatment': ['Surg', 'Rad', 'Surg', 'Rad'],
    'follow_up_months': [12, 24, 15, 21],
    'status': ['Alive', 'Alive', 'Alive', 'Alive']
})

print("Calling calculate_metrics...")
try:
    metrics = calculate_metrics(df_orig, df_aug)
    print("\n--- METRICS ---")
    print(metrics)
    print("--- SUCCESS ---")
except Exception as e:
    print(f"\nCRASH DETECTED: {e}")
