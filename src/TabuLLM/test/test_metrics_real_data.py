import pandas as pd
import json
from utils.metrics import calculate_metrics

print("Loading real sarcoma2.csv...")
df_orig = pd.read_csv("sarcoma2.csv")

# Sample context
context_df = df_orig.sample(frac=0.2)

# Create dummy generated data with exact same columns
filled_data = []
for i in range(8):
    row = {}
    for col in df_orig.columns:
        if df_orig[col].dtype == 'object':
             row[col] = df_orig[col].dropna().unique()[0] # pick first category
        else:
             row[col] = df_orig[col].mean() # use mean for numbers
    filled_data.append(row)

generated_df = pd.DataFrame(filled_data)

print("Concatenating augmented df...")
aug_df = pd.concat([context_df, generated_df], ignore_index=True)

print("Calling calculate_metrics on real data structures...")
try:
    metrics = calculate_metrics(df_orig, aug_df)
    print("\n--- METRICS ---")
    print(metrics)
    print("--- SUCCESS ---")
except Exception as e:
    print(f"\nCRASH DETECTED: {e}")
