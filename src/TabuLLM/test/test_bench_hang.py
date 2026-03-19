import sys
sys.path.append('.')
import pandas as pd
import json
from services.llm_client import generate_bulk_data
from utils.metrics import calculate_metrics

print("Loading data...", flush=True)
try:
    df_orig = pd.read_csv("sarcoma2.csv")
except:
    # Create dummy data if file not found
    df_orig = pd.DataFrame({
        'feat1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feat2': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'feat3': [10.5, 20.3, 15.2, 11.1, 14.4, 19.9, 18.2, 12.1, 13.3, 17.5]
    })

print(f"Dataset Rows: {len(df_orig)}", flush=True)

# Sample fraction Context (e.g. 0.2)
context_df = df_orig.sample(frac=0.2)
num_empty = len(df_orig) - len(context_df)

print(f"Context Rows: {len(context_df)}, To Generate: {num_empty}", flush=True)

context_str = f"General Dataset Patterns: {context_df.to_json(orient='records')}"
instructions = "Style: Concise / Standard\nConsistency: Strict (Perfectly uniform)\nUniqueness: Allow Duplicates (Standard)"
all_columns = list(df_orig.columns)

print("Calling generate_bulk_data...", flush=True)
print("\n--- GENERATING CURL COMMAND ---")
payload = {
    "model": "google/gemma-3-4b",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.3
}

with open("payload.json", "w") as f:
    json.dump(payload, f)

print("curl -X POST -H \"Content-Type: application/json\" -d @payload.json http://127.0.0.1:1234/v1/chat/completions")
print("--- END CURL COMMAND ---")
sys.exit(0)
result_bundle = generate_bulk_data(
    [], # Empty partial payload
    context_str,
    all_columns,
    "bulk",
    instructions,
    0.3, # temperature
    num_empty=num_empty
)

print("Done generating.", flush=True)
if result_bundle:
    print("Keys in result:", result_bundle.keys())
    print("Data length:", len(result_bundle["data"]))
else:
    print("Result bundle is None")


