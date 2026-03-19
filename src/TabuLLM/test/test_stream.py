import urllib.request
import json
import time

context = 'General Dataset Patterns: [{"patient_id":"S009","age":58,"sex":"Male","sarcoma_type":"Liposarcoma","tumor_size_cm":5.9,"stage":"II","metastasis":"No","treatment":"Surgery+Radiation","follow_up_months":26,"status":"Alive"},{"patient_id":"S004","age":53,"sex":"Female","sarcoma_type":"Leiomyosarcoma","tumor_size_cm":6.0,"stage":"II","metastasis":"No","treatment":"Surgery+Radiation","follow_up_months":30,"status":"Alive"}]'
instruct = 'Style: Concise / Standard\nConsistency: Strict (Perfectly uniform)\nUniqueness: Allow Duplicates (Standard)'
cols = ['patient_id', 'age', 'sex', 'sarcoma_type', 'tumor_size_cm', 'stage', 'metastasis', 'treatment', 'follow_up_months', 'status']

prompt = f"""
SYSTEM CONTEXT: {context}
USER INSTRUCTIONS: {instruct}
DATA STRUCTURE: The columns are {cols}.
CRITICAL: RAW NUMBERS only.
MUST Output JSON: {{"data": [...]}}

TASK:
Generate 8 NEW rows.

Return JSON.
"""

payload = {
    "model": "google/gemma-3-4b",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.3
}

print("Sending request to local LLM...")
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=60) as response:
        print("Response received! Reading stream incrementally...")
        full_content = b""
        chunk_count = 0
        while True:
            chunk = response.read(128)  # Read 128 bytes at a time
            if not chunk:
                break
            full_content += chunk
            chunk_count += 1
            # Print a dot for every chunk to see if it's streaming infinitely
            print(".", end="", flush=True)
            if chunk_count % 10 == 0:
                print(f"[{len(full_content)} bytes]", end="", flush=True)
            if len(full_content) > 100000:  # Safety limit 100KB
                print("\nStream exceeds 100KB safety limit! Aborting loop.")
                break
        
        print("\nFinished reading.")
        print(f"Total size: {len(full_content)} bytes")
        print("\n--- CONTENT ---")
        print(full_content.decode('utf-8', errors='ignore'))
        print("--- END CONTENT ---")

except Exception as e:
    print(f"\nError: {e}")
