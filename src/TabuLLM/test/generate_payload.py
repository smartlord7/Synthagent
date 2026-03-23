import json

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

with open("payload.json", "w") as f:
    json.dump(payload, f, indent=2)

print("Generated payload.json successfully.")
