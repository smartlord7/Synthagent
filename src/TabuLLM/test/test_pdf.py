import sys
import os
import pandas as pd
import io

# Add src to path
sys.path.append(r"c:\Users\Artur Seco\Documents\Git Repositories\TabuLLM\src\TabuLLM")

from services.report_generator import generate_comprehensive_pdf

# Dummy Data
report_data = {
    "executive_summary": {"avg_fidelity": "85.2%", "status": "Test Run"},
    "datasets": {
        "test_ds": {
            "filename": "sarcoma_test.csv",
            "description": "This is a test description for the Sarcoma dataset.",
            "params": {
                "generation": {"algorithm": "gemma-3-4b", "tone": "Professional", "consistency": "Strict"},
                "model": {"type": "Random Forest", "n_estimators": 100}
            },
            "metrics": {"Fidelity": "88.1%", "Privacy": "92.0%"},
            "plots": [],
            "samples": {
                "orig": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
                "aug": pd.DataFrame({"A": [1, 2, 5], "B": [3, 4, 6]})
            }
        }
    }
}

try:
    pdf_bytes = generate_comprehensive_pdf(report_data)
    with open(r"c:\Users\Artur Seco\Documents\Git Repositories\TabuLLM\src\TabuLLM\tmp_test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("SUCCESS: PDF generated successfully.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
