import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# output folder
output_dir = "uci_medical_datasets"
os.makedirs(output_dir, exist_ok=True)

# dataset registry (name, uci id)
datasets = [
    ("heart_disease", 45),
    ("breast_cancer_wisconsin_diagnostic", 17),
    ("diabetes", 34),
    ("heart_failure_clinical_records", 519),
    ("parkinsons", 174),
    ("indian_liver_patient_dataset", 225)
]

for name, dataset_id in datasets:
    print(f"\nDownloading: {name} (ID={dataset_id})")

    try:
        # fetch dataset
        dataset = fetch_ucirepo(id=dataset_id)

        # features and targets
        X = dataset.data.features
        y = dataset.data.targets

        # merge into one dataframe
        if y is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X

        # save CSV
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)

        # metadata
        metadata_path = os.path.join(output_dir, f"{name}_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(str(dataset.metadata))

        # variables
        variables_path = os.path.join(output_dir, f"{name}_variables.txt")
        with open(variables_path, "w", encoding="utf-8") as f:
            f.write(str(dataset.variables))

        print(f"Saved CSV: {csv_path}")
        print(f"Saved metadata: {metadata_path}")
        print(f"Saved variables: {variables_path}")

    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("\nAll datasets processed.")