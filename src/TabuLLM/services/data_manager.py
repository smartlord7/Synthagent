import pandas as pd
import io
import base64


def parse_uploaded_file(contents: str, filename: str) -> pd.DataFrame:
    """
    Decodes the base64 string from a Dash upload component and
    loads it into a Pandas DataFrame.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        print(f"Error processing file: {e}")
        # Return an empty dataframe with an error column as a fallback
        return pd.DataFrame({"Error": [f"Could not process {filename}"]})

    return df


def create_empty_dataframe(num_rows: int = 10, num_cols: int = 5) -> pd.DataFrame:
    """
    Creates an empty dataframe to initialize the table when no dataset is uploaded,
    allowing users to add values, columns, and rows manually.
    """
    columns = [f"Column_{i + 1}" for i in range(num_cols)]
    data = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    return pd.DataFrame(data, columns=columns)


def get_eda_metrics(df: pd.DataFrame) -> dict:
    """
    Calculates basic exploratory data metrics to be used for plotting
    and feeding context to the LLM.
    """
    if df.empty:
        return {}

    metrics = {
        "columns": list(df.columns),
        "shape": df.shape,  # (rows, columns)
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        # .describe() handles numerical stats (mean, std, min, max, etc.)
        "numerical_summary": df.describe().to_dict()
    }

    return metrics


def df_to_aggrid_format(df: pd.DataFrame) -> list[dict]:
    """
    Converts a Pandas DataFrame into the 'records' format required by dash-ag-grid.
    """
    return df.to_dict('records')


def aggrid_to_df(grid_data: list[dict]) -> pd.DataFrame:
    """
    Converts dash-ag-grid data back into a Pandas DataFrame for processing.
    """
    return pd.DataFrame(grid_data)


def select_context_rows(df: pd.DataFrame, strategy: str = "random", n_rows: int = 20) -> list[dict]:
    """
    Slices context rows based on algorithm selections for maximizing density or diversity profiles.
    """
    if df.empty: return []
    n_rows = min(n_rows, len(df))
    
    # Filter out internal/technical flags to avoid skewing distance models
    cols_to_use = [c for c in df.columns if c not in ['_row_id', 'is_context', 'is_target', 'ai_audit'] and not c.startswith('is_ai_')]
    sub_df = df[cols_to_use]

    if strategy == "dense":
        # 📊 DENSE Selection: Rows with maximum non-null values filled
        non_null_counts = sub_df.notnull().sum(axis=1)
        # Handle ties by picking first indices
        selected_indices = non_null_counts.nlargest(n_rows, keep='first').index
        return df.loc[selected_indices].to_dict('records')

    elif strategy == "diverse":
        # 🌐 DIVERSE Selection: Uses K-Means to identify diverse centroid profiles
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Select numeric distributions only for distance metrics
            num_df = sub_df.select_dtypes(include=[np.number]).fillna(0)
            
            if num_df.empty or num_df.shape[1] == 0:
                print("No numeric columns found for Diverse Selection. Falling back to Random.")
                return df.sample(n_rows).to_dict('records')
                
            # Gaussian Normalization to standard scalar
            num_norm = (num_df - num_df.mean()) / (num_df.std() + 1e-9)
            num_norm = num_norm.fillna(0) 
            
            k = min(n_rows, len(num_norm))
            if k == 0: return []
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(num_norm)
            centers = kmeans.cluster_centers_
            
            selected_indices = []
            for center in centers:
                distances = np.linalg.norm(num_norm.values - center, axis=1)
                best_idx = num_norm.index[np.argmin(distances)]
                if best_idx not in selected_indices:
                    selected_indices.append(best_idx)
                    
            # Backfill remaining if cluster centroids hit duplicates
            if len(selected_indices) < n_rows:
                 remaining_idx = df.index.difference(selected_indices)
                 extra_count = n_rows - len(selected_indices)
                 if not remaining_idx.empty:
                     extra = df.loc[remaining_idx].sample(min(extra_count, len(remaining_idx))).index
                     selected_indices.extend(extra.tolist())
                     
            return df.loc[selected_indices].to_dict('records')
        except Exception as e:
            print(f"Diverse Selection failure: {e}. Falling back to Random.")
            return df.sample(n_rows).to_dict('records')

    # 🎲 RANDOM Selection (Default)
    return df.sample(n_rows).to_dict('records')