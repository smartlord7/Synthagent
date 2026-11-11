import pandas as pd
import io
import base64
from utils.logging_config import logger

def parse_uploaded_file(contents: str, filename: str) -> pd.DataFrame:
    """
    Decodes the base64 string from a Dash upload component and
    loads it into a Pandas DataFrame.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return pd.DataFrame({"Error": [f"Could not process {filename}"]})

    return df

def create_empty_dataframe(num_rows: int = 10, num_cols: int = 5) -> pd.DataFrame:
    """Creates an empty dataframe to initialize the table."""
    columns = [f"Column_{i + 1}" for i in range(num_cols)]
    data = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    return pd.DataFrame(data, columns=columns)

def get_eda_metrics(df: pd.DataFrame) -> dict:
    """Calculates basic exploratory data metrics."""
    if df.empty:
        return {}

    metrics = {
        "columns": list(df.columns),
        "shape": df.shape,
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numerical_summary": df.describe().to_dict()
    }
    return metrics

def df_to_aggrid_format(df: pd.DataFrame) -> list[dict]:
    """Converts DataFrame to list-of-dicts for AG Grid."""
    return df.to_dict('records')

def aggrid_to_df(grid_data: list[dict]) -> pd.DataFrame:
    """Converts AG Grid list back to DataFrame."""
    return pd.DataFrame(grid_data)

def select_context_rows(df: pd.DataFrame, strategy: str = "random", n_rows: int = 20) -> list[dict]:
    """Slices context rows based on algorithm selections for maximizing density or diversity profiles."""
    if df.empty: return []
    target_count = min(n_rows, len(df))
    logger.info(f"Context Selection: Strategy='{strategy}' | RequestedRows={n_rows} | PoolSize={len(df)}")
    
    # Filter out internal/technical flags
    cols_to_use = [c for c in df.columns if c not in ['_row_id', 'is_context', 'is_target', 'ai_audit'] and not c.startswith('is_ai_')]
    sub_df = df[cols_to_use]

    if strategy == "dense":
        # 📊 DENSE Selection: Rows with maximum non-null values filled
        non_null_counts = sub_df.notnull().sum(axis=1)
        selected_indices = non_null_counts.nlargest(target_count, keep='first').index
        logger.info(f"Dense Selection: Selected {len(selected_indices)} rows with highest non-null count.")
        return df.loc[selected_indices].to_dict('records')

    elif strategy == "diverse":
        # 🌐 DIVERSE Selection: Uses K-Means to identify diverse centroid profiles
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            num_df = sub_df.select_dtypes(include=[np.number]).fillna(0)
            if num_df.empty or num_df.shape[1] == 0:
                logger.warning("Diverse Selection: No numeric columns found. Falling back to Random.")
                return df.sample(target_count).to_dict('records')
                
            # Gaussian Normalization
            num_norm = (num_df - num_df.mean()) / (num_df.std() + 1e-9)
            num_norm = num_norm.fillna(0) 
            
            k = min(target_count, len(num_norm))
            if k == 0: return []
            
            logger.info(f"Diverse Selection: Clustering into {k} groups via K-Means.")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(num_norm)
            centers = kmeans.cluster_centers_
            
            selected_indices = []
            for center in centers:
                distances = np.linalg.norm(num_norm.values - center, axis=1)
                best_idx = num_norm.index[np.argmin(distances)]
                if best_idx not in selected_indices:
                    selected_indices.append(best_idx)
                    
            # Backfill remaining
            if len(selected_indices) < target_count:
                 remaining_idx = df.index.difference(selected_indices)
                 extra_count = target_count - len(selected_indices)
                 if not remaining_idx.empty:
                     extra = df.loc[remaining_idx].sample(min(extra_count, len(remaining_idx))).index
                     selected_indices.extend(extra.tolist())
            
            logger.info(f"Diverse Selection: Final set contains {len(selected_indices)} unique rows.")
            return df.loc[selected_indices].to_dict('records')
        except Exception as e:
            logger.error(f"Diverse Selection failure: {e}. Falling back to Random.")
            return df.sample(target_count).to_dict('records')

    # 🎲 RANDOM Selection (Default)
    logger.info(f"Random Selection: Picking {target_count} rows at random.")
    return df.sample(target_count).to_dict('records')