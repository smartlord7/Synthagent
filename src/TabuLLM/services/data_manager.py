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