import uuid
from services.data_manager import create_empty_dataframe, df_to_aggrid_format

# Initial Data Setup
initial_df = create_empty_dataframe(10, 5)

# Create a reusable style function
ai_highlight_style = {
    "styleConditions": [
        {
            "condition": "params.data['is_ai_' + params.colDef.field]",
            "style": {"backgroundColor": "#e8f5e9", "borderLeft": "3px solid #4caf50"}
        }
    ]
}

# Update initial columns
initial_cols = []
for i, col in enumerate(initial_df.columns):
    col_def = {"field": col, "headerName": col, "cellStyle": ai_highlight_style}
    if i == 0:
        col_def["checkboxSelection"] = True
        col_def["headerCheckboxSelection"] = True
    initial_cols.append(col_def)

# Add the Audit Column (Read-only)
initial_cols.append({
    "field": "ai_audit",
    "headerName": "Generation Audit",
    "editable": False,
    "cellStyle": {"color": "#666", "fontSize": "0.8em", "fontStyle": "italic"}
})

initial_data = df_to_aggrid_format(initial_df)
for row in initial_data:
    row['_row_id'] = str(uuid.uuid4())

row_class_rules = {
    "row-context": "params.data.is_context",
    "row-target": "params.data.is_target",
}
