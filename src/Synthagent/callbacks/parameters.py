import dash
from dash import Input, Output, State, MATCH, ALL, dcc, html
import dash_bootstrap_components as dbc
from core import app
from services.llm_client import get_dataset_description
import pandas as pd

# --- DYNAMIC PARAMETER LOGIC ---
@app.callback(
    Output({"type": "custom-params-container", "index": MATCH}, "children"),
    Input({"type": "btn-add-param", "index": MATCH}, "n_clicks"),
    State({"type": "custom-params-container", "index": MATCH}, "children"),
    State({"type": "btn-add-param", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def add_custom_param_row_pm(n_clicks, children, btn_id):
    if not n_clicks: return dash.no_update
    dataset_id = btn_id["index"]
    
    new_row = dbc.Row([
        dbc.Col(
            dbc.Input(id={'type': 'cp-name', 'index': dataset_id, 'row': n_clicks}, placeholder="Rule Name (e.g. Price Range)", size="sm"),
            width=3),
        dbc.Col(dbc.Select(
            id={'type': 'cp-type', 'index': dataset_id, 'row': n_clicks},
            options=[{'label': 'Text Instruction', 'value': 'text'}, {'label': 'Specific Options', 'value': 'options'},
                     {'label': 'Number Range', 'value': 'number'}],
            value='text', size="sm"
        ), width=3),
        dbc.Col([
            dbc.Input(id={'type': 'cp-val-text', 'index': dataset_id, 'row': n_clicks}, type="text", placeholder="e.g. Under 50 words...",
                      size="sm", style={'display': 'block'}),
            dbc.InputGroup([
                dbc.InputGroupText("Min"),
                dbc.Input(id={'type': 'cp-val-min', 'index': dataset_id, 'row': n_clicks}, type="number"),
                dbc.InputGroupText("Max"),
                dbc.Input(id={'type': 'cp-val-max', 'index': dataset_id, 'row': n_clicks}, type="number")
            ], id={'type': 'cp-val-num-group', 'index': dataset_id, 'row': n_clicks}, size="sm", style={'display': 'none'})
        ], width=5),
        dbc.Col(
            dbc.Button(html.I(className="bi bi-trash"), id={'type': 'btn-remove-row', 'index': dataset_id, 'row': n_clicks}, color="danger", outline=True, size="sm", className="border-0 opacity-50 hover-opacity-100"),
            width=1, className="text-end"
        )
    ], className="mb-2 align-items-center g-2", id={'type': 'custom-param-row-wrapper', 'index': dataset_id, 'row': n_clicks})
    
    children = list(children) if children else []
    children.append(new_row)
    return children


@app.callback(
    Output({"type": "custom-params-container", "index": MATCH}, "children", allow_duplicate=True),
    Input({"type": "btn-remove-row", "index": MATCH, "row": ALL}, "n_clicks"),
    State({"type": "custom-params-container", "index": MATCH}, "children"),
    prevent_initial_call=True
)
def remove_custom_param_row_pm(n_clicks_list, children):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return dash.no_update
    
    # Parse which row stimulated the delete
    triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    import json
    triggered_id = json.loads(triggered_id_str)
    row_to_remove = triggered_id['row']
    
    # Build a new list of children, excluding the one that matches
    new_children = []
    for child in children:
        # Check if this child's ID matches the one to remove
        # In Dash, children are dicts (serialized components)
        child_id = child.get('props', {}).get('id', {})
        if child_id.get('row') == row_to_remove:
            continue
        new_children.append(child)
        
    return new_children
 
@app.callback(
    Output({"type": "custom-params-container", "index": MATCH}, "children", allow_duplicate=True),
    Input({"type": "btn-apply-audit-rules", "index": MATCH}, "n_clicks"),
    State({"type": "generated-rules-store", "index": MATCH}, "data"),
    State({"type": "custom-params-container", "index": MATCH}, "children"),
    State({"type": "btn-apply-audit-rules", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def apply_audit_rules_pm(n_clicks, rules, current_children, btn_id):
    if not n_clicks or not rules: return dash.no_update
    dataset_id = btn_id["index"]
    
    new_children = list(current_children) if current_children else []
    
    for i, rule in enumerate(rules):
        rule_name = rule.get("name", f"Rule {i+1}")
        rule_type = rule.get("type", "text")
        rule_val = rule.get("value", "")

        if isinstance(rule_val, list):
            rule_val = ", ".join(map(str, rule_val))
            
        # Unique ID for the new row to avoid conflicts
        row_id = f"audit-{n_clicks}-{i}"
        # Robustly parse rule_val for number ranges
        val_str = str(rule_val) if rule_val is not None else ""
        min_val = None
        max_val = None
        if rule_type == 'number':
            # Try various separators: ',', '-', ' to ', ':', '/'
            import re
            parts = re.split(r',|-| to |:|/', val_str)
            if len(parts) >= 2:
                try:
                    min_val = parts[0].strip()
                    max_val = parts[1].strip()
                except: pass
            elif len(parts) == 1 and parts[0].strip():
                # Check if it is actually a number
                candidate = parts[0].strip().replace('%', '')
                try:
                    float(candidate)
                    min_val = candidate
                except ValueError:
                    # Not a number! Fallback to text
                    rule_type = 'text'
        
        new_row = dbc.Row([
            dbc.Col(
                dbc.Input(id={'type': 'cp-name', 'index': dataset_id, 'row': row_id}, value=rule_name, size="sm"),
                width=3),
            dbc.Col(dbc.Select(
                id={'type': 'cp-type', 'index': dataset_id, 'row': row_id},
                options=[{'label': 'Text Instruction', 'value': 'text'}, {'label': 'Specific Options', 'value': 'options'},
                         {'label': 'Number Range', 'value': 'number'}],
                value=rule_type, size="sm"
            ), width=3),
            dbc.Col([
                dbc.Input(id={'type': 'cp-val-text', 'index': dataset_id, 'row': row_id}, value=rule_val if rule_type != 'number' else "",
                          size="sm", style={'display': 'block' if rule_type != 'number' else 'none'},
                          placeholder="e.g. Under 50 words..."),
                dbc.InputGroup([
                    dbc.InputGroupText("Min"),
                    dbc.Input(id={'type': 'cp-val-min', 'index': dataset_id, 'row': row_id}, type="number", value=min_val),
                    dbc.InputGroupText("Max"),
                    dbc.Input(id={'type': 'cp-val-max', 'index': dataset_id, 'row': row_id}, type="number", value=max_val)
                ], id={'type': 'cp-val-num-group', 'index': dataset_id, 'row': row_id}, size="sm", style={'display': 'flex' if rule_type == 'number' else 'none'})
            ], width=5),
            dbc.Col(
                dbc.Button(html.I(className="bi bi-trash"), id={'type': 'btn-remove-row', 'index': dataset_id, 'row': row_id}, color="danger", outline=True, size="sm", className="border-0 opacity-50 hover-opacity-100"),
                width=1, className="text-end"
            )
        ], className="mb-2 align-items-center bg-success bg-opacity-10 rounded p-1 g-2", id={'type': 'custom-param-row-wrapper', 'index': dataset_id, 'row': row_id})
        
        new_children.append(new_row)
        
    return new_children


@app.callback(
    Output({'type': 'cp-val-text', 'index': ALL, 'row': MATCH}, 'style', allow_duplicate=True),
    Output({'type': 'cp-val-text', 'index': ALL, 'row': MATCH}, 'placeholder', allow_duplicate=True),
    Output({'type': 'cp-val-num-group', 'index': ALL, 'row': MATCH}, 'style', allow_duplicate=True),
    Input({'type': 'cp-type', 'index': ALL, 'row': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_param_inputs_pm(p_type_list):
    if not p_type_list: return dash.no_update
    p_type = p_type_list[0]
    
    if p_type == 'number':
        return [{'display': 'none'}] * len(p_type_list), [""] * len(p_type_list), [{'display': 'flex'}] * len(p_type_list)
    elif p_type == 'options':
        return [{'display': 'block'}] * len(p_type_list), ["e.g., Active, Pending, Closed"] * len(p_type_list), [{'display': 'none'}] * len(p_type_list)
    else:
        return [{'display': 'block'}] * len(p_type_list), ["e.g., Must be under 50 words"] * len(p_type_list), [{'display': 'none'}] * len(p_type_list)


# --- DATASET DESCRIPTION / INSIGHTS LOGIC ---
@app.callback(
    Output({"type": "metadata-output", "index": MATCH}, "children"),
    Input({"type": "btn-analyze-metadata", "index": MATCH}, "n_clicks"),
    State({"type": "data-grid", "index": MATCH}, "rowData"),
    State({"type": "algo-selector", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def analyze_metadata_pm(n_clicks, row_data, algorithm):
    if not n_clicks or not row_data:
        return ""
    
    import pandas as pd
    df_sample = pd.DataFrame(row_data).head(20)
    for col in ['_row_id', 'is_context', 'is_target']:
        if col in df_sample.columns: df_sample.drop(columns=[col], inplace=True)
        
    description = get_dataset_description(df_sample, algorithm=algorithm)
    
    # Post-processing to strip conversational fillers by finding the first Markdown header
    if "#" in description:
        description = description[description.find("#"):].strip()
    else:
        # Fallback to simple stripping if no header found
        fillers = ["Okay", "Here is", "Certainly", "Sure", "I analyzed", "This dataset"]
        for f in fillers:
            if description.lstrip().startswith(f) and ":" in description[:100]:
                parts = description.split(":", 1)
                if len(parts) > 1:
                    description = parts[1].lstrip()
                break
            
    return description.strip()

@app.callback(
    Output({"type": "dataset-description-store", "index": MATCH}, "data"),
    Input({"type": "btn-gen-desc", "index": MATCH}, "n_clicks"),
    State({"type": "original-data-store", "index": MATCH}, "data"),
    prevent_initial_call=True
)
def generate_dataset_insights_pm(n_clicks, orig_data):
    if not n_clicks or not orig_data:
        return dash.no_update
    
    df = pd.DataFrame(orig_data)
    if '_row_id' in df.columns:
        df = df.drop(columns=['_row_id', 'is_context', 'is_target', 'ai_audit'], errors='ignore')
    
    df_sample = df.head(10)
    description = get_dataset_description(df_sample)
    
    # Post-processing to strip conversational fillers by finding the first Markdown header
    if "#" in description:
        description = description[description.find("#"):].strip()
    else:
        # Fallback to simple stripping if no header found
        fillers = ["Okay", "Here is", "Certainly", "Sure", "I analyzed", "This dataset"]
        for f in fillers:
            if description.lstrip().startswith(f) and ":" in description[:100]:
                parts = description.split(":", 1)
                if len(parts) > 1:
                    description = parts[1].lstrip()
                break
            
    return description.strip()

@app.callback(
    Output({"type": "dataset-description-edit", "index": MATCH}, "value"),
    Output({"type": "dataset-description-markdown", "index": MATCH}, "children"),
    Input({"type": "dataset-description-store", "index": MATCH}, "data"),
    prevent_initial_call=True
)
def update_description_ui_pm(description):
    # Break infinite loop strictly.
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
        
    triggered_id = ctx.triggered[0]['prop_id']
    
    desc = description or ""
    
    # If the user was the one who changed the store, DO NOT push the value back to the box.
    # This is the primary cause of cursor jumping / focus loss.
    if '"type":"dataset-description-edit"' in triggered_id:
        return dash.no_update, desc
        
    return desc, desc

@app.callback(
    Output({"type": "dataset-description-markdown-container", "index": MATCH}, "style"),
    Output({"type": "dataset-description-edit-container", "index": MATCH}, "style"),
    Output({"type": "btn-toggle-desc-view", "index": MATCH}, "color"),
    Output({"type": "btn-toggle-desc-edit", "index": MATCH}, "color"),
    Input({"type": "btn-toggle-desc-view", "index": MATCH}, "n_clicks"),
    Input({"type": "btn-toggle-desc-edit", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def toggle_description_mode_pm(n_view, n_edit):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id']
    if 'btn-toggle-desc-edit' in triggered_id:
        return {"display": "none"}, {"display": "block"}, "secondary", "info"
    else:
        return {"display": "block"}, {"display": "none"}, "info", "secondary"

@app.callback(
    Output({"type": "dataset-description-store", "index": MATCH}, "data", allow_duplicate=True),
    Input({"type": "dataset-description-edit", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def save_description_from_ui_pm(value):
    return value
