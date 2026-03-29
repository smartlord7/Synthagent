import dash
from dash import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from core import app

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
        ], width=6)
    ], className="mb-2 align-items-center")
    
    children = list(children) if children else []
    children.append(new_row)
    return children


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
