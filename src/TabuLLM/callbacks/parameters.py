import dash
from dash import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
from core import app

# --- DYNAMIC PARAMETER LOGIC ---
@app.callback(
    Output('custom-params-container', 'children'),
    Input('btn-add-param', 'n_clicks'),
    State('custom-params-container', 'children'),
    prevent_initial_call=True
)
def add_custom_param_row(n_clicks, children):
    new_idx = n_clicks
    new_row = dbc.Row([
        dbc.Col(
            dbc.Input(id={'type': 'cp-name', 'index': new_idx}, placeholder="Rule Name (e.g. Price Range)", size="sm"),
            width=3),
        dbc.Col(dbc.Select(
            id={'type': 'cp-type', 'index': new_idx},
            options=[{'label': 'Text Instruction', 'value': 'text'}, {'label': 'Specific Options', 'value': 'options'},
                     {'label': 'Number Range', 'value': 'number'}],
            value='text', size="sm"
        ), width=3),
        dbc.Col([
            dbc.Input(id={'type': 'cp-val-text', 'index': new_idx}, type="text", placeholder="e.g. Under 50 words...",
                      size="sm", style={'display': 'block'}),
            dbc.InputGroup([
                dbc.InputGroupText("Min"),
                dbc.Input(id={'type': 'cp-val-min', 'index': new_idx}, type="number"),
                dbc.InputGroupText("Max"),
                dbc.Input(id={'type': 'cp-val-max', 'index': new_idx}, type="number")
            ], id={'type': 'cp-val-num-group', 'index': new_idx}, size="sm", style={'display': 'none'})
        ], width=6)
    ], className="mb-2 align-items-center")
    children.append(new_row)
    return children


@app.callback(
    Output({'type': 'cp-val-text', 'index': MATCH}, 'style'),
    Output({'type': 'cp-val-text', 'index': MATCH}, 'placeholder'),
    Output({'type': 'cp-val-num-group', 'index': MATCH}, 'style'),
    Input({'type': 'cp-type', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_param_inputs(p_type):
    if p_type == 'number':
        return {'display': 'none'}, "", {'display': 'flex'}
    elif p_type == 'options':
        return {'display': 'block'}, "e.g., Active, Pending, Closed", {'display': 'none'}
    else:
        return {'display': 'block'}, "e.g., Must be under 50 words", {'display': 'none'}
