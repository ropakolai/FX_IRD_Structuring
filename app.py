import dash
from dash import dcc, html
import plotly.graph_objs as go
from src.simulation import simulate_rates, simulate_fx
from src.pricing import irs_npv, fx_option_price

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("FX & IRD Structuring Dashboard"),
    dcc.Graph(id='irs-npv-graph'),
    dcc.Graph(id='fx-option-graph')
])

# Callback (ici simplifié)
@app.callback(
    dash.dependencies.Output('irs-npv-graph', 'figure'),
    dash.dependencies.Output('fx-option-graph', 'figure'),
    []
)
def update_graph():
    # IRS simulation
    rates = simulate_rates()
    npvs = [irs_npv(100e6, 0.025, [0.02]*5, np.arange(1,6), lambda t: r[t-1]) for r in rates[:100]]
    fig1 = go.Figure()
    for npv in npvs:
        fig1.add_trace(go.Scatter(y=npv, mode='lines'))
    fig1.update_layout(title="IRS NPV scenarios")
    
    # FX Option simulation
    fx_paths = simulate_fx()
    prices = [fx_option_price(S[-1], 1.12, 1, 0.01, 0.005, 0.12, 'call') for S in fx_paths[:100]]
    fig2 = go.Figure([go.Histogram(x=prices)])
    fig2.update_layout(title="FX Option PnL distribution")
    
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True)