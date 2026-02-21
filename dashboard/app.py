import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objs as go
import numpy as np

from src.simulation import simulate_rates, simulate_fx
from src.pricing import irs_npv, fx_option_price

st.title("FX & IRD Structuring Dashboard")

# --- IRS NPV Simulation ---
st.subheader("IRS NPV Scenarios")

rates = simulate_rates()

npvs = [
    irs_npv(
        100e6,
        0.025,
        [0.02] * 5,
        np.arange(1, 6),
        lambda t, r=r: r[t - 1]
    )
    for r in rates[:100]
]

fig1 = go.Figure()
for npv in npvs:
    fig1.add_trace(go.Scatter(y=np.array(npv).flatten(), mode="lines"))

fig1.update_layout(title="IRS NPV Scenarios")

st.plotly_chart(fig1, use_container_width=True)


# --- FX Option Distribution ---
st.subheader("FX Option PnL Distribution")

fx_paths = simulate_fx()

prices = [
    fx_option_price(
        S[-1],
        1.12,
        1,
        0.01,
        0.005,
        0.12,
        "call"
    )
    for S in fx_paths[:100]
]

fig2 = go.Figure([go.Histogram(x=prices)])
fig2.update_layout(title="FX Option PnL Distribution")

st.plotly_chart(fig2, use_container_width=True)
