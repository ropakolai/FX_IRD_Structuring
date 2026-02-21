import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objs as go
import numpy as np

from src.simulation import simulate_rates, simulate_fx
from src.pricing import irs_npv, fx_option_price


# ===============================
# Config
# ===============================

st.set_page_config(layout="wide")

st.title("🏦 Quant Structuring Desk Dashboard")


# ===============================
# Monte Carlo Simulation
# ===============================

st.header("Monte Carlo Market Simulation")

rates = simulate_rates()
fx_paths = simulate_fx()

st.write("Rate MC shape:", rates.shape)
st.write("FX MC shape:", fx_paths.shape)


# ===============================
# IRD Pricing Engine
# ===============================

st.subheader("IRS NPV Distribution (Monte Carlo)")

notional = 100e6
fixed_rate = 0.025

maturities = np.arange(1, 6)

npv_scenarios = []

# Precompute discount curve = moyenne MC (quant best practice ⭐)
discount_curve = np.mean(rates, axis=0)

for scenario in rates[:500]:

    forward_curve = scenario[:len(maturities)]

    npv = irs_npv(
        notional,
        fixed_rate,
        forward_curve,
        maturities,
        lambda t, dc=discount_curve: dc[min(t-1, len(dc)-1)]
    )

    npv_scenarios.append(np.mean(npv))


npv_scenarios = np.nan_to_num(np.array(npv_scenarios))


fig1 = go.Figure()

fig1.add_trace(
    go.Histogram(
        x=npv_scenarios,
        nbinsx=40
    )
)

fig1.update_layout(
    title="IRS NPV Distribution",
    xaxis_title="NPV",
    yaxis_title="Frequency"
)

st.plotly_chart(fig1, use_container_width=True)


# ===============================
# FX Options Pricing Engine
# ===============================

st.subheader("FX Option PnL Distribution")

spot_price = 1.12
strike = 1.10

fx_pnls = []

for path in fx_paths[:500]:

    pnl = fx_option_price(
        path[-1],
        strike,
        1,
        0.01,
        0.005,
        0.12,
        "call"
    )

    if pnl is not None:
        fx_pnls.append(pnl)

fx_pnls = np.nan_to_num(np.array(fx_pnls))

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=fx_pnls, nbinsx=40))

fig2.update_layout(
    title="FX Option PnL Distribution",
    xaxis_title="PnL",
    yaxis_title="Frequency"
)

st.plotly_chart(fig2, use_container_width=True)


# ===============================
# Sample Path Visualization
# ===============================

st.subheader("Sample Market Paths")

fig3 = go.Figure()

# Plot 20 sample rate paths only (performance optimization ⭐)
for path in rates[:20]:

    fig3.add_trace(
        go.Scatter(
            y=path,
            mode="lines",
            opacity=0.7
        )
    )

fig3.update_layout(
    title="Sample Interest Rate Paths"
)

st.plotly_chart(fig3, use_container_width=True)
