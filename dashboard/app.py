import sys
import os

# Fix path pour trouver src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objs as go
import numpy as np

from src.simulation import simulate_rates, simulate_fx
from src.pricing import irs_npv, fx_option_price


# ===============================
# Dashboard Title
# ===============================

st.title("FX & IRD Structuring Dashboard")


# ===============================
# IRS NPV Simulation
# ===============================

st.subheader("IRS NPV Scenarios")

rates = simulate_rates()

fig1 = go.Figure()

# On trace un NPV path par scénario
for r in rates[:100]:

    npv_path = irs_npv(
        100e6,
        0.025,
        [0.02] * 5,
        np.arange(1, 6),
        lambda t, r=r: r[t - 1]
    )

    # Sécurité dimension (très important ⭐)
    npv_path = np.atleast_1d(npv_path)

    fig1.add_trace(
        go.Scatter(
            y=npv_path,
            mode="lines"
        )
    )

fig1.update_layout(title="IRS NPV Scenarios")

st.plotly_chart(fig1, use_container_width=True)


# ===============================
# FX Option Distribution
# ===============================

st.subheader("FX Option PnL Distribution")

fx_paths = simulate_fx()

prices = []

for S in fx_paths[:100]:

    price = fx_option_price(
        S[-1],
        1.12,
        1,
        0.01,
        0.005,
        0.12,
        "call"
    )

    prices.append(price)

fig2 = go.Figure([
    go.Histogram(x=prices)
])

fig2.update_layout(title="FX Option PnL Distribution")

st.plotly_chart(fig2, use_container_width=True)
