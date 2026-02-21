import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objs as go
import numpy as np

from src.simulation import simulate_rates, simulate_fx
from src.pricing import irs_npv, fx_option_price


# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(layout="wide")
st.title("🏦 Quant Multi-Asset Structuring Platform")


# ==================================
# SIDEBAR PARAMETERS
# ==================================

st.sidebar.header("Market Parameters")

n_scenarios = st.sidebar.slider("Monte Carlo Scenarios", 100, 5000, 1000, step=100)

initial_rate = st.sidebar.slider("Initial Rate", 0.0, 0.1, 0.02, step=0.005)
vol_rate = st.sidebar.slider("Rate Volatility", 0.001, 0.05, 0.01, step=0.001)

spot = st.sidebar.number_input("FX Spot", value=1.12)
strike = st.sidebar.number_input("FX Strike", value=1.10)
vol_fx = st.sidebar.slider("FX Volatility", 0.05, 0.5, 0.12, step=0.01)

fixed_rate = st.sidebar.slider("IRS Fixed Rate", 0.0, 0.1, 0.025, step=0.002)
notional = st.sidebar.number_input("IRS Notional", value=100_000_000)

correlation = st.sidebar.slider("Rate/FX Correlation", -1.0, 1.0, 0.0, step=0.1)

stress_rates = st.sidebar.slider("Stress Rates (bps)", -200, 200, 0)
stress_fx = st.sidebar.slider("Stress FX (%)", -20, 20, 0)


run_simulation = st.sidebar.button("🚀 Run Simulation")


# ==================================
# RUN SIMULATION
# ==================================

if run_simulation:

    # =============================
    # Correlated Monte Carlo
    # =============================

    dt = 1/50
    cov_matrix = [[1, correlation], [correlation, 1]]
    chol = np.linalg.cholesky(cov_matrix)

    rates = np.zeros((n_scenarios, 51))
    fx_paths = np.zeros((n_scenarios, 51))

    rates[:,0] = initial_rate
    fx_paths[:,0] = spot

    for t in range(1, 51):

        z = np.random.normal(size=(n_scenarios, 2))
        correlated_z = z @ chol.T

        rates[:,t] = rates[:,t-1] + vol_rate * rates[:,t-1] * correlated_z[:,0] * np.sqrt(dt)
        fx_paths[:,t] = fx_paths[:,t-1] * np.exp(( -0.5 * vol_fx**2) * dt + vol_fx * correlated_z[:,1] * np.sqrt(dt))

    # Apply stress
    rates += stress_rates / 10000
    fx_paths *= (1 + stress_fx/100)


    maturities = np.arange(1, 6)
    discount_curve = np.mean(rates, axis=0)


    # =============================
    # IRS Monte Carlo
    # =============================

    npv_results = []

    for scenario in rates:

        forward_curve = scenario[:len(maturities)]

        npv = irs_npv(
            notional,
            fixed_rate,
            forward_curve,
            maturities,
            lambda t, dc=discount_curve: dc[min(t-1, len(dc)-1)]
        )

        npv_results.append(np.mean(npv))

    npv_results = np.array(npv_results)

    mean_npv = np.mean(npv_results)
    std_npv = np.std(npv_results)
    var_95 = np.percentile(npv_results, 5)
    es_95 = np.mean(npv_results[npv_results <= var_95])


    # =============================
    # FX Option Monte Carlo
    # =============================

    fx_pnls = []

    for path in fx_paths:
        price = fx_option_price(
            path[-1],
            strike,
            1,
            0.01,
            0.005,
            vol_fx,
            "call"
        )
        fx_pnls.append(price)

    fx_pnls = np.array(fx_pnls)

    mean_fx = np.mean(fx_pnls)
    std_fx = np.std(fx_pnls)
    var_fx = np.percentile(fx_pnls, 5)
    es_fx = np.mean(fx_pnls[fx_pnls <= var_fx])


    # =============================
    # Greeks (finite difference)
    # =============================

    bump = 0.01

    price_up = fx_option_price(spot*(1+bump), strike, 1, 0.01, 0.005, vol_fx, "call")
    price_down = fx_option_price(spot*(1-bump), strike, 1, 0.01, 0.005, vol_fx, "call")

    delta = (price_up - price_down) / (2*spot*bump)
    gamma = (price_up - 2*mean_fx + price_down) / ((spot*bump)**2)

    vega_up = fx_option_price(spot, strike, 1, 0.01, 0.005, vol_fx+0.01, "call")
    vega = (vega_up - mean_fx) / 0.01


    # =============================
    # DISPLAY
    # =============================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("IRS Risk")

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=npv_results, nbinsx=40))
        st.plotly_chart(fig1, use_container_width=True)

        st.metric("Mean", f"{mean_npv:,.0f}")
        st.metric("Std", f"{std_npv:,.0f}")
        st.metric("VaR 95%", f"{var_95:,.0f}")
        st.metric("ES 95%", f"{es_95:,.0f}")

    with col2:
        st.subheader("FX Option Risk")

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=fx_pnls, nbinsx=40))
        st.plotly_chart(fig2, use_container_width=True)

        st.metric("Mean", f"{mean_fx:,.4f}")
        st.metric("Std", f"{std_fx:,.4f}")
        st.metric("VaR 95%", f"{var_fx:,.4f}")
        st.metric("ES 95%", f"{es_fx:,.4f}")
        st.metric("Delta", f"{delta:.4f}")
        st.metric("Gamma", f"{gamma:.4f}")
        st.metric("Vega", f"{vega:.4f}")


    # =============================
    # 3D Vol/Strike Surface
    # =============================

    st.subheader("Option Price Surface")

    strikes = np.linspace(strike*0.8, strike*1.2, 20)
    vols = np.linspace(vol_fx*0.5, vol_fx*1.5, 20)

    surface = np.zeros((len(vols), len(strikes)))

    for i, v in enumerate(vols):
        for j, k in enumerate(strikes):
            surface[i,j] = fx_option_price(spot, k, 1, 0.01, 0.005, v, "call")

    fig3 = go.Figure(data=[go.Surface(z=surface, x=strikes, y=vols)])
    st.plotly_chart(fig3, use_container_width=True)
