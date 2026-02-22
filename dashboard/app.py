import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.quant_engine import *

import streamlit as st
import plotly.graph_objs as go
import numpy as np

# =========================
# UI
# =========================

st.set_page_config(layout="wide")
st.title("🏦 FX & IRD Desk")

st.sidebar.header("Market Params")

n_scenarios = st.sidebar.slider("MC Scenarios",500,5000,2000,step=100)

initial_rate = st.sidebar.slider("Initial Rate",0.0,0.1,0.02,0.005)
vol_rate = st.sidebar.slider("Rate Vol",0.001,0.05,0.01,0.001)

spot = st.sidebar.number_input("FX Spot",1.12)
strike = st.sidebar.number_input("FX Strike",1.10)
vol_fx = st.sidebar.slider("FX Vol",0.05,0.5,0.12,0.01)

fixed_rate = st.sidebar.slider("IRS Fixed Rate",0.0,0.1,0.025,0.002)
notional = st.sidebar.number_input("Notional",100_000_000)

corr = st.sidebar.slider("Rate FX Corr",-1.,1.,0.)

run = st.sidebar.button("Run Simulation")


# =========================
# SIMULATION
# =========================

if run:

    dt = 1/50
    n_steps = 50

    cov = [[1,corr],[corr,1]]
    chol = np.linalg.cholesky(cov)

    z = np.random.normal(size=(n_scenarios,n_steps,2))
    corr_z = z @ chol.T

    # =====================
    # Hull White Rates
    # =====================

    hw = HullWhite1F(a=0.1,sigma=vol_rate)

    curve = np.full(n_steps,initial_rate)
    theta = hw.calibrate_theta(curve,dt)

    rates = hw.simulate(
        initial_rate,
        theta,
        dt,
        n_steps,
        n_scenarios
    )

    # =====================
    # FX Simulation
    # =====================

    fx_model = FXModel(vol_fx)

    fx_paths = fx_model.simulate(
        spot,
        rates,
        np.zeros_like(rates),
        corr_z[:,:,0],
        dt
    )

    # =====================
    # Pricing Today (Reference Prices)
    # =====================

    fx_today_price = fx_option_bs(
        spot,
        strike,
        1,
        initial_rate,
        0,
        vol_fx
    )

    # =====================
    # FX Option PnL Distribution
    # =====================

    fx_pnls = []

    for path in fx_paths:

        price_scenario = fx_option_bs(
            path[-1],
            strike,
            1,
            initial_rate,
            0,
            vol_fx
        )

        fx_pnls.append(price_scenario - fx_today_price)

    fx_pnls = np.array(fx_pnls)

    # =====================
    # IRS Pricing PnL
    # =====================

    irs_today = np.mean(
        irs_price_mc(
            np.full((n_scenarios,n_steps),initial_rate),
            notional,
            fixed_rate,
            np.arange(1,6),
            np.exp(-initial_rate*np.arange(1,6))
        )
    )

    irs_pnls = []

    for scenario_rates in rates:

        price_scenario = np.mean(
            irs_price_mc(
                scenario_rates.reshape(1,-1),
                notional,
                fixed_rate,
                np.arange(1,6),
                np.exp(-np.cumsum(scenario_rates*dt))
            )
        )

        irs_pnls.append(price_scenario - irs_today)

    irs_pnls = np.array(irs_pnls)

    # =====================
    # Risk Metrics
    # =====================

    var_irs, es_irs = var_es(irs_pnls)
    var_fx, es_fx = var_es(fx_pnls)

    # =====================
    # Greeks (corrected)
    # =====================

    bump = 0.01

    mid_price = fx_today_price

    up = fx_option_bs(
        spot*(1+bump),
        strike,
        1,
        initial_rate,
        0,
        vol_fx
    )

    down = fx_option_bs(
        spot*(1-bump),
        strike,
        1,
        initial_rate,
        0,
        vol_fx
    )

    delta = (up-down)/(2*spot*bump)
    gamma = (up - 2*mid_price + down) / ((spot*bump)**2)

    vega_up = fx_option_bs(
        spot,
        strike,
        1,
        initial_rate,
        0,
        vol_fx+0.01
    )

    vega = (vega_up-mid_price)/0.01

    # =====================
    # DISPLAY
    # =====================

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("IRS Risk")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=irs_pnls))

        st.plotly_chart(fig,use_container_width=True)

        st.metric("VaR 95%",f"{var_irs:,.4f}")
        st.metric("ES 95%",f"{es_irs:,.4f}")

    with col2:

        st.subheader("FX Risk")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=fx_pnls))

        st.plotly_chart(fig,use_container_width=True)

        st.metric("VaR 95%",f"{var_fx:,.4f}")
        st.metric("ES 95%",f"{es_fx:,.4f}")
        st.metric("Delta",f"{delta:.4f}")
        st.metric("Gamma",f"{gamma:.4f}")
        st.metric("Vega",f"{vega:.4f}")

    # =============================
    # Option Surface Pricing
    # =============================

    st.subheader("Option Price Surface")

    strikes = np.linspace(strike*0.8, strike*1.2, 25)
    vols = np.linspace(vol_fx*0.5, vol_fx*1.5, 25)

    surface = np.zeros((len(vols),len(strikes)))

    for i,v in enumerate(vols):
        for j,k in enumerate(strikes):

            surface[i,j] = fx_option_bs(
                spot,
                k,
                1,
                initial_rate,
                0,
                v
            )

    fig = go.Figure(
        data=[
            go.Surface(
                z=surface,
                x=strikes,
                y=vols
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Vol",
            zaxis_title="Price"
        )
    )

    st.plotly_chart(fig,use_container_width=True)
    # =============================
    # PATH VISUALIZATION (PRO LEVEL)
    # =============================

    st.subheader("Market Simulation Paths")

    # -----------------------------
    # Rates Paths + Confidence Bands
    # -----------------------------

    st.markdown("### Interest Rate Simulation")

    rates_mean = np.mean(rates, axis=0)
    rates_p5 = np.percentile(rates, 5, axis=0)
    rates_p95 = np.percentile(rates, 95, axis=0)

    fig_rates = go.Figure()

    # Individual paths (light)
    for i in range(min(30, n_scenarios)):
        fig_rates.add_trace(
            go.Scatter(
                y=rates[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.2,
                name="Path"
            )
        )

    # Mean path
    fig_rates.add_trace(
        go.Scatter(
            y=rates_mean,
            mode="lines",
            line=dict(width=3),
            name="Mean Path"
        )
    )

    # Confidence bands
    fig_rates.add_trace(
        go.Scatter(
            y=rates_p95,
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig_rates.add_trace(
        go.Scatter(
            y=rates_p5,
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="95% Confidence Band"
        )
    )

    fig_rates.update_layout(
        xaxis_title="Time",
        yaxis_title="Interest Rate"
    )

    st.plotly_chart(fig_rates, use_container_width=True)


    # -----------------------------
    # FX Paths + Confidence Bands
    # -----------------------------

    st.markdown("### FX Simulation")

    fx_mean = np.mean(fx_paths, axis=0)
    fx_p5 = np.percentile(fx_paths, 5, axis=0)
    fx_p95 = np.percentile(fx_paths, 95, axis=0)

    fig_fx = go.Figure()

    for i in range(min(30, n_scenarios)):
        fig_fx.add_trace(
            go.Scatter(
                y=fx_paths[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.2
            )
        )

    fig_fx.add_trace(
        go.Scatter(
            y=fx_mean,
            mode="lines",
            line=dict(width=3),
            name="Mean Path"
        )
    )

    fig_fx.add_trace(
        go.Scatter(
            y=fx_p95,
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig_fx.add_trace(
        go.Scatter(
            y=fx_p5,
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="95% Confidence Band"
        )
    )

    fig_fx.update_layout(
        xaxis_title="Time",
        yaxis_title="FX Rate"
    )

    st.plotly_chart(fig_fx, use_container_width=True)    
