"""Page 2 — Feature Configuration."""

import streamlit as st
from components.sidebar import render_sidebar

render_sidebar()

st.header("Feature Configuration")
st.markdown("Select which features to compute for regime detection.")

# Initialize defaults
if "feature_config" not in st.session_state:
    st.session_state["feature_config"] = {}

fc = st.session_state["feature_config"]

st.subheader("Return Features")
col1, col2 = st.columns(2)
with col1:
    fc["log_returns"] = st.checkbox("Log Returns", value=fc.get("log_returns", True))
with col2:
    fc["simple_returns"] = st.checkbox("Simple Returns", value=fc.get("simple_returns", True))

st.subheader("Volatility")
fc["volatility_windows"] = st.multiselect(
    "Rolling Volatility Windows",
    options=[5, 10, 20, 30, 60],
    default=fc.get("volatility_windows", [5, 10, 20]),
)

st.subheader("Rolling Statistics")
fc["rolling_mean_windows"] = st.multiselect(
    "Rolling Mean Return Windows",
    options=[5, 10, 20, 30, 60],
    default=fc.get("rolling_mean_windows", [5, 20]),
)

col1, col2 = st.columns(2)
with col1:
    fc["drawdown"] = st.checkbox("Drawdown", value=fc.get("drawdown", True))
    fc["rolling_max_drawdown"] = st.checkbox("Rolling Max Drawdown (20d)", value=fc.get("rolling_max_drawdown", True))
    fc["atr_range"] = st.checkbox("ATR / Normalized Range", value=fc.get("atr_range", True))
with col2:
    fc["volume_change"] = st.checkbox("Volume Change", value=fc.get("volume_change", True))
    fc["z_scored_return"] = st.checkbox("Z-Scored Return (20d)", value=fc.get("z_scored_return", True))
    fc["realized_vol"] = st.checkbox("Realized Volatility (20d)", value=fc.get("realized_vol", True))

st.subheader("Momentum")
fc["momentum_windows"] = st.multiselect(
    "Momentum Lookback Windows",
    options=[5, 10, 20, 30, 60],
    default=fc.get("momentum_windows", [10, 20]),
)

st.subheader("Technical Indicators")
col1, col2 = st.columns(2)
with col1:
    fc["rsi"] = st.checkbox("RSI (14d)", value=fc.get("rsi", True))
    fc["macd"] = st.checkbox("MACD Signal", value=fc.get("macd", True))
with col2:
    fc["rolling_skew"] = st.checkbox("Rolling Skewness (20d)", value=fc.get("rolling_skew", True))
    fc["rolling_kurtosis"] = st.checkbox("Rolling Kurtosis (20d)", value=fc.get("rolling_kurtosis", True))

st.divider()

# Count enabled features
enabled = sum([
    fc.get("log_returns", False),
    fc.get("simple_returns", False),
    len(fc.get("volatility_windows", [])),
    len(fc.get("rolling_mean_windows", [])),
    fc.get("drawdown", False),
    fc.get("rolling_max_drawdown", False),
    fc.get("atr_range", False),
    fc.get("volume_change", False),
    fc.get("z_scored_return", False),
    len(fc.get("momentum_windows", [])),
    fc.get("realized_vol", False),
    fc.get("rsi", False),
    fc.get("macd", False),
    fc.get("rolling_skew", False),
    fc.get("rolling_kurtosis", False),
])
st.info(f"**{enabled}** features enabled")
st.session_state["feature_config"] = fc
