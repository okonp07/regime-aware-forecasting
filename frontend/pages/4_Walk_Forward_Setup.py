"""Page 4 — Walk-Forward Validation Setup."""

import streamlit as st
from components.sidebar import render_sidebar

render_sidebar()

st.header("Walk-Forward Validation Setup")
st.markdown("Configure the walk-forward engine for chronological model evaluation.")

if "wf_config" not in st.session_state:
    st.session_state["wf_config"] = {}

wf = st.session_state["wf_config"]

col1, col2 = st.columns(2)

with col1:
    wf["mode"] = st.selectbox(
        "Window Mode",
        options=["expanding", "rolling"],
        index=0 if wf.get("mode", "expanding") == "expanding" else 1,
        help="Expanding: train grows each step. Rolling: fixed-size train window moves forward.",
    )
    wf["train_window"] = st.number_input(
        "Train Window (trading days)",
        min_value=100, max_value=5000,
        value=wf.get("train_window", 504),
        help="~504 days = 2 years of trading days",
    )
    wf["test_window"] = st.number_input(
        "Test Window (trading days)",
        min_value=10, max_value=504,
        value=wf.get("test_window", 63),
        help="~63 days = 1 quarter",
    )

with col2:
    wf["step_size"] = st.number_input(
        "Step Size (trading days)",
        min_value=1, max_value=252,
        value=wf.get("step_size", 63),
        help="How far to advance between folds",
    )
    wf["min_observations"] = st.number_input(
        "Minimum Observations",
        min_value=50, max_value=1000,
        value=wf.get("min_observations", 252),
    )
    wf["refit_every"] = st.number_input(
        "Refit Every N Steps",
        min_value=1, max_value=20,
        value=wf.get("refit_every", 1),
        help="1 = refit each fold. Higher values reuse the model across N folds.",
    )

st.divider()

# Estimate folds
data_preview = st.session_state.get("data_preview")
if data_preview:
    n_rows = data_preview["n_rows"]
    n_folds_est = max(0, (n_rows - wf.get("train_window", 504) - wf.get("test_window", 63)) // wf.get("step_size", 63) + 1)
    st.info(f"Estimated **{n_folds_est}** folds with {n_rows} data points")
else:
    st.warning("Fetch data first to see fold estimates.")

st.subheader("Configuration Summary")
st.json(wf)

st.session_state["wf_config"] = wf
