"""Page 3 — Model Configuration."""

import streamlit as st
from components.sidebar import render_sidebar

render_sidebar()

st.header("Model Configuration")
st.markdown("Configure the regime detection model parameters.")

if "model_config" not in st.session_state:
    st.session_state["model_config"] = {}

mc = st.session_state["model_config"]

st.subheader("Model Type")
mc["model_type"] = st.selectbox(
    "Model",
    options=["hmm", "markov"],
    index=0 if mc.get("model_type", "hmm") == "hmm" else 1,
    help="HMM (Gaussian Hidden Markov Model) is the primary model. Markov (statsmodels MarkovRegression) is an optional alternative.",
)

st.subheader("Hidden States")
mc["n_states"] = st.slider(
    "Number of Hidden States",
    min_value=2, max_value=6,
    value=mc.get("n_states", 3),
    help="Typically 2-4 states. More states capture finer distinctions but risk overfitting.",
)

st.subheader("HMM Parameters")
col1, col2 = st.columns(2)
with col1:
    mc["covariance_type"] = st.selectbox(
        "Covariance Type",
        options=["full", "diag", "tied", "spherical"],
        index=["full", "diag", "tied", "spherical"].index(mc.get("covariance_type", "full")),
    )
    mc["n_iter"] = st.number_input(
        "Max Iterations",
        min_value=10, max_value=1000,
        value=mc.get("n_iter", 200),
    )
with col2:
    mc["tol"] = st.number_input(
        "Convergence Tolerance",
        min_value=1e-8, max_value=1e-1,
        value=mc.get("tol", 1e-4),
        format="%.1e",
    )
    mc["random_seed"] = st.number_input(
        "Random Seed",
        min_value=0, max_value=99999,
        value=mc.get("random_seed", 42),
    )

mc["scaling"] = st.checkbox(
    "Scale features (StandardScaler, fit on train only)",
    value=mc.get("scaling", True),
)

st.divider()
st.subheader("Summary")
st.json({
    "model_type": mc["model_type"],
    "n_states": mc["n_states"],
    "covariance_type": mc["covariance_type"],
    "n_iter": mc["n_iter"],
    "tol": mc["tol"],
    "random_seed": mc["random_seed"],
    "scaling": mc["scaling"],
})

st.session_state["model_config"] = mc
