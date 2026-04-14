"""Streamlit main entry — Home / Overview page."""

import streamlit as st
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="Regime-Aware Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("Regime-Aware Forecasting")
st.markdown("### SPY Market Regime Detection & Walk-Forward Validation")

st.markdown("""
---

**What this app does:**

1. **Data Ingestion** — Fetch SPY historical OHLCV data from Yahoo Finance
2. **Feature Engineering** — Compute regime-relevant features (volatility, momentum, drawdown, etc.)
3. **Regime Detection** — Identify latent market regimes using Gaussian HMM
4. **Walk-Forward Validation** — Evaluate model robustness with chronological train/test splits
5. **Results & Export** — Inspect regime assignments, compare folds, and download reports

---

**Getting started:**

Use the sidebar to navigate through the workflow pages in order:

| Step | Page | Description |
|------|------|-------------|
| 1 | Data Ingestion | Fetch and preview market data |
| 2 | Feature Configuration | Select features for regime detection |
| 3 | Model Configuration | Set HMM parameters |
| 4 | Walk-Forward Setup | Configure validation engine |
| 5 | Run Analysis | Execute the full pipeline |
| 6 | Results Dashboard | Explore charts, metrics, and regime assignments |
| 7 | Run History | View past analysis runs |
| 8 | Export & Report | Download artifacts and reports |

---
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Default Ticker:** SPY")
with col2:
    st.info("**Default States:** 3 (Bull, Transition, Stress)")
with col3:
    st.info("**Validation:** Walk-Forward (Expanding)")
