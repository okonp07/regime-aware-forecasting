"""Page 6 — Results Dashboard."""

import streamlit as st
import requests
import pandas as pd
from components.sidebar import render_sidebar, api_url
from components.charts import (
    price_regime_chart,
    transition_matrix_heatmap,
    occupancy_bar_chart,
    return_distribution_chart,
    fold_timeline_chart,
    drawdown_chart,
)
from components.cards import kpi_row, regime_table

render_sidebar()

st.header("Results Dashboard")

# Try to load results from session state or fetch from backend
result = st.session_state.get("last_result")

if not result:
    st.info("No results in current session. Select a completed run to load results.")

    # Fetch run list from backend
    try:
        resp = requests.get(api_url("/runs"), timeout=10)
        if resp.status_code == 200:
            runs = resp.json().get("runs", [])
            completed = [r for r in runs if r["status"] == "completed"]
            if completed:
                run_options = {f"{r['id']} — {r['ticker']} ({r['n_folds']} folds, {r['created_at']})": r['id'] for r in completed}
                selected = st.selectbox("Select a completed run", list(run_options.keys()))
                run_id = run_options[selected]

                if st.button("Load Results", type="primary"):
                    with st.spinner("Loading results..."):
                        try:
                            res = requests.get(api_url(f"/analysis/run/{run_id}"), timeout=60)
                            if res.status_code == 200:
                                result = res.json()
                                st.session_state["last_result"] = result
                                st.session_state["last_run_id"] = run_id
                                st.rerun()
                            else:
                                st.error(f"Could not load results: {res.json().get('detail', res.text)}")
                        except Exception as e:
                            st.error(f"Failed to load: {e}")
            else:
                st.warning("No completed runs found. Run an analysis first.")
        else:
            st.error("Could not fetch run list from backend.")
    except requests.ConnectionError:
        st.error("Cannot connect to backend.")
    st.stop()

# KPI row
kpi_row([
    {"label": "Run ID", "value": result["run_id"]},
    {"label": "Ticker", "value": result["ticker"]},
    {"label": "States", "value": result["n_states"]},
    {"label": "Folds", "value": result["n_folds"]},
    {"label": "Duration", "value": f"{result['duration_secs']}s"},
])

st.divider()

# Regime labels
st.subheader("Regime Labels")
labels = result.get("regime_labels", {})
cols = st.columns(len(labels))
for col, (sid, label) in zip(cols, sorted(labels.items(), key=lambda x: int(x[0]))):
    col.info(f"**State {sid}:** {label}")

# Charts
charts = result.get("charts", {})

tab_overview, tab_folds, tab_robustness = st.tabs(["Overview", "Per-Fold", "Robustness"])

with tab_overview:
    if "price_regimes" in charts:
        st.plotly_chart(price_regime_chart(charts["price_regimes"]), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "transition_matrix" in charts:
            st.plotly_chart(transition_matrix_heatmap(charts["transition_matrix"]), use_container_width=True)
    with col2:
        if "occupancy" in charts:
            st.plotly_chart(occupancy_bar_chart(charts["occupancy"]), use_container_width=True)

    if "return_distributions" in charts:
        st.plotly_chart(return_distribution_chart(charts["return_distributions"]), use_container_width=True)

    if "drawdown" in charts:
        st.plotly_chart(drawdown_chart(charts["drawdown"]), use_container_width=True)

    if "fold_timeline" in charts:
        st.plotly_chart(fold_timeline_chart(charts["fold_timeline"]), use_container_width=True)

with tab_folds:
    folds = result.get("folds", [])
    if not folds:
        st.info("No fold data available.")
    else:
        fold_ids = [f"Fold {f['fold_id']}" for f in folds]
        selected = st.selectbox("Select Fold", fold_ids)
        idx = fold_ids.index(selected)
        fold = folds[idx]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train", f"{fold['train_start']} → {fold['train_end']}")
        col2.metric("Test", f"{fold['test_start']} → {fold['test_end']}")
        col3.metric("Train Persistence", f"{fold['train_persistence']:.3f}")
        col4.metric("Test Persistence", f"{fold['test_persistence']:.3f}")

        st.subheader("Train Regime Statistics")
        regime_table(fold.get("regime_stats", []), fold.get("regime_labels", {}))

        st.subheader("Test Regime Statistics")
        regime_table(fold.get("test_regime_stats", []), fold.get("regime_labels", {}))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Train Occupancy")
            st.json(fold.get("train_occupancy", {}))
        with col2:
            st.subheader("Test Occupancy")
            st.json(fold.get("test_occupancy", {}))

        if fold.get("transition_matrix"):
            st.subheader("Transition Matrix (This Fold)")
            st.plotly_chart(
                transition_matrix_heatmap({
                    "matrix": fold["transition_matrix"],
                    "labels": fold.get("regime_labels", {}),
                }),
                use_container_width=True,
            )

        info_cols = st.columns(3)
        info_cols[0].metric("Log-Likelihood", f"{fold.get('log_likelihood', 'N/A')}")
        info_cols[1].metric("AIC", f"{fold.get('aic', 'N/A')}")
        info_cols[2].metric("BIC", f"{fold.get('bic', 'N/A')}")

        if fold.get("warnings"):
            st.warning("Warnings: " + "; ".join(fold["warnings"]))

with tab_robustness:
    robust = result.get("robustness", {})
    if not robust:
        st.info("Robustness data not available.")
    else:
        st.subheader("Robustness Summary")
        st.markdown(f"**Interpretation:** {robust.get('interpretation', 'N/A')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Folds", robust.get("n_folds", 0))
        col2.metric("Stable Regimes", f"{robust.get('stable_regimes', 0)} / {robust.get('n_states', 0)}")
        col3.metric("Avg Test Persistence", f"{robust.get('avg_test_persistence', 0):.3f}")

        st.subheader("Regime Consistency Across Folds")
        consistency = robust.get("regime_consistency", {})
        if consistency:
            rows = []
            for sid, stats in consistency.items():
                label = labels.get(str(sid), f"State {sid}")
                rows.append({
                    "Regime": label,
                    "Mean Return CV": stats.get("mean_return_cv", "N/A"),
                    "Volatility CV": stats.get("vol_cv", "N/A"),
                    "Folds Present": stats.get("n_folds_present", 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
