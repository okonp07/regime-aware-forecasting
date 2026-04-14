"""Page 8 — Export & Report."""

import streamlit as st
import requests
from components.sidebar import render_sidebar, api_url

render_sidebar()

st.header("Export & Report")
st.markdown("Download analysis artifacts for a completed run.")

run_id = st.session_state.get("last_run_id")

if not run_id:
    # Let user pick from completed runs
    try:
        resp = requests.get(api_url("/runs"), timeout=10)
        if resp.status_code == 200:
            runs = resp.json().get("runs", [])
            completed = [r for r in runs if r["status"] == "completed"]
            if completed:
                run_options = {f"{r['id']} — {r['ticker']} ({r['n_folds']} folds)": r['id'] for r in completed}
                selected = st.selectbox("Select a completed run", list(run_options.keys()))
                run_id = run_options[selected]
            else:
                st.warning("No completed runs. Run an analysis first.")
                st.stop()
        else:
            st.error("Could not fetch runs.")
            st.stop()
    except requests.ConnectionError:
        st.error("Cannot connect to backend.")
        st.stop()

st.info(f"Exporting artifacts for run: **{run_id}**")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Summary JSON")
    try:
        resp = requests.get(api_url(f"/export/{run_id}/json"), timeout=30)
        if resp.status_code == 200:
            st.download_button(
                "Download JSON",
                data=resp.content,
                file_name=f"{run_id}_summary.json",
                mime="application/json",
            )
        else:
            st.caption("Not available")
    except Exception:
        st.caption("Error loading")

with col2:
    st.subheader("Fold Metrics")
    try:
        resp = requests.get(api_url(f"/export/{run_id}/csv"), timeout=30)
        if resp.status_code == 200:
            st.download_button(
                "Download CSV",
                data=resp.content,
                file_name=f"{run_id}_fold_metrics.csv",
                mime="text/csv",
            )
        else:
            st.caption("Not available")
    except Exception:
        st.caption("Error loading")

with col3:
    st.subheader("State Assignments")
    try:
        resp = requests.get(api_url(f"/export/{run_id}/states"), timeout=30)
        if resp.status_code == 200:
            st.download_button(
                "Download States",
                data=resp.content,
                file_name=f"{run_id}_states.csv",
                mime="text/csv",
            )
        else:
            st.caption("Not available")
    except Exception:
        st.caption("Error loading")

with col4:
    st.subheader("Report")
    try:
        resp = requests.get(api_url(f"/export/{run_id}/report"), timeout=30)
        if resp.status_code == 200:
            st.download_button(
                "Download Report",
                data=resp.content,
                file_name=f"{run_id}_report.md",
                mime="text/markdown",
            )
        else:
            st.caption("Not available")
    except Exception:
        st.caption("Error loading")

st.divider()

# Preview report inline
st.subheader("Report Preview")
try:
    resp = requests.get(api_url(f"/export/{run_id}/report"), timeout=10)
    if resp.status_code == 200:
        st.markdown(resp.text)
    else:
        st.info("Report not yet generated.")
except Exception:
    st.info("Could not load report preview.")
