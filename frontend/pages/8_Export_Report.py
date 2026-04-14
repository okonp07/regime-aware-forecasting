"""Page 8 — Export & Report."""

import streamlit as st
import requests
from components.sidebar import render_sidebar, api_url

render_sidebar()

st.header("Export & Report")
st.markdown("Download analysis artifacts for the latest run.")

run_id = st.session_state.get("last_run_id")

if not run_id:
    st.warning("No completed run to export. Run an analysis first.")
    st.stop()

st.info(f"Exporting artifacts for run: **{run_id}**")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Summary JSON")
    if st.button("Download JSON", key="json"):
        try:
            resp = requests.get(api_url(f"/export/{run_id}/json"), timeout=30)
            if resp.status_code == 200:
                st.download_button(
                    "Save summary.json",
                    data=resp.content,
                    file_name=f"{run_id}_summary.json",
                    mime="application/json",
                )
            else:
                st.error("Export not found")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.subheader("Fold Metrics")
    if st.button("Download CSV", key="csv"):
        try:
            resp = requests.get(api_url(f"/export/{run_id}/csv"), timeout=30)
            if resp.status_code == 200:
                st.download_button(
                    "Save fold_metrics.csv",
                    data=resp.content,
                    file_name=f"{run_id}_fold_metrics.csv",
                    mime="text/csv",
                )
            else:
                st.error("Export not found")
        except Exception as e:
            st.error(f"Error: {e}")

with col3:
    st.subheader("State Assignments")
    if st.button("Download States", key="states"):
        try:
            resp = requests.get(api_url(f"/export/{run_id}/states"), timeout=30)
            if resp.status_code == 200:
                st.download_button(
                    "Save states.csv",
                    data=resp.content,
                    file_name=f"{run_id}_states.csv",
                    mime="text/csv",
                )
            else:
                st.error("Export not found")
        except Exception as e:
            st.error(f"Error: {e}")

with col4:
    st.subheader("Report")
    if st.button("Download Report", key="report"):
        try:
            resp = requests.get(api_url(f"/export/{run_id}/report"), timeout=30)
            if resp.status_code == 200:
                st.download_button(
                    "Save report.md",
                    data=resp.content,
                    file_name=f"{run_id}_report.md",
                    mime="text/markdown",
                )
            else:
                st.error("Export not found")
        except Exception as e:
            st.error(f"Error: {e}")

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
