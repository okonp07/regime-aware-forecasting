"""Page 7 — Run History."""

import streamlit as st
import requests
import pandas as pd
from components.sidebar import render_sidebar, api_url

render_sidebar()

st.header("Run History")
st.markdown("View and manage past analysis runs.")

if st.button("Refresh", type="secondary"):
    pass  # triggers rerun

try:
    resp = requests.get(api_url("/runs"), timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        runs = data.get("runs", [])
        if not runs:
            st.info("No runs yet. Go to Run Analysis to start one.")
        else:
            st.metric("Total Runs", data.get("total", len(runs)))

            df = pd.DataFrame(runs)
            display_cols = [
                "id", "status", "ticker", "n_states", "n_folds",
                "window_mode", "duration_secs", "created_at",
            ]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[available],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.TextColumn("Run ID"),
                    "status": st.column_config.TextColumn("Status"),
                    "duration_secs": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
                },
            )

            # Delete functionality
            st.divider()
            st.subheader("Delete a Run")
            run_ids = [r["id"] for r in runs]
            to_delete = st.selectbox("Select run to delete", run_ids)
            if st.button("Delete", type="secondary"):
                del_resp = requests.delete(api_url(f"/runs/{to_delete}"), timeout=10)
                if del_resp.status_code == 200:
                    st.success(f"Deleted run {to_delete}")
                    st.rerun()
                else:
                    st.error(f"Failed to delete: {del_resp.text}")
    else:
        st.error(f"Failed to fetch runs: {resp.text}")
except requests.ConnectionError:
    st.error("Cannot connect to backend.")
