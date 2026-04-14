"""Page 1 — Data Ingestion."""

import streamlit as st
import requests
import pandas as pd
from components.sidebar import render_sidebar, api_url
from components.charts import candlestick_chart

render_sidebar()

st.header("Data Ingestion")
st.markdown("Fetch SPY (or other ticker) OHLCV data from Yahoo Finance.")

with st.form("fetch_form"):
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "SPY"))
        start_date = st.text_input("Start Date", value=st.session_state.get("start_date", "2010-01-01"))
    with col2:
        end_date = st.text_input("End Date", value=st.session_state.get("end_date", "2025-12-31"))
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    auto_adjust = st.checkbox("Auto-adjust prices", value=True)
    submitted = st.form_submit_button("Fetch Data", type="primary")

if submitted:
    with st.spinner("Fetching data..."):
        try:
            resp = requests.post(api_url("/data/fetch"), json={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "auto_adjust": auto_adjust,
            }, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["ticker"] = ticker
                st.session_state["start_date"] = start_date
                st.session_state["end_date"] = end_date
                st.session_state["data_preview"] = data
                st.success(f"Loaded {data['n_rows']} rows for {ticker}")
            else:
                st.error(f"Error: {resp.json().get('detail', resp.text)}")
        except requests.ConnectionError:
            st.error("Cannot connect to backend. Is it running?")

# Show preview
if "data_preview" in st.session_state:
    data = st.session_state["data_preview"]
    st.subheader("Data Preview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data["n_rows"])
    col2.metric("Start", data["start_date"])
    col3.metric("End", data["end_date"])

    tab1, tab2, tab3 = st.tabs(["Head", "Tail", "Candlestick"])

    with tab1:
        st.dataframe(pd.DataFrame(data["head"]), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(pd.DataFrame(data["tail"]), use_container_width=True, hide_index=True)
    with tab3:
        head_data = pd.DataFrame(data["head"])
        if all(c in head_data.columns for c in ["Date", "Open", "High", "Low", "Close"]):
            st.info("Candlestick chart is shown for the full dataset on the Results page after analysis.")
        else:
            st.info("Candlestick data not available in preview.")

    if data.get("missing_pct"):
        st.subheader("Missing Data (%)")
        st.json(data["missing_pct"])
