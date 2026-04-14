"""Shared sidebar for backend connection status."""

import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def render_sidebar():
    with st.sidebar:
        st.title("RAF")
        st.caption("Regime-Aware Forecasting")
        st.divider()
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=3)
            if r.status_code == 200:
                st.success("Backend connected")
            else:
                st.error("Backend unhealthy")
        except Exception:
            st.error("Backend offline")
        st.divider()
        st.caption(f"API: {BACKEND_URL}")


def api_url(path: str) -> str:
    return f"{BACKEND_URL}{path}"
