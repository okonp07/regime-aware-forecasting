"""Page 5 — Run Analysis."""

import time
import streamlit as st
import requests
from components.sidebar import render_sidebar, api_url

render_sidebar()

st.header("Run Analysis")
st.markdown("Execute the full regime detection pipeline with walk-forward validation.")

# Gather configs
ticker = st.session_state.get("ticker", "SPY")
start_date = st.session_state.get("start_date", "2010-01-01")
end_date = st.session_state.get("end_date", "2025-12-31")
fc = st.session_state.get("feature_config", {})
mc = st.session_state.get("model_config", {})
wf = st.session_state.get("wf_config", {})

# Build request payload
feature_config = {
    "log_returns": fc.get("log_returns", True),
    "simple_returns": fc.get("simple_returns", True),
    "volatility_windows": fc.get("volatility_windows", [5, 10, 20]),
    "rolling_mean_windows": fc.get("rolling_mean_windows", [5, 20]),
    "drawdown": fc.get("drawdown", True),
    "rolling_max_drawdown": fc.get("rolling_max_drawdown", True),
    "atr_range": fc.get("atr_range", True),
    "volume_change": fc.get("volume_change", True),
    "z_scored_return": fc.get("z_scored_return", True),
    "momentum_windows": fc.get("momentum_windows", [10, 20]),
    "realized_vol": fc.get("realized_vol", True),
    "rsi": fc.get("rsi", True),
    "macd": fc.get("macd", True),
    "rolling_skew": fc.get("rolling_skew", True),
    "rolling_kurtosis": fc.get("rolling_kurtosis", True),
}

model_config = {
    "model_type": mc.get("model_type", "hmm"),
    "n_states": mc.get("n_states", 3),
    "covariance_type": mc.get("covariance_type", "full"),
    "n_iter": mc.get("n_iter", 200),
    "tol": mc.get("tol", 1e-4),
    "random_seed": mc.get("random_seed", 42),
    "scaling": mc.get("scaling", True),
}

walkforward_config = {
    "train_window": wf.get("train_window", 504),
    "test_window": wf.get("test_window", 63),
    "step_size": wf.get("step_size", 63),
    "mode": wf.get("mode", "expanding"),
    "min_observations": wf.get("min_observations", 252),
    "refit_every": wf.get("refit_every", 1),
}

# Show summary before running
with st.expander("Review Configuration", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data**")
        st.text(f"Ticker: {ticker}")
        st.text(f"Range: {start_date} to {end_date}")
    with col2:
        st.markdown("**Model**")
        st.text(f"Type: {model_config['model_type']}")
        st.text(f"States: {model_config['n_states']}")
        st.text(f"Covariance: {model_config['covariance_type']}")
    with col3:
        st.markdown("**Walk-Forward**")
        st.text(f"Mode: {walkforward_config['mode']}")
        st.text(f"Train: {walkforward_config['train_window']}d")
        st.text(f"Test: {walkforward_config['test_window']}d")
        st.text(f"Step: {walkforward_config['step_size']}d")

st.divider()

if st.button("Run Analysis", type="primary", use_container_width=True):
    payload = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "feature_config": feature_config,
        "regime_model": model_config,
        "walkforward_config": walkforward_config,
    }

    try:
        # Start the analysis (returns immediately)
        resp = requests.post(api_url("/analysis/run"), json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            run_id = data["run_id"]
            st.session_state["pending_run_id"] = run_id
            st.info(f"Analysis started. Run ID: **{run_id}**")
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Failed to start analysis: {detail}")
    except requests.ConnectionError:
        st.error("Cannot connect to backend. Is it running?")
    except requests.Timeout:
        st.error("Request timed out starting analysis.")

# Poll for completion
pending_id = st.session_state.get("pending_run_id")
if pending_id:
    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Running walk-forward analysis...")

    max_wait = 600  # 10 minutes max
    poll_interval = 3
    elapsed = 0

    while elapsed < max_wait:
        try:
            status_resp = requests.get(api_url(f"/analysis/status/{pending_id}"), timeout=10)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                current_status = status_data.get("status", "unknown")

                if current_status == "completed":
                    progress_bar.progress(100, text="Complete!")
                    # Fetch full results
                    result_resp = requests.get(api_url(f"/analysis/run/{pending_id}"), timeout=60)
                    if result_resp.status_code == 200:
                        result = result_resp.json()
                        st.session_state["last_result"] = result
                        st.session_state["last_run_id"] = result["run_id"]
                        del st.session_state["pending_run_id"]
                        st.success(
                            f"Analysis complete! Run ID: {result['run_id']} | "
                            f"{result['n_folds']} folds | {result['duration_secs']}s"
                        )
                        st.balloons()
                        st.info("Go to **Results Dashboard** to explore the output.")
                    else:
                        st.warning("Analysis completed but could not fetch full results. Check Results Dashboard.")
                        del st.session_state["pending_run_id"]
                    break

                elif current_status == "failed":
                    progress_bar.empty()
                    error_msg = status_data.get("error_message", "Unknown error")
                    st.error(f"Analysis failed: {error_msg}")
                    del st.session_state["pending_run_id"]
                    break

                else:
                    # Still running
                    pct = min(int((elapsed / max_wait) * 95), 95)
                    duration = status_data.get("duration_secs")
                    progress_bar.progress(pct, text=f"Running... ({elapsed}s elapsed)")

        except Exception:
            pass  # transient network issue, keep polling

        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        progress_bar.empty()
        st.error("Analysis timed out after 10 minutes. Check Run History for results.")
        if "pending_run_id" in st.session_state:
            del st.session_state["pending_run_id"]
