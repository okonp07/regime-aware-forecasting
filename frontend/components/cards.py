"""Reusable metric cards and UI components."""

import streamlit as st


def metric_card(label: str, value, delta=None, help_text=None):
    st.metric(label=label, value=value, delta=delta, help=help_text)


def kpi_row(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            metric_card(m["label"], m["value"], m.get("delta"), m.get("help"))


def status_banner(status: str, message: str):
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)


def regime_table(stats: list[dict], labels: dict):
    import pandas as pd
    rows = []
    for s in stats:
        sid = s.get("state", 0)
        label = labels.get(str(sid), labels.get(sid, f"State {sid}"))
        rows.append({
            "Regime": label,
            "Count": s.get("count", 0),
            "Mean Return": f"{s.get('mean_return', 0):.6f}",
            "Volatility": f"{s.get('std_return', 0):.6f}",
            "Sharpe": f"{s.get('sharpe', 0):.2f}",
            "Max DD": f"{s.get('max_drawdown', 0):.4f}",
            "% Positive": f"{s.get('positive_pct', 0):.1%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
