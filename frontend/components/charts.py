"""Plotly chart builders for the Streamlit frontend."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGIME_COLORS = [
    "#2ecc71",  # green
    "#e74c3c",  # red
    "#3498db",  # blue
    "#f39c12",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
]


def price_regime_chart(data: dict[str, Any]) -> go.Figure:
    dates = data.get("dates", [])
    close = data.get("close", [])
    states = data.get("states", [])
    labels = data.get("labels", {})

    fig = go.Figure()

    if close:
        fig.add_trace(go.Scatter(
            x=dates, y=close, mode="lines", name="Close",
            line=dict(color="white", width=1),
        ))

        # Regime background bands
        unique_states = sorted(set(states))
        for s in unique_states:
            label = labels.get(str(s), labels.get(s, f"State {s}"))
            color = REGIME_COLORS[s % len(REGIME_COLORS)]
            mask_dates, mask_close = [], []
            for i, st in enumerate(states):
                if st == s and i < len(close):
                    mask_dates.append(dates[i])
                    mask_close.append(close[i])
                else:
                    if mask_dates:
                        fig.add_trace(go.Scatter(
                            x=mask_dates, y=mask_close, mode="markers",
                            marker=dict(color=color, size=3),
                            name=label, showlegend=(i == next(j for j, x in enumerate(states) if x == s)),
                            legendgroup=str(s),
                        ))
                        mask_dates, mask_close = [], []
            if mask_dates:
                fig.add_trace(go.Scatter(
                    x=mask_dates, y=mask_close, mode="markers",
                    marker=dict(color=color, size=3),
                    name=label, showlegend=False, legendgroup=str(s),
                ))

    fig.update_layout(
        title="Price with Regime Overlay",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_dark", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def transition_matrix_heatmap(data: dict[str, Any]) -> go.Figure:
    matrix = data.get("matrix", [])
    labels = data.get("labels", {})
    n = len(matrix)
    tick_labels = [labels.get(str(i), labels.get(i, f"State {i}")) for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=tick_labels, y=tick_labels,
        colorscale="Blues", text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}", textfont=dict(size=14),
        hovertemplate="From: %{y}<br>To: %{x}<br>P: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Transition Probability Matrix",
        xaxis_title="To State", yaxis_title="From State",
        template="plotly_dark", height=400,
    )
    return fig


def occupancy_bar_chart(data: dict[str, Any]) -> go.Figure:
    occ = data.get("occupancy", {})
    fig = go.Figure(data=go.Bar(
        x=list(occ.keys()),
        y=list(occ.values()),
        marker_color=[REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(len(occ))],
    ))
    fig.update_layout(
        title="State Occupancy",
        xaxis_title="Regime", yaxis_title="Proportion",
        template="plotly_dark", height=350,
        yaxis=dict(range=[0, 1]),
    )
    return fig


def return_distribution_chart(data: dict[str, Any]) -> go.Figure:
    distributions = data.get("distributions", {})
    fig = go.Figure()
    for i, (label, returns) in enumerate(distributions.items()):
        fig.add_trace(go.Histogram(
            x=returns, name=label, opacity=0.7,
            marker_color=REGIME_COLORS[i % len(REGIME_COLORS)],
            nbinsx=50,
        ))
    fig.update_layout(
        title="Return Distributions by Regime",
        xaxis_title="Log Return", yaxis_title="Count",
        barmode="overlay", template="plotly_dark", height=400,
    )
    return fig


def fold_timeline_chart(data: dict[str, Any]) -> go.Figure:
    folds = data.get("folds", [])
    fig = go.Figure()
    for f in folds:
        fid = f["fold"]
        fig.add_trace(go.Scatter(
            x=[f["train_start"], f["train_end"]],
            y=[fid, fid],
            mode="lines", line=dict(color="#3498db", width=8),
            name="Train" if fid == 0 else None,
            showlegend=(fid == 0), legendgroup="train",
        ))
        fig.add_trace(go.Scatter(
            x=[f["test_start"], f["test_end"]],
            y=[fid, fid],
            mode="lines", line=dict(color="#e74c3c", width=8),
            name="Test" if fid == 0 else None,
            showlegend=(fid == 0), legendgroup="test",
        ))
    fig.update_layout(
        title="Walk-Forward Fold Timeline",
        xaxis_title="Date", yaxis_title="Fold",
        template="plotly_dark", height=max(300, len(folds) * 30 + 100),
    )
    return fig


def drawdown_chart(data: dict[str, Any]) -> go.Figure:
    dates = data.get("dates", [])
    dd = data.get("drawdown", [])
    fig = go.Figure(data=go.Scatter(
        x=dates, y=dd, fill="tozeroy",
        line=dict(color="#e74c3c", width=1),
    ))
    fig.update_layout(
        title="Cumulative Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown",
        template="plotly_dark", height=350,
    )
    return fig


def candlestick_chart(dates, open_, high, low, close) -> go.Figure:
    fig = go.Figure(data=go.Candlestick(
        x=dates, open=open_, high=high, low=low, close=close,
    ))
    fig.update_layout(
        title="SPY Candlestick Chart",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_dark", height=500,
        xaxis_rangeslider_visible=False,
    )
    return fig
