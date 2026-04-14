"""Build JSON-serializable payloads for Plotly charts."""

from typing import Any

import numpy as np
import pandas as pd


def price_with_regimes(
    dates: list[str], close: list[float], states: list[int], labels: dict[int, str]
) -> dict[str, Any]:
    return {
        "chart_type": "price_regimes",
        "dates": dates,
        "close": close,
        "states": states,
        "labels": labels,
    }


def transition_matrix_payload(matrix: list[list[float]], labels: dict[int, str]) -> dict[str, Any]:
    return {
        "chart_type": "transition_matrix",
        "matrix": matrix,
        "labels": labels,
    }


def occupancy_payload(occupancy: dict[int, float], labels: dict[int, str]) -> dict[str, Any]:
    return {
        "chart_type": "occupancy",
        "occupancy": {labels.get(int(k), f"State {k}"): v for k, v in occupancy.items()},
    }


def regime_return_distributions(
    returns: list[float], states: list[int], labels: dict[int, str]
) -> dict[str, Any]:
    data = {}
    arr_r, arr_s = np.array(returns), np.array(states)
    for s in sorted(set(states)):
        mask = arr_s == s
        label = labels.get(s, f"State {s}")
        data[label] = arr_r[mask].tolist()
    return {"chart_type": "return_distributions", "distributions": data}


def fold_timeline(folds: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for f in folds:
        items.append({
            "fold": f.get("fold_id", 0),
            "train_start": f.get("train_start"),
            "train_end": f.get("train_end"),
            "test_start": f.get("test_start"),
            "test_end": f.get("test_end"),
        })
    return {"chart_type": "fold_timeline", "folds": items}


def drawdown_payload(dates: list[str], returns: list[float]) -> dict[str, Any]:
    r = np.array(returns)
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max).tolist()
    return {"chart_type": "drawdown", "dates": dates, "drawdown": dd}
