"""Model fitting, prediction, and regime interpretation."""

from typing import Any

import numpy as np

from backend.app.core.logging import logger
from backend.app.models.base import RegimeModelAdapter
from backend.app.models.hmm_adapter import HMMAdapter
from backend.app.schemas.analysis import ModelConfig
from backend.app.utils.exceptions import ModelFitError


def create_model(config: ModelConfig) -> RegimeModelAdapter:
    if config.model_type == "markov":
        from backend.app.models.statsmodels_adapter import MarkovRegressionAdapter
        return MarkovRegressionAdapter(
            n_states=config.n_states,
            random_seed=config.random_seed,
            n_iter=config.n_iter,
        )
    return HMMAdapter(
        n_states=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        tol=config.tol,
        random_seed=config.random_seed,
    )


def fit_and_predict(
    model: RegimeModelAdapter,
    train_X: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        model.fit(train_X)
    except Exception as e:
        raise ModelFitError(f"Model fitting failed: {e}") from e

    train_states = model.predict(train_X)
    test_states = model.predict(test_X)
    return train_states, test_states


def interpret_regimes(
    returns: np.ndarray,
    states: np.ndarray,
    n_states: int,
) -> dict[int, str]:
    """Map state indices to human-friendly labels based on observed statistics."""
    state_stats = []
    for s in range(n_states):
        mask = states == s
        r = returns[mask]
        if len(r) == 0:
            state_stats.append({"state": s, "mean": 0, "vol": 999})
            continue
        state_stats.append({
            "state": s,
            "mean": float(np.mean(r)),
            "vol": float(np.std(r)),
        })

    sorted_by_vol = sorted(state_stats, key=lambda x: x["vol"])
    labels = {}

    if n_states == 2:
        low_vol, high_vol = sorted_by_vol
        labels[low_vol["state"]] = "Bull / Low Vol" if low_vol["mean"] >= 0 else "Calm / Negative"
        labels[high_vol["state"]] = "Stress / High Vol" if high_vol["mean"] < 0 else "Volatile / Positive"

    elif n_states == 3:
        low, mid, high = sorted_by_vol
        if low["mean"] >= 0:
            labels[low["state"]] = "Bull / Low Vol"
        else:
            labels[low["state"]] = "Calm / Mild Decline"
        labels[mid["state"]] = "Transition / Moderate"
        if high["mean"] < 0:
            labels[high["state"]] = "Stress / High Vol"
        else:
            labels[high["state"]] = "Volatile / Recovery"

    else:
        for i, stat in enumerate(sorted_by_vol):
            pct = i / (n_states - 1) if n_states > 1 else 0
            if pct < 0.25:
                prefix = "Bull" if stat["mean"] >= 0 else "Calm"
                labels[stat["state"]] = f"{prefix} (Low Vol)"
            elif pct < 0.5:
                labels[stat["state"]] = "Moderate"
            elif pct < 0.75:
                labels[stat["state"]] = "Elevated Vol"
            else:
                prefix = "Stress" if stat["mean"] < 0 else "Volatile"
                labels[stat["state"]] = f"{prefix} (High Vol)"

    logger.info("Regime labels: %s", labels)
    return labels
