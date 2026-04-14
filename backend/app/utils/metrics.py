"""Regime evaluation metrics — all unsupervised."""

from typing import Any

import numpy as np
import pandas as pd


def state_occupancy(states: np.ndarray, n_states: int) -> dict[int, float]:
    counts = np.bincount(states.astype(int), minlength=n_states)
    total = len(states)
    return {int(s): round(float(counts[s]) / total, 4) for s in range(n_states)}


def average_regime_duration(states: np.ndarray, n_states: int) -> dict[int, float]:
    durations: dict[int, list[int]] = {s: [] for s in range(n_states)}
    current, count = int(states[0]), 1
    for s in states[1:]:
        s = int(s)
        if s == current:
            count += 1
        else:
            durations[current].append(count)
            current, count = s, 1
    durations[current].append(count)
    return {s: round(float(np.mean(d)), 2) if d else 0.0 for s, d in durations.items()}


def regime_persistence(states: np.ndarray) -> float:
    if len(states) < 2:
        return 1.0
    same = np.sum(states[1:] == states[:-1])
    return round(float(same) / (len(states) - 1), 4)


def state_distribution_drift(
    train_states: np.ndarray, test_states: np.ndarray, n_states: int
) -> dict[str, Any]:
    train_occ = state_occupancy(train_states, n_states)
    test_occ = state_occupancy(test_states, n_states)
    drift = {s: round(test_occ.get(s, 0) - train_occ.get(s, 0), 4) for s in range(n_states)}
    return {"train": train_occ, "test": test_occ, "drift": drift}


def per_regime_stats(
    returns: np.ndarray, states: np.ndarray, n_states: int
) -> list[dict[str, Any]]:
    results = []
    for s in range(n_states):
        mask = states == s
        r = returns[mask]
        if len(r) == 0:
            results.append({"state": s, "count": 0})
            continue
        cum = np.cumsum(r)
        running_max = np.maximum.accumulate(cum)
        drawdowns = cum - running_max
        results.append({
            "state": s,
            "count": int(mask.sum()),
            "mean_return": round(float(np.mean(r)), 6),
            "std_return": round(float(np.std(r)), 6),
            "sharpe": round(float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else 0, 4),
            "max_drawdown": round(float(np.min(drawdowns)), 6),
            "skewness": round(float(pd.Series(r).skew()), 4),
            "positive_pct": round(float(np.mean(r > 0)), 4),
        })
    return results


def state_separation_score(
    features: np.ndarray, states: np.ndarray, n_states: int
) -> float:
    """Pseudo-silhouette: ratio of between-cluster to within-cluster variance."""
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    grand_mean = features.mean(axis=0)
    between, within = 0.0, 0.0
    for s in range(n_states):
        mask = states == s
        if mask.sum() < 2:
            continue
        cluster = features[mask]
        centroid = cluster.mean(axis=0)
        between += mask.sum() * np.sum((centroid - grand_mean) ** 2)
        within += np.sum((cluster - centroid) ** 2)
    return round(float(between / within) if within > 0 else 0.0, 4)


def regime_consistency_across_folds(
    all_fold_states: list[dict[str, Any]], n_states: int
) -> dict[str, Any]:
    """Check if regimes maintain consistent characteristics across folds."""
    per_state_means: dict[int, list[float]] = {s: [] for s in range(n_states)}
    per_state_vols: dict[int, list[float]] = {s: [] for s in range(n_states)}
    for fold in all_fold_states:
        for stat in fold.get("regime_stats", []):
            s = stat["state"]
            if stat.get("count", 0) > 0:
                per_state_means[s].append(stat.get("mean_return", 0))
                per_state_vols[s].append(stat.get("std_return", 0))

    consistency = {}
    for s in range(n_states):
        means = per_state_means[s]
        vols = per_state_vols[s]
        consistency[s] = {
            "mean_return_cv": round(float(np.std(means) / np.abs(np.mean(means))) if means and np.mean(means) != 0 else 999, 4),
            "vol_cv": round(float(np.std(vols) / np.mean(vols)) if vols and np.mean(vols) > 0 else 999, 4),
            "n_folds_present": len(means),
        }
    return consistency


def robustness_summary(
    fold_results: list[dict[str, Any]], n_states: int
) -> dict[str, Any]:
    consistency = regime_consistency_across_folds(fold_results, n_states)
    occupancies = [f.get("test_occupancy", {}) for f in fold_results]
    persistences = [f.get("test_persistence", 0) for f in fold_results]

    stable_regimes = sum(
        1 for s in range(n_states)
        if consistency.get(s, {}).get("mean_return_cv", 999) < 2.0
        and consistency.get(s, {}).get("vol_cv", 999) < 1.0
    )

    return {
        "n_folds": len(fold_results),
        "n_states": n_states,
        "stable_regimes": stable_regimes,
        "regime_consistency": consistency,
        "avg_test_persistence": round(float(np.mean(persistences)) if persistences else 0, 4),
        "interpretation": _interpret_robustness(stable_regimes, n_states, persistences),
    }


def _interpret_robustness(stable: int, n_states: int, persistences: list) -> str:
    lines = []
    ratio = stable / n_states if n_states > 0 else 0
    if ratio >= 0.8:
        lines.append("Regimes are stable across folds — model generalizes well.")
    elif ratio >= 0.5:
        lines.append("Some regimes are stable; others show cross-fold variability.")
    else:
        lines.append("Regimes are unstable across folds — consider fewer states or more data.")
    avg_p = float(np.mean(persistences)) if persistences else 0
    if avg_p > 0.9:
        lines.append("High regime persistence suggests meaningful state structure.")
    elif avg_p > 0.7:
        lines.append("Moderate persistence — regimes capture some temporal structure.")
    else:
        lines.append("Low persistence — regime assignments may be noisy.")
    return " ".join(lines)
