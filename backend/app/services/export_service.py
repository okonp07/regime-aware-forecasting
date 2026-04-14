"""Export service — generate downloadable artifacts for completed runs."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.analysis import AnalysisResult


def _run_dir(run_id: str) -> Path:
    d = settings.outputs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_artifacts(result: AnalysisResult) -> Path:
    d = _run_dir(result.run_id)

    # summary.json
    summary = {
        "run_id": result.run_id,
        "status": result.status,
        "ticker": result.ticker,
        "n_folds": result.n_folds,
        "n_states": result.n_states,
        "duration_secs": result.duration_secs,
        "regime_labels": result.regime_labels,
        "robustness": result.robustness,
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # fold_metrics.csv
    rows = []
    for f in result.folds:
        row = {
            "fold_id": f.fold_id,
            "train_start": f.train_start,
            "train_end": f.train_end,
            "test_start": f.test_start,
            "test_end": f.test_end,
            "train_persistence": f.train_persistence,
            "test_persistence": f.test_persistence,
            "state_separation": f.state_separation,
            "log_likelihood": f.log_likelihood,
            "aic": f.aic,
            "bic": f.bic,
        }
        for occ_key, occ_val in f.test_occupancy.items():
            row[f"test_occ_state_{occ_key}"] = occ_val
        rows.append(row)
    pd.DataFrame(rows).to_csv(d / "fold_metrics.csv", index=False)

    # state_assignments.csv
    assign_rows = []
    for f in result.folds:
        for dt, st in zip(f.test_dates, f.test_states):
            assign_rows.append({
                "date": dt,
                "fold_id": f.fold_id,
                "state": st,
                "label": f.regime_labels.get(st, f"State {st}"),
            })
    pd.DataFrame(assign_rows).to_csv(d / "state_assignments.csv", index=False)

    # report.md
    report = _generate_report(result)
    (d / "report.md").write_text(report)

    logger.info("Saved artifacts to %s", d)
    return d


def get_csv_export(run_id: str) -> Path:
    return _run_dir(run_id) / "fold_metrics.csv"


def get_json_export(run_id: str) -> Path:
    return _run_dir(run_id) / "summary.json"


def get_report_export(run_id: str) -> Path:
    return _run_dir(run_id) / "report.md"


def get_state_assignments(run_id: str) -> Path:
    return _run_dir(run_id) / "state_assignments.csv"


def _generate_report(result: AnalysisResult) -> str:
    r = result
    robust = r.robustness
    lines = [
        f"# Regime-Aware Forecasting Report",
        f"",
        f"## Dataset",
        f"- **Ticker:** {r.ticker}",
        f"- **States:** {r.n_states}",
        f"- **Folds:** {r.n_folds}",
        f"- **Duration:** {r.duration_secs}s",
        f"",
        f"## Regime Labels",
    ]
    for s, label in sorted(r.regime_labels.items(), key=lambda x: int(x[0])):
        lines.append(f"- State {s}: **{label}**")

    lines.extend([
        f"",
        f"## Robustness",
        f"- Stable regimes: {robust.get('stable_regimes', 'N/A')} / {r.n_states}",
        f"- Avg test persistence: {robust.get('avg_test_persistence', 'N/A')}",
        f"- {robust.get('interpretation', '')}",
        f"",
        f"## Regime Characteristics (Last Fold)",
    ])
    if r.folds:
        last = r.folds[-1]
        for stat in last.regime_stats:
            s = stat["state"]
            label = last.regime_labels.get(s, f"State {s}")
            lines.append(f"### {label}")
            lines.append(f"- Count: {stat.get('count', 0)}")
            lines.append(f"- Mean return: {stat.get('mean_return', 'N/A')}")
            lines.append(f"- Volatility: {stat.get('std_return', 'N/A')}")
            lines.append(f"- Sharpe: {stat.get('sharpe', 'N/A')}")
            lines.append(f"- Max drawdown: {stat.get('max_drawdown', 'N/A')}")
            lines.append("")

    lines.extend([
        f"## Caveats",
        f"- Regime detection is unsupervised — labels are inferred from statistics, not ground truth.",
        f"- HMM assumes Gaussian emissions; real market returns have fat tails.",
        f"- Walk-forward validation reduces overfitting but does not eliminate it.",
        f"- Results are sensitive to the number of states and feature selection.",
        f"",
        f"## Next Steps",
        f"- Compare 2, 3, and 4-state models.",
        f"- Test regime-conditioned trading strategies.",
        f"- Add macro or sentiment features for richer state characterization.",
    ])
    return "\n".join(lines)
