"""Walk-forward validation engine for regime detection."""

import time
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.logging import logger
from backend.app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResult,
    FoldResult,
    ModelConfig,
    WalkForwardConfig,
)
from backend.app.services.feature_service import (
    fit_scaler,
    get_feature_columns,
    scale_features,
)
from backend.app.services.model_service import (
    create_model,
    fit_and_predict,
    interpret_regimes,
)
from backend.app.utils.metrics import (
    average_regime_duration,
    per_regime_stats,
    regime_persistence,
    robustness_summary,
    state_distribution_drift,
    state_occupancy,
    state_separation_score,
)
from backend.app.utils.plotting_payloads import (
    drawdown_payload,
    fold_timeline,
    occupancy_payload,
    price_with_regimes,
    regime_return_distributions,
    transition_matrix_payload,
)


def generate_folds(
    n_samples: int, wf: WalkForwardConfig
) -> list[tuple[int, int, int, int]]:
    """Generate (train_start, train_end, test_start, test_end) index tuples."""
    folds = []
    cursor = 0
    while True:
        if wf.mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, cursor)

        train_end = cursor + wf.train_window
        test_start = train_end
        test_end = test_start + wf.test_window

        if test_end > n_samples:
            break
        if (train_end - train_start) < wf.min_observations:
            cursor += wf.step_size
            continue

        folds.append((train_start, train_end, test_start, test_end))
        cursor += wf.step_size

    return folds


def run_walkforward(
    featured_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    request: AnalysisRequest,
    run_id: str,
) -> AnalysisResult:
    """Execute walk-forward validation and return structured results."""
    t0 = time.time()
    wf = request.walkforward_config
    mc = request.regime_model
    feature_cols = get_feature_columns(featured_df)

    if mc.feature_subset:
        feature_cols = [c for c in mc.feature_subset if c in feature_cols]
    if not feature_cols:
        raise ValueError("No features selected for modeling")

    feature_matrix = featured_df[feature_cols].values
    dates = featured_df.index
    close = featured_df["Close"].values if "Close" in featured_df.columns else None
    log_ret = featured_df["log_return"].values if "log_return" in featured_df.columns else np.zeros(len(featured_df))

    folds_idx = generate_folds(len(feature_matrix), wf)
    if not folds_idx:
        raise ValueError("No valid folds generated. Try reducing window sizes.")

    logger.info("Walk-forward: %d folds, mode=%s", len(folds_idx), wf.mode)

    fold_results: list[FoldResult] = []
    all_test_states = []
    all_test_dates = []
    prev_model = None

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds_idx):
        warnings_list: list[str] = []
        try:
            train_X = feature_matrix[tr_s:tr_e]
            test_X = feature_matrix[te_s:te_e]
            train_ret = log_ret[tr_s:tr_e]
            test_ret = log_ret[te_s:te_e]

            # Scale — fit on train only
            if mc.scaling:
                scaler = fit_scaler(train_X)
                train_X_s = scale_features(train_X, scaler)
                test_X_s = scale_features(test_X, scaler)
            else:
                train_X_s, test_X_s = train_X, test_X

            # Fit or reuse model
            refit = (i % request.walkforward_config.refit_every == 0) or prev_model is None
            if refit:
                model = create_model(mc)
                train_states, test_states = fit_and_predict(model, train_X_s, test_X_s)
                prev_model = model
            else:
                model = prev_model
                train_states = model.predict(train_X_s)
                test_states = model.predict(test_X_s)

            # Metrics
            labels = interpret_regimes(train_ret, train_states, mc.n_states)
            tr_occ = state_occupancy(train_states, mc.n_states)
            te_occ = state_occupancy(test_states, mc.n_states)
            tr_persist = regime_persistence(train_states)
            te_persist = regime_persistence(test_states)
            train_stats = per_regime_stats(train_ret, train_states, mc.n_states)
            test_stats = per_regime_stats(test_ret, test_states, mc.n_states)
            sep = state_separation_score(train_X_s, train_states, mc.n_states)

            transmat = model.transition_matrix
            ll, aic, bic = None, None, None
            try:
                ll = model.score(train_X_s)
                aic = model.aic(train_X_s)
                bic = model.bic(train_X_s)
            except Exception:
                warnings_list.append("Could not compute information criteria")

            fold_results.append(FoldResult(
                fold_id=i,
                train_start=str(dates[tr_s].date()),
                train_end=str(dates[tr_e - 1].date()),
                test_start=str(dates[te_s].date()),
                test_end=str(dates[te_e - 1].date()),
                train_states=train_states.tolist(),
                test_states=test_states.tolist(),
                train_dates=[str(d.date()) for d in dates[tr_s:tr_e]],
                test_dates=[str(d.date()) for d in dates[te_s:te_e]],
                regime_labels=labels,
                transition_matrix=transmat.tolist() if transmat is not None else None,
                train_occupancy=tr_occ,
                test_occupancy=te_occ,
                train_persistence=tr_persist,
                test_persistence=te_persist,
                regime_stats=train_stats,
                test_regime_stats=test_stats,
                state_separation=sep,
                log_likelihood=ll,
                aic=aic,
                bic=bic,
                warnings=warnings_list,
            ))

            all_test_states.extend(test_states.tolist())
            all_test_dates.extend([str(d.date()) for d in dates[te_s:te_e]])

        except Exception as e:
            logger.error("Fold %d failed: %s", i, e)
            warnings_list.append(f"Fold failed: {str(e)}")

    # Global results
    duration = round(time.time() - t0, 2)
    robust = robustness_summary(
        [f.model_dump() for f in fold_results], mc.n_states
    )

    # Build chart payloads from last fold for overview
    global_labels = fold_results[-1].regime_labels if fold_results else {}
    charts: dict[str, Any] = {}
    if fold_results:
        last = fold_results[-1]
        charts["price_regimes"] = price_with_regimes(
            all_test_dates,
            close[folds_idx[0][2]:folds_idx[-1][3]].tolist() if close is not None else [],
            all_test_states,
            global_labels,
        )
        if last.transition_matrix:
            charts["transition_matrix"] = transition_matrix_payload(
                last.transition_matrix, global_labels
            )
        charts["occupancy"] = occupancy_payload(last.test_occupancy, global_labels)
        charts["return_distributions"] = regime_return_distributions(
            log_ret[folds_idx[0][2]:folds_idx[-1][3]].tolist(),
            all_test_states,
            global_labels,
        )
        charts["fold_timeline"] = fold_timeline([f.model_dump() for f in fold_results])
        charts["drawdown"] = drawdown_payload(
            all_test_dates,
            log_ret[folds_idx[0][2]:folds_idx[-1][3]].tolist(),
        )

    return AnalysisResult(
        run_id=run_id,
        status="completed",
        ticker=request.ticker,
        n_folds=len(fold_results),
        n_states=mc.n_states,
        duration_secs=duration,
        folds=fold_results,
        robustness=robust,
        charts=charts,
        regime_labels=global_labels,
    )
