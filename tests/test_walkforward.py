"""Tests for walk-forward fold generation and no-leakage splits."""

import pytest

from backend.app.schemas.analysis import WalkForwardConfig
from backend.app.services.walkforward_service import generate_folds


def test_expanding_folds():
    wf = WalkForwardConfig(
        train_window=100, test_window=20, step_size=20,
        mode="expanding", min_observations=50,
    )
    folds = generate_folds(300, wf)
    assert len(folds) > 0
    # In expanding mode, train always starts at 0
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_s == 0
        assert te_s == tr_e  # no gap
        assert te_e > te_s


def test_rolling_folds():
    wf = WalkForwardConfig(
        train_window=100, test_window=20, step_size=20,
        mode="rolling", min_observations=50,
    )
    folds = generate_folds(300, wf)
    assert len(folds) > 0
    # In rolling mode, train window size is fixed
    for tr_s, tr_e, te_s, te_e in folds:
        assert (tr_e - tr_s) == 100
        assert te_s == tr_e
        assert te_e > te_s


def test_no_overlap_between_train_and_test():
    wf = WalkForwardConfig(
        train_window=100, test_window=20, step_size=20,
        mode="expanding", min_observations=50,
    )
    folds = generate_folds(500, wf)
    for tr_s, tr_e, te_s, te_e in folds:
        assert te_s >= tr_e, "Test data must come after train data (no leakage)"


def test_chronological_ordering():
    wf = WalkForwardConfig(
        train_window=100, test_window=20, step_size=20,
        mode="expanding", min_observations=50,
    )
    folds = generate_folds(500, wf)
    for i in range(1, len(folds)):
        assert folds[i][2] >= folds[i - 1][2], "Folds must be chronologically ordered"


def test_empty_if_data_too_short():
    wf = WalkForwardConfig(
        train_window=500, test_window=100, step_size=50,
        mode="rolling", min_observations=252,
    )
    folds = generate_folds(100, wf)
    assert len(folds) == 0
