"""Tests for the HMM model adapter."""

import numpy as np
import pytest

from backend.app.models.hmm_adapter import HMMAdapter


@pytest.fixture
def simple_data():
    np.random.seed(42)
    # Two distinct clusters
    data = np.concatenate([
        np.random.randn(100, 2) * 0.5 + [1, 0],
        np.random.randn(100, 2) * 1.5 + [-1, 0],
    ])
    return data


def test_hmm_fit_predict_shape(simple_data):
    model = HMMAdapter(n_states=2, random_seed=42)
    model.fit(simple_data)
    states = model.predict(simple_data)
    assert states.shape == (200,)
    assert set(np.unique(states)).issubset({0, 1})


def test_hmm_transition_matrix_shape(simple_data):
    model = HMMAdapter(n_states=2, random_seed=42)
    model.fit(simple_data)
    tm = model.transition_matrix
    assert tm.shape == (2, 2)
    np.testing.assert_array_almost_equal(tm.sum(axis=1), [1.0, 1.0], decimal=5)


def test_hmm_score(simple_data):
    model = HMMAdapter(n_states=2, random_seed=42)
    model.fit(simple_data)
    score = model.score(simple_data)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_hmm_aic_bic(simple_data):
    model = HMMAdapter(n_states=2, random_seed=42)
    model.fit(simple_data)
    aic = model.aic(simple_data)
    bic = model.bic(simple_data)
    assert np.isfinite(aic)
    assert np.isfinite(bic)


def test_hmm_three_states(simple_data):
    model = HMMAdapter(n_states=3, random_seed=42)
    model.fit(simple_data)
    states = model.predict(simple_data)
    assert len(np.unique(states)) <= 3
