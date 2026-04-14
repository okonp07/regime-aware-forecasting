"""Tests for feature engineering."""

import numpy as np

from backend.app.services.feature_service import engineer_features, get_feature_columns
from backend.app.schemas.analysis import FeatureConfig


def test_engineer_features_produces_expected_columns(sample_ohlcv):
    df = engineer_features(sample_ohlcv, FeatureConfig())
    cols = get_feature_columns(df)
    assert "log_return" in cols
    assert "simple_return" in cols
    assert "volatility_5" in cols
    assert "volatility_20" in cols
    assert "drawdown" in cols
    assert "rsi_14" in cols
    assert "macd_signal" in cols


def test_no_nans_after_engineering(sample_ohlcv):
    df = engineer_features(sample_ohlcv, FeatureConfig())
    cols = get_feature_columns(df)
    assert df[cols].isnull().sum().sum() == 0


def test_feature_count(sample_ohlcv):
    df = engineer_features(sample_ohlcv, FeatureConfig())
    cols = get_feature_columns(df)
    assert len(cols) >= 15  # plenty of features by default


def test_no_lookahead_in_returns(sample_ohlcv):
    df = engineer_features(sample_ohlcv, FeatureConfig())
    # log_return at index i should only depend on close[i] and close[i-1]
    close = sample_ohlcv["Close"]
    expected = np.log(close / close.shift(1)).dropna()
    actual = df["log_return"]
    # They should overlap where both exist
    common = expected.index.intersection(actual.index)
    np.testing.assert_array_almost_equal(
        expected.loc[common].values[:10],
        actual.loc[common].values[:10],
        decimal=10,
    )
