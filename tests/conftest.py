"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + np.abs(np.random.randn(n) * 0.5),
        "Low": close - np.abs(np.random.randn(n) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_features(sample_ohlcv) -> pd.DataFrame:
    from backend.app.schemas.analysis import FeatureConfig
    from backend.app.services.feature_service import engineer_features
    return engineer_features(sample_ohlcv, FeatureConfig())
