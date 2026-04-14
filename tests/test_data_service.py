"""Tests for data service."""

import pandas as pd
import pytest


def test_sample_ohlcv_shape(sample_ohlcv):
    assert len(sample_ohlcv) == 600
    assert list(sample_ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_sample_ohlcv_index_is_datetime(sample_ohlcv):
    assert isinstance(sample_ohlcv.index, pd.DatetimeIndex)


def test_no_missing_values(sample_ohlcv):
    assert sample_ohlcv.isnull().sum().sum() == 0
