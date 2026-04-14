"""Feature engineering pipeline — no look-ahead leakage."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.app.core.logging import logger
from backend.app.schemas.analysis import FeatureConfig

FEATURE_REGISTRY: dict[str, str] = {
    "log_return": "Log return",
    "simple_return": "Simple return",
    "volatility_5": "Rolling vol (5d)",
    "volatility_10": "Rolling vol (10d)",
    "volatility_20": "Rolling vol (20d)",
    "rolling_mean_5": "Rolling mean return (5d)",
    "rolling_mean_20": "Rolling mean return (20d)",
    "drawdown": "Drawdown from running max",
    "rolling_max_dd_20": "Rolling max drawdown (20d)",
    "atr_range": "Normalized high-low range",
    "volume_change": "Volume pct change",
    "z_return": "Z-scored return (20d)",
    "momentum_10": "Momentum (10d)",
    "momentum_20": "Momentum (20d)",
    "realized_vol_20": "Realized vol proxy (20d)",
    "rsi_14": "RSI (14d)",
    "macd_signal": "MACD histogram",
    "rolling_skew_20": "Rolling skewness (20d)",
    "rolling_kurtosis_20": "Rolling kurtosis (20d)",
}


def engineer_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute regime-relevant features from OHLCV data. All rolling uses past data only."""
    out = df.copy()
    close = out["Close"]

    # Returns
    if config.log_returns:
        out["log_return"] = np.log(close / close.shift(1))
    if config.simple_returns:
        out["simple_return"] = close.pct_change()

    log_ret = np.log(close / close.shift(1))

    # Rolling volatility
    for w in config.volatility_windows:
        out[f"volatility_{w}"] = log_ret.rolling(w, min_periods=w).std()

    # Rolling mean return
    for w in config.rolling_mean_windows:
        out[f"rolling_mean_{w}"] = log_ret.rolling(w, min_periods=w).mean()

    # Drawdown
    if config.drawdown:
        cummax = close.cummax()
        out["drawdown"] = (close - cummax) / cummax

    # Rolling max drawdown
    if config.rolling_max_drawdown:
        dd = (close - close.cummax()) / close.cummax()
        out["rolling_max_dd_20"] = dd.rolling(20, min_periods=20).min()

    # ATR-style range
    if config.atr_range:
        out["atr_range"] = (out["High"] - out["Low"]) / close

    # Volume change
    if config.volume_change:
        out["volume_change"] = out["Volume"].pct_change()

    # Z-scored return
    if config.z_scored_return:
        roll_mean = log_ret.rolling(20, min_periods=20).mean()
        roll_std = log_ret.rolling(20, min_periods=20).std()
        out["z_return"] = (log_ret - roll_mean) / roll_std.replace(0, np.nan)

    # Momentum
    for w in config.momentum_windows:
        out[f"momentum_{w}"] = close / close.shift(w) - 1

    # Realized volatility proxy
    if config.realized_vol:
        out["realized_vol_20"] = np.sqrt((log_ret**2).rolling(20, min_periods=20).sum())

    # RSI
    if config.rsi:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    if config.macd:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        out["macd_signal"] = macd_line - signal_line

    # Rolling skew / kurtosis
    if config.rolling_skew:
        out["rolling_skew_20"] = log_ret.rolling(20, min_periods=20).skew()
    if config.rolling_kurtosis:
        out["rolling_kurtosis_20"] = log_ret.rolling(20, min_periods=20).kurt()

    # Drop warmup NaNs
    out = out.dropna()
    logger.info("Engineered %d features, %d rows remain", _count_features(out), len(out))
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    ohlcv = {"Open", "High", "Low", "Close", "Volume", "Date"}
    return [c for c in df.columns if c not in ohlcv]


def _count_features(df: pd.DataFrame) -> int:
    return len(get_feature_columns(df))


def fit_scaler(train_features: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def scale_features(
    features: np.ndarray, scaler: StandardScaler
) -> np.ndarray:
    return scaler.transform(features)
