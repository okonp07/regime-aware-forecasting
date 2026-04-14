"""Data ingestion, validation, and caching via yfinance."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.data import DataPreview, FetchRequest
from backend.app.utils.dates import ensure_tz_naive, validate_date_range
from backend.app.utils.exceptions import DataFetchError


def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    fname = f"{ticker}_{start}_{end}_{interval}.parquet"
    return settings.data_dir / fname


def fetch_data(req: FetchRequest) -> pd.DataFrame:
    validate_date_range(req.start_date, req.end_date)
    cache = _cache_path(req.ticker, req.start_date, req.end_date, req.interval)

    if cache.exists():
        logger.info("Loading cached data from %s", cache)
        df = pd.read_parquet(cache)
        return df

    logger.info("Fetching %s from yfinance [%s → %s]", req.ticker, req.start_date, req.end_date)
    df = yf.download(
        req.ticker,
        start=req.start_date,
        end=req.end_date,
        interval=req.interval,
        auto_adjust=req.auto_adjust,
        progress=False,
    )

    if df is None or df.empty:
        raise DataFetchError(f"No data returned for {req.ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize
    df.index = ensure_tz_naive(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise DataFetchError(f"Missing columns: {missing_cols}")

    df = df[expected]

    # Handle missing values
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning("Forward-filling %d missing values", n_missing)
        df = df.ffill(limit=5)
        df = df.dropna()

    # Cache
    df.to_parquet(cache)
    logger.info("Cached %d rows to %s", len(df), cache)

    return df


def get_preview(df: pd.DataFrame, ticker: str) -> DataPreview:
    missing_pct = {col: round(float(df[col].isnull().mean()), 4) for col in df.columns}
    return DataPreview(
        ticker=ticker,
        start_date=str(df.index.min().date()),
        end_date=str(df.index.max().date()),
        n_rows=len(df),
        columns=list(df.columns),
        head=df.head(5).reset_index().to_dict(orient="records"),
        tail=df.tail(5).reset_index().to_dict(orient="records"),
        missing_pct=missing_pct,
    )
