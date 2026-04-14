"""Date utility functions."""

from datetime import datetime

import pandas as pd


def parse_date(date_str: str) -> datetime:
    return pd.Timestamp(date_str).to_pydatetime()


def validate_date_range(start: str, end: str) -> tuple[datetime, datetime]:
    s, e = parse_date(start), parse_date(end)
    if s >= e:
        raise ValueError(f"start ({start}) must be before end ({end})")
    return s, e


def ensure_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx
