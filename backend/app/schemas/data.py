"""Schemas for data ingestion endpoints."""

from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    ticker: str = Field("SPY", description="Ticker symbol")
    start_date: str = Field("2010-01-01")
    end_date: str = Field("2025-12-31")
    interval: str = Field("1d")
    auto_adjust: bool = True


class DataPreview(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    n_rows: int
    columns: list[str]
    head: list[dict]
    tail: list[dict]
    missing_pct: dict[str, float]
    file_path: str | None = None
