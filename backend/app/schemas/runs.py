"""Schemas for run history endpoints."""

from typing import Any

from pydantic import BaseModel


class RunSummary(BaseModel):
    id: str
    created_at: str
    status: str
    ticker: str
    start_date: str | None
    end_date: str | None
    n_states: int | None
    n_folds: int | None
    duration_secs: float | None
    window_mode: str | None
    error_message: str | None = None


class RunList(BaseModel):
    runs: list[RunSummary]
    total: int
