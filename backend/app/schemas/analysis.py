"""Schemas for analysis / walk-forward endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    log_returns: bool = True
    simple_returns: bool = True
    volatility_windows: list[int] = Field(default=[5, 10, 20])
    rolling_mean_windows: list[int] = Field(default=[5, 20])
    drawdown: bool = True
    rolling_max_drawdown: bool = True
    atr_range: bool = True
    volume_change: bool = True
    z_scored_return: bool = True
    momentum_windows: list[int] = Field(default=[10, 20])
    realized_vol: bool = True
    rsi: bool = True
    macd: bool = True
    rolling_skew: bool = True
    rolling_kurtosis: bool = True


class ModelConfig(BaseModel):
    model_type: str = Field("hmm", description="hmm or markov")
    n_states: int = Field(3, ge=2, le=6)
    covariance_type: str = Field("full", description="full, diag, tied, spherical")
    n_iter: int = Field(200, ge=10)
    tol: float = Field(1e-4)
    random_seed: int = 42
    scaling: bool = True
    feature_subset: list[str] | None = None


class WalkForwardConfig(BaseModel):
    train_window: int = Field(504, ge=100)
    test_window: int = Field(63, ge=10)
    step_size: int = Field(63, ge=1)
    mode: str = Field("expanding", description="expanding or rolling")
    min_observations: int = Field(252, ge=50)
    refit_every: int = Field(1, ge=1)


class AnalysisRequest(BaseModel):
    ticker: str = "SPY"
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    feature_config: FeatureConfig = Field(default_factory=FeatureConfig)
    regime_model: ModelConfig = Field(default_factory=ModelConfig)
    walkforward_config: WalkForwardConfig = Field(default_factory=WalkForwardConfig)


class FoldResult(BaseModel):
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_states: list[int]
    test_states: list[int]
    train_dates: list[str]
    test_dates: list[str]
    regime_labels: dict[int, str]
    transition_matrix: list[list[float]] | None = None
    train_occupancy: dict[int, float]
    test_occupancy: dict[int, float]
    train_persistence: float
    test_persistence: float
    regime_stats: list[dict[str, Any]]
    test_regime_stats: list[dict[str, Any]]
    state_separation: float
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    warnings: list[str] = []


class AnalysisResult(BaseModel):
    run_id: str
    status: str
    ticker: str
    n_folds: int
    n_states: int
    duration_secs: float
    folds: list[FoldResult]
    robustness: dict[str, Any]
    charts: dict[str, Any]
    regime_labels: dict[int, str]
