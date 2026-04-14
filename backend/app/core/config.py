"""Application configuration via environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8501", "https://*.onrender.com"]

    # Database
    database_url: str = "sqlite:///./data/raf.db"

    # Paths
    data_dir: Path = Path("./data")
    outputs_dir: Path = Path("./outputs")

    # Defaults
    default_ticker: str = "SPY"
    default_start_date: str = "2010-01-01"
    default_end_date: str = "2025-12-31"
    default_n_states: int = 3
    default_train_window: int = 504
    default_test_window: int = 63
    default_step_size: int = 63
    random_seed: int = 42

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
