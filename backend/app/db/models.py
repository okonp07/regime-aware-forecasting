"""SQLAlchemy ORM models for run metadata."""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from backend.app.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class RunRecord(Base):
    __tablename__ = "runs"

    id = Column(String(12), primary_key=True, default=_uuid)
    created_at = Column(DateTime, server_default=func.now())
    status = Column(String(20), default="pending")  # pending | running | completed | failed
    ticker = Column(String(10), nullable=False)
    start_date = Column(String(10))
    end_date = Column(String(10))
    n_states = Column(Integer)
    covariance_type = Column(String(20), default="full")
    train_window = Column(Integer)
    test_window = Column(Integer)
    step_size = Column(Integer)
    window_mode = Column(String(20), default="expanding")
    feature_config = Column(JSON)
    model_config_json = Column(JSON)
    n_folds = Column(Integer)
    duration_secs = Column(Float)
    error_message = Column(Text)
    summary_json = Column(JSON)
    output_dir = Column(String(500))
