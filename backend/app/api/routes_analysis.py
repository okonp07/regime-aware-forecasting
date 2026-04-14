"""Analysis / walk-forward endpoints."""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.routes_data import get_current_df, get_current_ticker
from backend.app.core.logging import logger
from backend.app.db.models import RunRecord
from backend.app.db.session import get_db
from backend.app.schemas.analysis import AnalysisRequest, AnalysisResult
from backend.app.services.data_service import fetch_data
from backend.app.services.export_service import save_run_artifacts
from backend.app.services.feature_service import engineer_features
from backend.app.services.walkforward_service import run_walkforward
from backend.app.schemas.data import FetchRequest

router = APIRouter(prefix="/analysis", tags=["analysis"])

# In-memory results cache
_results_cache: dict[str, AnalysisResult] = {}


@router.post("/run", response_model=AnalysisResult)
def run_analysis(req: AnalysisRequest, db: Session = Depends(get_db)):
    run_id = uuid.uuid4().hex[:12]

    # Create DB record
    record = RunRecord(
        id=run_id,
        status="running",
        ticker=req.ticker,
        start_date=req.start_date,
        end_date=req.end_date,
        n_states=req.regime_model.n_states,
        covariance_type=req.regime_model.covariance_type,
        train_window=req.walkforward_config.train_window,
        test_window=req.walkforward_config.test_window,
        step_size=req.walkforward_config.step_size,
        window_mode=req.walkforward_config.mode,
        feature_config=req.feature_config.model_dump(),
        model_config_json=req.regime_model.model_dump(),
    )
    db.add(record)
    db.commit()

    t0 = time.time()
    try:
        # Fetch data
        raw_df = fetch_data(FetchRequest(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
        ))

        # Feature engineering
        featured_df = engineer_features(raw_df, req.feature_config)

        # Walk-forward
        result = run_walkforward(featured_df, raw_df, req, run_id)

        # Save artifacts
        output_dir = save_run_artifacts(result)

        # Update DB
        record.status = "completed"
        record.n_folds = result.n_folds
        record.duration_secs = round(time.time() - t0, 2)
        record.output_dir = str(output_dir)
        record.summary_json = result.robustness
        db.commit()

        _results_cache[run_id] = result
        return result

    except Exception as e:
        logger.error("Analysis run %s failed: %s", run_id, e)
        record.status = "failed"
        record.error_message = str(e)
        record.duration_secs = round(time.time() - t0, 2)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/run/{run_id}", response_model=AnalysisResult)
def get_run(run_id: str):
    if run_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Run not found in cache. Re-run analysis.")
    return _results_cache[run_id]


@router.get("/run/{run_id}/folds")
def get_folds(run_id: str):
    if run_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"folds": [f.model_dump() for f in _results_cache[run_id].folds]}


@router.get("/run/{run_id}/summary")
def get_summary(run_id: str):
    if run_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Run not found")
    r = _results_cache[run_id]
    return {
        "run_id": r.run_id,
        "ticker": r.ticker,
        "n_folds": r.n_folds,
        "n_states": r.n_states,
        "duration_secs": r.duration_secs,
        "robustness": r.robustness,
        "regime_labels": r.regime_labels,
    }


@router.get("/run/{run_id}/charts")
def get_charts(run_id: str):
    if run_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Run not found")
    return _results_cache[run_id].charts
