"""Run history endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db.models import RunRecord
from backend.app.db.session import get_db
from backend.app.schemas.runs import RunList, RunSummary

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=RunList)
def list_runs(db: Session = Depends(get_db)):
    records = db.query(RunRecord).order_by(RunRecord.created_at.desc()).all()
    runs = [
        RunSummary(
            id=r.id,
            created_at=str(r.created_at),
            status=r.status,
            ticker=r.ticker,
            start_date=r.start_date,
            end_date=r.end_date,
            n_states=r.n_states,
            n_folds=r.n_folds,
            duration_secs=r.duration_secs,
            window_mode=r.window_mode,
            error_message=r.error_message,
        )
        for r in records
    ]
    return RunList(runs=runs, total=len(runs))


@router.delete("/{run_id}")
def delete_run(run_id: str, db: Session = Depends(get_db)):
    record = db.query(RunRecord).filter(RunRecord.id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete(record)
    db.commit()
    return {"deleted": run_id}
