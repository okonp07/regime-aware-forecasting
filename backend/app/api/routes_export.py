"""Export endpoints for download artifacts."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.export_service import (
    get_csv_export,
    get_json_export,
    get_report_export,
    get_state_assignments,
)

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/{run_id}/csv")
def export_csv(run_id: str):
    path = get_csv_export(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV export not found")
    return FileResponse(path, filename=f"{run_id}_fold_metrics.csv", media_type="text/csv")


@router.get("/{run_id}/json")
def export_json(run_id: str):
    path = get_json_export(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="JSON export not found")
    return FileResponse(path, filename=f"{run_id}_summary.json", media_type="application/json")


@router.get("/{run_id}/report")
def export_report(run_id: str):
    path = get_report_export(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, filename=f"{run_id}_report.md", media_type="text/markdown")


@router.get("/{run_id}/states")
def export_states(run_id: str):
    path = get_state_assignments(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="State assignments not found")
    return FileResponse(path, filename=f"{run_id}_states.csv", media_type="text/csv")
