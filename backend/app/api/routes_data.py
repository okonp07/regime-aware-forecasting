"""Data ingestion endpoints."""

from fastapi import APIRouter, HTTPException

from backend.app.schemas.data import DataPreview, FetchRequest
from backend.app.services.data_service import fetch_data, get_preview
from backend.app.utils.exceptions import DataFetchError

router = APIRouter(prefix="/data", tags=["data"])

# In-memory cache for current session
_current_df = {}


@router.post("/fetch", response_model=DataPreview)
def fetch(req: FetchRequest):
    try:
        df = fetch_data(req)
        _current_df["df"] = df
        _current_df["ticker"] = req.ticker
        return get_preview(df, req.ticker)
    except DataFetchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/preview", response_model=DataPreview)
def preview():
    if "df" not in _current_df:
        raise HTTPException(status_code=404, detail="No data loaded. Call POST /data/fetch first.")
    return get_preview(_current_df["df"], _current_df.get("ticker", "SPY"))


def get_current_df():
    return _current_df.get("df")


def get_current_ticker():
    return _current_df.get("ticker", "SPY")
