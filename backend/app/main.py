"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes_analysis import router as analysis_router
from backend.app.api.routes_data import router as data_router
from backend.app.api.routes_export import router as export_router
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_runs import router as runs_router
from backend.app.core.config import settings
from backend.app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Regime-Aware Forecasting API",
    description="SPY regime detection with walk-forward validation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(data_router)
app.include_router(analysis_router)
app.include_router(runs_router)
app.include_router(export_router)
