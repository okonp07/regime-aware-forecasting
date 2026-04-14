"""Integration test for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.session import init_db


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_runs_list_empty(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert "total" in data


def test_data_preview_before_fetch(client):
    resp = client.get("/data/preview")
    assert resp.status_code == 404


def test_export_missing_run(client):
    resp = client.get("/export/nonexistent/csv")
    assert resp.status_code == 404
