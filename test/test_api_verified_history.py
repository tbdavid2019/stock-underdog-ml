from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.predictions import get_duckdb, router


def test_history_exposes_only_verified_actual_for_matching_forecast():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    db = MagicMock()
    db.db_path = "/tmp/fixture.duckdb"
    db.resolve_ticker.return_value = {"ticker": "AAPL", "name": "Apple"}
    db.get_ticker_history.return_value = [{"ticker": "AAPL", "model_name": "TimesFM", "timestamp": "2025-01-01", "current_price": 100}]
    app.dependency_overrides[get_duckdb] = lambda: db
    store = MagicMock()
    store.query_records.return_value = [
        {"forecast": {"forecast_id": "f1", "ticker": "AAPL", "model_name": "timesfm", "model_revision": "3",
            "last_complete_bar": "2026-01-05", "issued_at": "2026-01-05T22:00:00Z", "target_date": "2026-01-06",
            "horizon": 1, "predicted_price": 101, "current_price": 100, "evidence_type": "prospective"},
         "actual": {"actual_price": 102, "target_date": "2026-01-06"}, "status": "settled"},
        {"forecast": {"forecast_id": "f2", "ticker": "AAPL", "model_name": "timestral", "model_revision": "r1",
            "last_complete_bar": "2026-01-05", "issued_at": "2026-01-05T22:00:00Z", "target_date": "2026-01-06",
            "horizon": 1, "predicted_price": 103, "current_price": 100},
         "actual": {"reason": "missing"}, "status": "pending"},
    ]
    with patch("api.routes.predictions.ForecastStore", return_value=store):
        response = TestClient(app).get("/api/v1/predictions/history/AAPL")
    assert response.status_code == 200
    rows = response.json()["data"]
    first = next(r for r in rows if r.get("forecast_id") == "f1")
    second = next(r for r in rows if r.get("forecast_id") == "f2")
    assert (first["verification_status"], first["target_date"], first["actual_price"]) == ("settled", "2026-01-06", 102)
    assert first["forecast_session"] == "2026-01-05"
    assert second["verification_status"] == "pending"
    assert second["actual_price"] is None
    assert any(r["verification_status"] == "legacy_unverified" for r in rows)
