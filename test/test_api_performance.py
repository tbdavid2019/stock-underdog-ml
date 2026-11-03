from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.performance import get_forecast_store, router
from test.test_forecast_performance import records


def make_client(rows):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    class Store:
        def query_records(self, as_of=None):
            return rows
    app.dependency_overrides[get_forecast_store] = lambda: Store()
    return TestClient(app)


def test_stock_performance_and_invalid_window():
    client = make_client(records())
    response = client.get("/api/v1/performance/stocks", params={"as_of": "2026-10-30T23:00:00Z", "window": 120, "market": "US"})
    assert response.status_code == 200
    assert response.json()["data"][0]["ticker"] == "AAPL"
    assert response.json()["data"][0]["sample_count"] == 40
    assert client.get("/api/v1/performance/stocks?window=7").status_code == 422


def test_models_common_sample_and_per_ticker():
    client = make_client(records() + records(model="moirai", count=35))
    params = {"as_of": "2026-10-30T23:00:00Z", "window": 120, "market": "US"}
    response = client.get("/api/v1/performance/models", params=params)
    assert response.status_code == 200
    assert response.json()["common_sample_count"] == 35
    assert client.get("/api/v1/performance/tickers/AAPL", params=params).json()["data"]


def test_empty_returns_explicit_state_and_no_fake_leader():
    response = make_client([]).get("/api/v1/performance/stocks")
    assert response.status_code == 200
    assert response.json()["data"] == []
    assert response.json()["status"] == "insufficient_data"


def test_unsupported_nonprice_target_is_rejected_instead_of_mis_scored():
    client = make_client([])
    for target in ("return", "volatility"):
        assert client.get("/api/v1/performance/models", params={"target_type": target}).status_code == 422
