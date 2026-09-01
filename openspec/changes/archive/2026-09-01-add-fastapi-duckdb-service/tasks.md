## 1. DuckDB Manager Analytical Query Extensions

- [x] 1.1 Add analytical query methods to `data/duckdb_manager.py` (`get_latest_predictions`, `get_top_bullish`, `get_top_bearish`, `get_resonance_candidates`, `get_xuantie_candidates`, `get_ticker_history`, `get_db_stats`); verify with unit tests.

## 2. FastAPI Application and Routers Implementation

- [x] 2.1 Add `fastapi` and `uvicorn[standard]` to `requirements.txt`.
- [x] 2.2 Create `api/schemas.py` defining Pydantic response models.
- [x] 2.3 Implement `api/routes/predictions.py` with filtering, pagination, and sorting for all required dimensions.
- [x] 2.4 Implement `api/routes/macro.py` and `api/routes/stats.py`.
- [x] 2.5 Create `api/main.py` assembling CORS, lifespan, OpenAPI documentation, and routers.

## 3. Testing and Verification

- [x] 3.1 Create `test/test_api.py` using `fastapi.testclient.TestClient` to verify all endpoints against DuckDB.
- [x] 3.2 Update `Dockerfile`, `docker-compose.yml` (add `stock-ml-api` service on port 8000), and `docker/entrypoint.sh` (add `api` mode).
- [x] 3.3 Verify live API responses on remote host `10.9.0.99`.
