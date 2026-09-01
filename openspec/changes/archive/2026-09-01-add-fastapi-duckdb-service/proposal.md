# Change: Add FastAPI High-Performance REST & MCP-Ready Service on DuckDB

## Why
With over 44,000 historical records and live daily multi-strategy predictions stored in the local columnar DuckDB (`stock_quant.duckdb`), external clients, dashboards, and AI agents (MCP, WebMCP, skills) need a unified, high-performance REST API. 

This API provides query capabilities for:
1. Multi-strategy resonance and top overlap candidates (`🏆三重共振` / `土洋合買`).
2. Technical pullback candidates (`玄鐵重劍` MA60/120).
3. Machine learning prediction rankings (LSTM Top Bullish and Top Bearish).
4. Valuation and fundamental filters (PE, PB, EV/EBITDA, Forward PE).
5. Ticker historical prediction trajectories and backtest accuracy tracking.
6. US Macro Regime status and exposure suggestions.

## What Changes
1. Add `fastapi` and `uvicorn[standard]` dependencies.
2. Implement `api/` module with FastAPI app, Pydantic schemas, and query routers:
   - `api/main.py`: Application entrypoint, CORS, OpenAPI documentation (`/docs`, `/redoc`), and health check.
   - `api/schemas.py`: Pydantic models for request/response serialization.
   - `api/routes/predictions.py`: Endpoints for latest snapshots, resonance, XuanTie, LSTM top N bullish/bearish, ticker history.
   - `api/routes/macro.py`: Macro regime status and exposure endpoint.
   - `api/routes/stats.py`: Database summary and accuracy statistics.
3. Update `data/duckdb_manager.py` with analytical query methods (e.g. `get_latest_predictions`, `get_top_bullish`, `get_top_bearish`, `get_resonance_candidates`, `get_ticker_history`, `get_db_stats`).
4. Update `docker-compose.yml` to expose `stock-ml-api` on port `8000`.
5. Update `docker/entrypoint.sh` to support `api` / `server` mode.
6. Add unit tests for API endpoints (`test/test_api.py`).

## Verification
- Unit test coverage with FastAPI `TestClient`.
- Verified endpoint responses for TW50, TW100, and S&P500.
- OpenAPI Swagger doc available at `/docs`.
- Verified in Docker container and remote host `10.9.0.99`.
