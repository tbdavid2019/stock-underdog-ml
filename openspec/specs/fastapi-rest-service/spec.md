# fastapi-rest-service Specification

## Purpose
TBD - created by archiving change add-fastapi-duckdb-service. Update Purpose after archive.

## Requirements

### Requirement: Predictions Query Endpoints
The system SHALL provide REST endpoints to query quantitative predictions and strategy signals from DuckDB:
1. `GET /api/v1/predictions/latest`: Query latest snapshot by index with pagination.
2. `GET /api/v1/predictions/resonance`: Filter multi-strategy overlap and `🏆多維共振` candidates.
3. `GET /api/v1/predictions/xuantie`: Filter XuanTie MA pullback buy points.
4. `GET /api/v1/predictions/timestral/top-bullish`: Sort and retrieve top Mistral Timestral predicted gainers.
5. `GET /api/v1/predictions/timestral/top-bearish`: Sort and retrieve top Mistral Timestral predicted losers.
6. `GET /api/v1/predictions/history/{ticker}`: Retrieve time-series prediction trajectory.

#### Scenario: Querying latest resonance stocks
- **WHEN** client sends `GET /api/v1/predictions/resonance?min_hits=2`
- **THEN** system returns JSON list of candidates matching dual/triple resonance criteria with PE, PB, TimesFM, Timestral, and strategy tags.

#### Scenario: Querying Timestral top bullish predictions
- **WHEN** client sends `GET /api/v1/predictions/timestral/top-bullish?limit=20`
- **THEN** system returns top stocks sorted by Timestral forward potential percentage with P10/P50/P90 quantile data.

#### Scenario: Backward compatibility for legacy LSTM endpoints
- **WHEN** client sends `GET /api/v1/predictions/lstm/top-bullish` or `/api/v1/predictions/lstm/top-bearish`
- **THEN** system redirects or maps to Timestral foundation model prediction results with an HTTP response containing backward-compatible field aliases.

### Requirement: Macro Regime Status Endpoint
The system SHALL provide `GET /api/v1/macro/latest` returning current regime, exposure level, VIX, and warning signals.

#### Scenario: Querying macro status
- **GIVEN** Macro regime evaluation has been executed
- **WHEN** client sends `GET /api/v1/macro/latest`
- **THEN** system returns current market regime, suggested exposure percentage, and VIX/SOX indicators.

### Requirement: Database Analytics and Health Endpoints
The system SHALL provide `GET /health` and `GET /api/v1/stats/summary` reporting database health, record counts, and date coverage.

#### Scenario: Querying system health
- **GIVEN** DuckDB is initialized and accessible
- **WHEN** client sends `GET /health`
- **THEN** system returns HTTP 200 with database status `healthy` and total row count.

### Requirement: Container and Service Integration
The system SHALL support launching the FastAPI server on port 8000 via `docker compose up -d stock-ml-api` and `python -m api.main`.

#### Scenario: Running via Docker compose
- **GIVEN** `docker-compose.yml` is configured with `stock-ml-api` service
- **WHEN** user runs `docker compose run --rm stock-ml-api`
- **THEN** uvicorn launches on `0.0.0.0:8000` with Swagger docs enabled.
