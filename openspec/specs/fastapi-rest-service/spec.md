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

### Requirement: Discovery Engine REST Endpoints
The system SHALL provide REST endpoints under `/api/v1/discovery` to serve discovery candidates, arXiv papers, news signals, history trends, and pipeline status with read-only database connections and cache acceleration.

#### Scenario: Query discovery candidates
- **WHEN** client sends `GET /api/v1/discovery/candidates`
- **THEN** system returns ranked candidate stocks with total_score breakdown, recommendation label, and supporting signals

#### Scenario: Query arXiv paper signals
- **WHEN** client sends `GET /api/v1/discovery/papers`
- **THEN** system returns list of recent technical papers with extracted bottlenecks and mapped equity tickers

#### Scenario: Query discovery history trends
- **WHEN** client sends `GET /api/v1/discovery/history`
- **THEN** system returns timeline data including candidate counts, paper counts, and signal counts across past runs

### Requirement: Discovery Health and Stale Status Reporting
The system SHALL provide `/api/v1/discovery/status` returning pipeline execution health, data freshness timestamp, and stale indicator.

#### Scenario: Report healthy fresh status
- **WHEN** pipeline completed successfully on current date
- **THEN** endpoint returns `status: "success"`, `stale: false`, and current `as_of` date

#### Scenario: Report stale fallback status
- **WHEN** pipeline encountered failure and retained previous snapshot
- **THEN** endpoint returns `status: "fallback"`, `stale: true`, and the previous valid snapshot timestamp

### Requirement: Forecast performance read endpoints
系統 SHALL 提供 GET `/api/v1/performance/stocks`、`/api/v1/performance/models`、`/api/v1/performance/tickers/{ticker}`、`/api/v1/models/health`，支援市場、版本、目標類型、步長、視窗與 as_of 等適用篩選，回傳樣本、覆蓋、方法版本、產出時間、用途資格與 stale 狀態。

#### Scenario: Invalid filter or insufficient evidence
- **WHEN** 視窗或步長無效，或有效樣本不足
- **THEN** 無效參數回傳驗證錯誤；證據不足回傳明確狀態與空排名，維持穩定分頁及排序

### Requirement: Simulation and risk read endpoints
系統 SHALL 提供 GET `/api/v1/backtests/{run_id}`、`/api/v1/paper/accounts/{account_id}`、其 `/orders`、`/equity` 與 `/risk` 子資源；僅公開經設定允許的模擬帳戶，所有查詢維持唯讀且不觸發推論、成交或帳戶重設。

#### Scenario: Query public simulated account
- **WHEN** 使用者查詢被允許公開的帳戶
- **THEN** 回傳模擬標記、成本與績效、資料時間及風控狀態；未知或未公開帳戶回傳 404

### Requirement: Verification provenance in history
系統 SHALL 在歷史預測查詢提供 forecast ID、模型版本、目標交易日、驗證狀態與正式開獎值，保留既有欄位相容性。

#### Scenario: Pending actuals in historical chart
- **WHEN** 目標日尚未正式開獎
- **THEN** API 標記 pending，圖表與 PK 不使用後續預測紀錄冒充實際行情
