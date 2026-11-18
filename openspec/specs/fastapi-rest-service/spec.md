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

### Requirement: Watchlist and Quant Alert REST and MCP Endpoints
系統 SHALL 提供自選股與量化告警之 REST 端點（`/api/v1/watchlist`、`/api/v1/alerts`、`/api/v1/alerts/check`）與唯讀公開 MCP 查詢工具；生產環境下所有寫入與巡檢操作 MUST 驗證授權金鑰（無論透過 REST 或業務層呼叫），遇跨進程資料庫寫鎖忙碌則明確回傳 HTTP 503。

#### Scenario: Querying enriched watchlist via read-only connection
- **WHEN** 客戶端發送 `GET /api/v1/watchlist` 或呼叫公開唯讀 MCP 自選查詢工具
- **THEN** 系統以唯讀連線回傳持久化自選股清單及最新預測摘要，絕不升級搶佔寫入鎖

#### Scenario: Rejecting unauthorized mutation or alert check across REST and MCP
- **WHEN** 生產環境下未帶有效金鑰呼叫 `POST /api/v1/watchlist`、`POST /api/v1/alerts/check` 或嘗試執行寫入操作
- **THEN** 系統拒絕執行並回傳 HTTP 401（或授權失敗錯誤），不修改資料庫亦不觸發外部推播

### Requirement: Cross-Market Live Session Summary Including Lunch Break
系統 SHALL 提供 `GET /api/v1/market/calendar/sessions` 端點，回傳台股（TW）、美股（US，含夏冬令與提前收盤日）與日股（JP，含東京證交所 11:30–12:30 午間休市）之當前交易階段（`pre_market`、`regular`、`lunch_break`、`after_hours`、`closed`）、下次轉換時間戳與休市原因。

#### Scenario: Fetching Tokyo midday break and US DST-aware session status
- **WHEN** 於東京時間交易日 12:00 呼叫 `GET /api/v1/market/calendar/sessions`
- **THEN** 系統將 `JP` 狀態標示為 `lunch_break` 並提供 12:30 後場開盤倒數，不誤標為盤中連續交易

### Requirement: Ticker Signal Alignment and Sanitized Health Endpoints
系統 SHALL 提供 `GET /api/v1/predictions/alignment/{ticker}` REST 端點與對應唯讀 MCP 工具回傳多源訊號共識，並對健康度檢查端點（含 `/api/v1/decision/health`）套用滑動視窗限流、回應快取與上游 URL/憑證遮罩。

#### Scenario: Querying signal alignment and rate-limited sanitized health
- **WHEN** 客戶端查詢 `/api/v1/predictions/alignment/2330.TW` 或 `/api/v1/decision/health`
- **THEN** 訊號端點回傳結構化共識明細；健康端點回傳遮罩敏感路徑與參數後之狀態且受速率限制保護

### Requirement: Bounded Read-Only Risk X-Ray and Protected Paper Account Endpoints
系統必須（SHALL）提供受每 IP 滑動視窗限流與進程級併發 Semaphore 保護之唯讀端點：`POST /api/v1/risk/xray`（上限 `<= 100` 檔、`window` `30..504`）、`GET /api/v1/watchlist/risk-xray`（當自選股表未就緒時優雅回傳 `status: "watchlist_unavailable"`）與 `POST /api/v1/risk/optimize-weights`（上限 `<= 50` 檔，零 DB 寫入）；私有模擬帳戶診斷端點 `GET /api/v1/paper/accounts/{account_id}/risk-xray` 必須（MUST）強制驗證 `account_id` 格式並在生產模式要求 API Key 認證以防枚舉洩漏。

#### Scenario: Oversized basket rejected with HTTP 422 before matrix computation
- **WHEN** 客戶端向 `POST /api/v1/risk/xray` 傳入超過 100 檔代號或含負數/非有限權重之 payload
- **THEN** API 在讀取日 K 與建構共變異矩陣前直接回傳 HTTP 422，且不完整回顯超長輸入以防日誌污染

#### Scenario: Unauthenticated request to private paper account Risk X-Ray blocked
- **WHEN** 客戶端在 `production` 模式未帶合法 API Key 呼叫 `GET /api/v1/paper/accounts/main/risk-xray`
- **THEN** 伺服器回傳 HTTP 401/403 拒絕存取，不洩漏該 `account_id` 是否存在及其持倉明細

### Requirement: Cached Canary Health Snapshot Endpoint
系統必須（SHALL）提供 `GET /api/v1/stats/canary-health` 唯讀端點與對應唯讀 MCP 工具，僅讀取最近一次由背景排程或 CLI 產出之 `data/cache/canary_health_latest.json` 快照（若檔案尚未建立則回傳 `status: "not_initialized"`），嚴禁由公開匿名 GET 請求直接觸發對外部上游網站之即時無快取探測風暴。

#### Scenario: Public GET reads cached snapshot without live upstream requests
- **WHEN** 匿名客戶端連續呼叫 `GET /api/v1/stats/canary-health`
- **THEN** 伺服器僅讀取本地白名單已遮罩快照檔並回傳各資料源健康分類，不向 TWSE、Investing 或其他外部來源發起任何即時網路連線
