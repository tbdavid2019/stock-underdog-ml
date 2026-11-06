# Spec Delta

## ADDED Requirements

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
