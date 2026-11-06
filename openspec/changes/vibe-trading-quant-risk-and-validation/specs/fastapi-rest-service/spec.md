# FastAPI REST Service Specification (Delta)

## ADDED Requirements

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
