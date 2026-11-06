# Upstream Canary Health Lane Specification

## Purpose

提供與生產資料庫完全隔離、具最小冷卻保護且強制繞過本地快取之多資料源金絲雀健康探測通道（Upstream Canary Health Lane），以固定探針標的平行實測上游連通性與回傳欄位結構，精準區分網路/限流異常與解析器欄位漂移（Schema Drift），並以白名單欄位原子快照提供遮罩安全之診斷檢視。

## ADDED Requirements

### Requirement: Isolated Cooldown-Protected Parallel Canary Execution
系統必須（SHALL）針對核心資料源（TWSE、TPEx、yfinance、answerbook、Investing、Polymarket、BigGo 法說會、arXiv）定義固定金絲雀探針標的，以平行執行緒、單探針嚴格逾時及預設 30 分鐘最小冷卻間隔（未傳 `--force` 且快照未滿 30 分鐘時直接回傳既有快照）在繞過快取且零生產 DuckDB 寫入連線之隔離模式下執行探測。

#### Scenario: Canary probe bypasses cache and opens zero writable DuckDB connections
- **WHEN** 排程或管理員執行 `python scripts/check_loader_health.py --force`
- **THEN** 各探針繞過本地快取向目標來源發起最小查詢，期間開啟之可寫 DuckDB 連線數為 `0`，且生產 DuckDB 資料表完全不變

#### Scenario: Rapid rerun within cooldown interval reuses recent snapshot safely
- **WHEN** 距離上次金絲雀快照產生未滿 30 分鐘且執行 `python scripts/check_loader_health.py` 未帶 `--force`
- **THEN** 腳本直接回傳現有快照而不重複向 TWSE 或外部網站發起請求，避免觸發上游 IP 封鎖

### Requirement: Deterministic Failure Taxonomy Classification
系統必須（SHALL）將每個資料源探針結果明確歸類為 `healthy`、`upstream_unreachable_or_ratelimited`（連線逾時、DNS 失敗、HTTP 403/429/5xx 或熔斷冷卻中）或 `schema_drift_parse_error`（HTTP 200 但必要欄位缺失、型別不符或空結構），並記錄延遲毫秒數與檢查時間。

#### Scenario: HTTP 429 or timeout classified as upstream_unreachable_or_ratelimited
- **WHEN** 某外部資料源回傳 HTTP 429、HTTP 503 或發生連線逾時
- **THEN** 該資料源狀態標記為 `upstream_unreachable_or_ratelimited`，且不誤判為解析器壞損

#### Scenario: HTTP 200 with missing required fields classified as schema_drift_parse_error
- **WHEN** 某外部資料源回傳 HTTP 200，但回傳 payload 缺少契約必要欄位（如收盤價、日期或預期 JSON 鍵值）
- **THEN** 該資料源狀態標記為 `schema_drift_parse_error`，並在摘要中列出缺失欄位名稱以利定位改版

### Requirement: Whitelist Atomic Sanitized Snapshot Persistence
系統必須（SHALL）以白名單欄位結構透過暫存檔寫入加 `os.replace` 原子替換更新 `data/cache/canary_health_latest.json`（強制 `allow_nan=False`），嚴禁寫入原始 Traceback，且所有摘要必須（MUST）遮罩 API Token、Cookie、內部私有主機 IP 與完整查詢字串。

#### Scenario: Snapshot written atomically with whitelist fields and secret redaction
- **WHEN** 任一探針拋出含憑證參數、私有主機 IP 或查詢字串之例外訊息
- **THEN** 寫入 `data/cache/canary_health_latest.json` 之白名單欄位已遮罩所有敏感字串且通過嚴格有限數值驗證
