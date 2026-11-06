# Proposal

## Why

888 StockQuant 在時序基礎大模型（TimesFM 3.0、Timestral-209M、MOIRAI-2）、台股三大法人/分點籌碼、AI 產業鏈圖譜與 FastMCP 工具生態上已具備深厚護城河，但在高頻日常操盤的「前台切換效率、個人自選追蹤與目標價/防守位自動盯盤閉環、跨分頁多源訊號共識判讀、公開連結社群/SEO 預覽、以及外部上游限流熔斷與部署前環境健檢」仍有明顯優化空間。借鏡開源熱門專案 OpenStock 的高留存產品體驗，並結合本專案獨有的本地 DuckDB 列式庫、跨市場行事曆與量化分位數（P10/P50/MA60）優勢，可在維持「免登入牆、免前端編譯步驟、單體極速部署」架構下完成全方位升級。

## What Changes

- **P0 操盤體驗三件套（前端互動與 DuckDB 自選同步）**：
  - 新增全域 `⌘K / Ctrl+K` 極速搜尋指令面板（串接毫秒級股票名稱/代號解析器），支援鍵盤上下選股直接展開個股抽屜或跳轉歷史 K 線 PK 頁。
  - 新增瀏覽器式「最近瀏覽 / 釘選個股快捷列（Recent Ticker Tabs）」，方便於多檔研究標的間一鍵切換比對。
  - 新增 **DuckDB 持久化 `★ 我的自選股`（Watchlist）**：支援主力機跨瀏覽器同步、前端樂觀更新與 URL 分享參數匯出，並於看板篩選器「維度 1：標的市場」新增 `★ 我的自選 (N)` 快速過濾按鈕。
  - 新增跨市場（🇹🇼 台股 / 🇺🇸 美股 / 🇯🇵 日股）即時開休市狀態、盤前倒數指示器與明確資料新鮮度（`as_of` / `stale`）標記，串接既有 `MarketCalendar` 引擎。
- **P1 量化決策閉環與公開連結成長引擎**：
  - 新增 **DuckDB 持久化「量化價位與均線告警（Quant Price & Regime Alerts）」**：支援一鍵訂閱「跌破模型 P10 下行防守價」、「觸及 P50 目標價」、「回踩 MA60/MA120 支撐」或自訂目標價，由排程/API 巡檢並透過既有 Telegram / Discord / Email 通道推播（含防重複洗版冷卻機制）。
  - 新增 **個股「多源訊號一致度卡（Signal Alignment Card）」**：不依賴外部付費情緒 API，直接聚合平台內建之「雙時序大模型方向、三大法人/主力分點籌碼（海外市場明確標示不適用）、Podcast 科技實體多空姿態、arXiv 前沿發現訊號」，輸出一致看多、一致看空或「模型與籌碼/輿情背離警示（Divergence Warning）」。
  - 新增 **公開個股獨立 SSR 落地頁（`GET /s/{ticker}`）與動態 Sitemap（`/sitemap.xml`、`/robots.txt`）**：提供免登入可索引之個股量化摘要、動態 OpenGraph/Twitter 社群預覽標題描述、JSON-LD 結構化標記，以及基於產業價值鏈之同業關聯標的（Peers）橫向跳轉。
- **P2 上游韌性守衛與運維健檢工具**：
  - 新增共用上游守護模組（`core/upstream_guard.py`）：支援逗號分隔多把 API Key 輪詢池（Key Pool Rotation）、單 Key 最大併發請求閘門（In-flight Semaphore）與 60 秒失敗熔斷冷卻（`RecentFailure` Cooldown，避免上游故障時阻塞請求或洗版日誌）。
  - 新增環境變數啟動前健檢工具（`scripts/check_env.py`）：分類檢查必填、選填（標註未設定時之自動降級行為）與已棄用變數，嚴格遮罩敏感值不外洩 secrets，並整合至 `scripts/install_new_machine.sh`。

## Capabilities

### New Capabilities
- `watchlist-and-quant-alerts`: 基於 DuckDB 持久化之免登入自選股管理、量化關鍵價位（目標價 / P10 防守位 / MA60 支撐）告警規則設定、自動巡檢觸發與 Telegram/Discord/Email 防洗版推播通知。
- `signal-alignment-synthesis`: 聚合平台內部多維訊號（時序基礎模型預測、台股法人與分點籌碼、Podcast 實體多空、前沿雷達訊號）以計算個股多源共識度與背離警示之合成評估能力。
- `public-ticker-seo`: 提供伺服器端渲染之 `/s/{ticker}` 公開個股量化摘要頁、動態社群預覽 Meta、JSON-LD 結構化資料、同業關聯導覽與 `/sitemap.xml` 搜尋引擎索引支援。
- `upstream-resilience-and-env-doctor`: 提供外部 API 多金鑰輪詢池、單金鑰併發限制、短時熔斷冷卻機制，以及遮罩安全之 `.env` 部署前環境健檢腳本。

### Modified Capabilities
- `web-dashboard-ui`: 擴充全域 `⌘K` 搜尋面板、最近查看個股快捷分頁列、`★ 我的自選` 市場篩選膠囊、跨市場開休市盤前倒數與資料新鮮度標章，並於個股抽屜整合多源訊號一致度卡與量化到價告警訂閱介面。
- `fastapi-rest-service`: 新增自選股 CRUD、量化告警 CRUD 與巡檢、跨市場即時交易時段摘要、以及個股多源訊號一致度查詢之 REST API 端點與對應 FastMCP/WebMCP 工具宣告。

## Impact

- **資料庫 (`data/duckdb_manager.py`)**：新增 `watchlist_items` 與 `quant_price_alerts` 兩張輕量資料表（不影響既有 `predictions`、`tw_daily_bars` 等歷史表結構；唯讀查詢維持 `read_only=True`，寫入操作透過既有 `_WRITE_LOCK` 序列化並支援選配 `WATCHLIST_API_KEY` 寫入保護）。
- **後端與 API (`api/main.py`, `api/routes/`, `mcp_server.py`, `core/`)**：
  - 新增 `api/routes/watchlist.py`（含自選股與量化告警端點）、擴充 `api/routes/market.py`（跨市場即時 session 摘要）與 `api/routes/predictions.py`（個股多源訊號一致度）。
  - `api/main.py` 新增 `/s/{ticker}` SSR 路由、`/sitemap.xml` 與 `/robots.txt`。
  - 新增 `core/upstream_guard.py`、`data/signal_alignment.py`、`data/alert_evaluator.py`。
  - `mcp_server.py` 同步暴露自選股、量化告警與訊號一致度查詢工具。
- **前端 (`api/templates/`, `api/static/js/`)**：維持免 Node.js 編譯之原生 Vue 3 + Tailwind + Jinja2 partials 架構，嚴格遵循既有曜石黑 (`#030302`) / 暖米白 (`#FFF9E8`) + 琥珀金 (`#FFB010`) 雙主題與 WCAG AA/AAA 對比度標準。
- **腳本與文件 (`scripts/`, `run_daily.sh`, `README.md`, `docs/CHANGELOG.md`)**：新增 `scripts/check_env.py` 與告警巡檢入口，並依開發鐵律同步更新 `README.md` 與 `docs/CHANGELOG.md`。
