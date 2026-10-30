# Proposal: AI 產業鏈前沿挖掘機 (AI Tech Radar & Discovery Engine)

## Why

目前系統的量化選股體系主要聚焦於「右側交易與技術面跟隨」（玄鐵均線、TimesFM / Timestral 時序走勢預測、三大法人籌碼鎖碼與 Podcast 研報解讀），但缺乏「左側前瞻題材挖掘」能力——即由學術界前沿技術突破（如 arXiv 論文發布）、新興硬體與架構瓶頸，直接映射到台美股核心供應鏈的早期探索機制。

借鑒 `stocks.mastersgo.cc`（monk 宇宙 · 挖掘機）的高資訊密度、輕量化邊緣架構與前沿雷達設計，我們引入「AI 產業鏈前沿挖掘機」，在技術瓶頸萌芽初期即主動捕捉候選標的，並透過「代號正規化與白名單硬過濾」、「確定性 7 維多因子評分」與「原子化快取發布」，與既有雙 Foundation Model 及籌碼面形成「左側發現 + 右側確認」的高可靠決策閉環。

## What Changes

- **建立代號正規化與白名單校驗器 (Ticker Canonicalizer & Whitelist)**：統一處理台股代號（`.TW` / `.TWO` 正規化）、美股大小寫，並與系統現有股票池進行白名單硬交集，杜絕虛構或下市標的。
- **外部化台美供應鏈對照庫 (Supply Chain Map)**：建立 `config/discovery_supply_chain.json`，對齊前沿技術瓶頸（CPO、HBM/KV-Cache、液冷散熱、具身智能機器人）至美股錨點與台灣關鍵供應鏈。
- **固化確定性 7 維度多因子評分引擎 (Deterministic Scoring Engine)**：明訂各維度權重向量與歸一化標準，輸出閉區間 `[0, 100]` 之總分，並嚴格分流為 `propose_add`、`observe`、`already_in_pool` 與 `reject`。
- **資料管線與原子雙層持久化 (Pipeline & Atomic Persistence)**：於 `data/duckdb_manager.py` 擴充資料表結構，實作交易式寫入與 `tempfile + os.replace` 原子替換靜態 JSON 快取，防止分裂腦。
- **DuckDB 讀寫連線分離與防撞鎖機制**：FastAPI 查詢端強制使用 `read_only=True`，Pipeline 寫入端具備重試退避機制；每日排程加入 `flock` 防重疊執行。
- **擴充 FastAPI REST 服務**：提供 `/api/v1/discovery/*` 端點，支援快取加速、條件篩選與 stale 降級狀態標示。
- **全新彭博終端分頁「前沿發現」(`tab_discovery.html`)**：適配手機端底部第 5 個 Tab 滾動容器，實作 4/2/1 響應式斷點卡片佈局、原生 Inline SVG 迷你趨勢圖與個股研究卡。
- **公開契約與文件同步**：同步更新 `docs/CHANGELOG.md`、`README.md` 以及公開 AI 契約 `/llms.txt`。

## Capabilities

### New Capabilities
- `tech-radar-discovery-engine`: 負責前沿論文與科技快訊定時擷取、LLM 技術瓶頸語意解析、產業鏈題材映射、標的正規化校驗、7 維度確定性評分與原子快照發布。

### Modified Capabilities
- `fastapi-rest-service`: 新增 Discovery 相關 REST 接口，支援唯讀 DuckDB 與快取加速讀取，並提供健康狀態與資料過期標示。
- `web-dashboard-ui`: 在既有深淺雙色彭博風格介面中擴充「前沿發現 (Discovery)」視圖，支援手機底部導覽列滾動適配、4/2/1 斷點矩陣、趨勢迷你圖與研究卡展開。

## Impact

- **設定與資料**：新增 `config/discovery_supply_chain.json`。
- **核心模組**：新增 `core/ticker_canonicalizer.py`、`core/discovery_scorer.py`、`core/supply_chain_map.py`。
- **資料庫與管線**：修改 `data/duckdb_manager.py`，新增 `pipeline/discovery_pipeline.py`。
- **API 與資料模型**：修改 `api/schemas.py`，新增 `api/routes/discovery.py`，更新 `api/main.py` 與根目錄 `/llms.txt`。
- **前端範本與樣式**：新增 `api/templates/partials/tab_discovery.html`，更新 `api/templates/partials/header.html` 與 `api/templates/index.html`。
- **自動化排程**：更新 `run_daily.sh`（增加 `flock` 檔案鎖）。
