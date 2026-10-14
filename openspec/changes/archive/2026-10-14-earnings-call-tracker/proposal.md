# Proposal

## Why

法人說明會（Earnings Call / Investor Conference）是資本市場中影響股價與估值的核心催化劑（Catalyst）。無論是台股上市櫃公司的季報/自結法說，或是美股標普 500 與科技巨頭（Magnificent 7）的季度財報電話會議，皆伴隨著營運展望（Guidance）、資本支出（CapEx）調整、產品新動向以及管理層問答等關鍵非結構化訊息，直接牽動機構法人資金重新配置與波段趨勢轉折。

目前系統現有的宏觀行事曆僅提供簡易的美股財報日期，缺乏台股上市櫃法說會追蹤、缺少直播與簡報連結，亦未將法說會催化劑與個股量化訊號深度整合。本變更建立「台股與美股法說會追蹤體系（Earnings Call Tracker）」，將台股與美股法說會列為第一公民，提供一體化的跨市場行事曆、資料管道、DuckDB 快取、REST API 與金融終端互動介面。

## What Changes

- **跨市場資料擷取管道（Data Ingestion Pipeline）**：
  - **台股資料源**：對接台灣證交所 TWSE OpenAPI (`t100sb02_1`)、櫃買中心 TPEx OpenAPI (`mopsfin_t100sb02_1`)，並監聽公開資訊觀測站（MOPS）重大訊息中的法說會公告，取得會議時間、地點、簡報 (PDF) 下載網址、影音連結與會議語言。
  - **美股資料源**：對接 SEC EDGAR 8-K / 6-K 財報公告、Yahoo Finance / Investing.com 財報與法說行事曆，抓取 S&P 500、Nasdaq 100 與重磅科技股之法說會預約時程、EPS 預估值 vs 實際公布值、官方 IR Webcast 直播網址及逐字稿（Transcript）來源。
  - **自動容錯與多級快取**：實作快取降級保護機制（優先抓取上游最新資料，上游連線失敗或限流時自動使用本地快取備援）。
- **時序與行事曆快取資料庫（DuckDB Storage）**：
  - 於本地 DuckDB 建立 `earnings_calls` 專屬資料表，支援市場識別（`TW` / `US`）、股票代碼、公司名稱、會議起訖時間戳（UTC/本地標準時）、線上/線下形式、簡報/直播連結、EPS 指標及多維標籤。
  - 建立複合索引提升跨市場區間查詢與個股代碼查詢效率。
- **後端 REST API 服務**：
  - 新增 `GET /api/v1/earnings-call/upcoming`：支援 `market`（`TW`、`US`、`ALL`）、`days`（預設 14 天，亦支援 `start_date`/`end_date` 自訂區間）、`limit`、`is_weight_stock` 篩選與倒數天數計算。
  - 新增 `GET /api/v1/earnings-call/{ticker}`：查詢指定台股或美股標的之歷史與未來法說會清單、歷次簡報連結與 EPS 記錄。
  - 新增 `GET /api/v1/earnings-call/calendar`：提供以日/週/月為單位的日曆聚合摘要，統計台美股每日法說會場次。
  - 新增 `POST /api/v1/earnings-call/sync`：供排程器或管理者手動觸發台美股法說會資料同步。
- **金融終端介面整合（Frontend UI）**：
  - **操盤看板催化劑專區（Catalyst Hub）**：於看板頂部流動性與總經事件區，新增「台美重磅法說會倒數雷達」，即時標示未來 7 天內權值股法說會警告。
  - **個股詳情抽屜（Stock Drawer）**：在個股詳細抽屜中新增「法說會資訊」獨立標籤頁，展開歷年法說簡報、近期排程與官方直播入口。
  - **獨立法說會行事曆視圖（Earnings Call Calendar View）**：新增專屬行事曆頁籤，提供市場切換（台股 TW / 美股 US / 全市場 ALL）、雙時區切換（台灣 CST / 美東 EST/EDT）、關鍵字搜尋、簡報下載與 Webcast 直播一鍵跳轉。

## Capabilities

### New Capabilities
- `earnings-call-tracker`: 全市場台股（TWSE/TPEx/MOPS）與美股（SEC EDGAR/Yahoo/IR Webcast）法人說明會資料擷取、時序儲存、多維查詢與行事曆終端呈現能力。

### Modified Capabilities
<!-- 無既有 specs 需求改變，此為新增之法說會追蹤與催化劑能力 -->

## Impact

- **Affected Code**:
  - `data/earnings_fetcher.py`（新模組：負責 TWSE, TPEx, MOPS 及美股法說會/Webcast 資料抓取與解析）
  - `data/duckdb_manager.py`（擴充：新增 `earnings_calls` 表結構、查詢與批次寫入方法）
  - `api/routes/earnings_call.py`（新模組：提供法說會相關 REST API）
  - `api/main.py`（註冊 `earnings_call` 路由）
  - `api/templates/index.html`（或 partial 模組：嵌入催化劑預警、個股抽屜法說標籤與專屬法說會日曆介面）
  - `run_daily.sh`（排程：每日定時同步最新台美法說會資料）
- **APIs**:
  - 新增 `/api/v1/earnings-call/upcoming`
  - 新增 `/api/v1/earnings-call/{ticker}`
  - 新增 `/api/v1/earnings-call/calendar`
  - 新增 `/api/v1/earnings-call/sync`
- **Dependencies**:
  - 複用現有 `requests`, `pandas`, `duckdb`，無引入重型或破壞性依賴。
