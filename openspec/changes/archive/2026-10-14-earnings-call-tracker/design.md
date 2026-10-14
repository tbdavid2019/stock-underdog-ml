# Design

## Context

法人說明會（Earnings Call）是驅動股票市場情緒與估值重塑的最強催化劑之一。台股與美股投資人皆高度依賴法說會發布的財報結果、資本支出（CapEx）修訂與管理層對次季/全年的業績指引（Guidance）。

現有系統已具備總經利率、Commodity 與基礎美股財報日期的抓取能力（見 `data/investing_service.py`），並透過 DuckDB 儲存時序行情與量化預測。然而，現狀存在以下缺口：
1. 缺乏台股上市公司（TWSE）與上櫃公司（TPEx）官方第一手之法說會結構化時程與簡報下載管道。
2. 美股財報會議資訊未與 Webcast 直播、逐字稿與實際/預期 EPS 數據對齊，美股未被視為一級公民。
3. 操盤看板與個股抽屜缺乏即時法說會催化劑預警，使用者無法在進行量化選股與籌碼分析時同步評估法說會事件風險。

本設計為 `earnings-call-tracker` 提供跨市場架構藍圖。更多背景說明參見 [proposal.md](proposal.md)。

## Goals / Non-Goals

**Goals:**
- **跨市場一級公民架構**：完整納入台股（TWSE/TPEx/MOPS）與美股（S&P 500、Nasdaq 100、Magnificent 7）法說會時程、簡報與影音直播。
- **統一資料模型與持久化儲存**：於 DuckDB 建立標準化 `earnings_calls` 時序結構，抹平跨國交易所資料格式差異。
- **雙時區即時轉換**：儲存標準 UTC 時間戳，API 同步輸出台灣時間（CST, UTC+8）與美東時間（EST/EDT），支援前端一鍵切換。
- **催化劑與終端深度整合**：將法說會融入操盤看板催化劑預警、個股詳情抽屜（Stock Drawer）以及獨立法說會行事曆視圖。
- **健壯的快取與降級機制**：遵照「優先抓取最新、異常時本地快取降級」鐵律，並維持輕量運算資源消耗。

**Non-Goals:**
- 本地端不進行音訊語音辨識（ASR）或自行託管大型多媒體檔案，直播與錄音直接採用官方 IR/YouTube/Webcast 串流外鏈。
- 不引入需高額付費授權之彭博/路透專有金融終端 API。

## Decisions

### 1. 混合式跨市場資料擷取器 (`EarningsCallFetcher`)

- **台股管道**：
  - TWSE OpenAPI `t100sb02_1`（上市公司法人說明會一覽表）
  - TPEx OpenAPI `mopsfin_t100sb02_1`（上櫃公司法人說明會一覽表）
  - 公開資訊觀測站 (MOPS) 重大訊息法說會公告備援
- **美股管道**：
  - Yahoo Finance Earnings Calendar & Quote Summary API
  - Investing.com 財報與法說行事曆
  - SEC EDGAR 8-K / 6-K 財報公告與官方企業 IR 頁面 Webcast 抓取
- **決策理由**：
  - TWSE 與 TPEx 提供免費且權威的官方 OpenAPI，單次請求即可取得全市場資料。
  - 美股結合 Yahoo Finance 與 Investing.com 既有客戶端，能以最低網路開銷取得 EPS 預估/實際值與 Webcast 串流。
- **替代方案**：
  - *FinMind / AlphaVantage*：免費額度限制嚴格，且缺乏台股法說會簡報 PDF 原始連結。

### 2. DuckDB `earnings_calls` 表模型設計

於 `data/duckdb_manager.py` 初始化建立 `earnings_calls` 表：

```sql
CREATE TABLE IF NOT EXISTS earnings_calls (
    id VARCHAR PRIMARY KEY,             -- 格式: {MARKET}_{TICKER}_{CALL_DATE} (e.g. TW_2330.TW_20261015, US_NVDA_20261120)
    market VARCHAR NOT NULL,            -- TW / US
    ticker VARCHAR NOT NULL,            -- 2330.TW, NVDA, AAPL 等
    name VARCHAR NOT NULL,              -- 台灣積體電路, NVIDIA Corporation
    call_date VARCHAR NOT NULL,         -- YYYY-MM-DD
    call_time VARCHAR,                  -- HH:MM (例如 14:00)
    timezone VARCHAR NOT NULL,          -- Asia/Taipei 或 America/New_York
    timestamp_utc TIMESTAMP,            -- 標準化 UTC 時間戳，供計算倒數與跨時區排序
    meeting_type VARCHAR,               -- physical / online / hybrid
    location VARCHAR,                   -- 實體地址或線上平台 (Webex / Teams / Zoom)
    webcast_url VARCHAR,                -- 官方直播或錄音播放 URL
    presentation_url VARCHAR,           -- 簡報 PDF 下載連結
    transcript_url VARCHAR,             -- 逐字稿連結
    language VARCHAR,                   -- zh-TW / en-US / bilingual
    eps_estimate DOUBLE,                -- 市場預估 EPS
    eps_actual DOUBLE,                  -- 實際公布 EPS
    revenue_estimate DOUBLE,            -- 預估營收
    revenue_actual DOUBLE,              -- 實際營收
    is_weight_stock BOOLEAN DEFAULT FALSE, -- 是否為重量級權值股 (台股 50 大 / 美股 M7, S&P 500)
    remarks VARCHAR,                    -- 備註說明 (如: 召開 2026 年第三季財務及營運報告)
    source VARCHAR,                     -- TWSE / TPEX / MOPS / YAHOO / SEC_EDGAR
    created_at VARCHAR,
    updated_at VARCHAR
);
```

- **索引策略**：建立 `(market, call_date)`、`(ticker, call_date)` 索引，支援即將到期排序與個股歷程檢索。
- **Upsert 機制**：寫入時以 `id` 進行 `INSERT OR REPLACE`，保留最新公布的簡報與實際財務數字。

### 3. 雙時區與催化劑倒數智慧轉換

- 後端統一維護 `timestamp_utc`，API 響應中自動附加：
  - `datetime_cst`: 台灣時間字串（如 `2026-10-15 14:00 CST`）
  - `datetime_est`: 美東時間字串（如 `2026-10-15 02:00 EDT`）
  - `countdown_days`: 距今剩餘天數（負值代表已結束）
  - `status`: `upcoming` / `today` / `completed`
- 前端全域支援「台灣時間 (CST)」與「美東時間 (EST)」切換開關，徹底解決台美跨時區查看困擾。

### 4. RESTful API 規格

在 `api/routes/earnings_call.py` 提供四支高階端點：
1. `GET /api/v1/earnings-call/upcoming`:
   - 參數：`market` (`TW` | `US` | `ALL`), `days` (預設 14), `is_weight_stock` (bool), `limit` (預設 50)。
   - 返回即將舉行的法說會排序清單。
2. `GET /api/v1/earnings-call/{ticker}`:
   - 參數：`ticker` (如 `2330.TW`, `NVDA`, `TSLA`)。
   - 返回該標的之歷史法說會簡報列表、未來預定時間與歷次 EPS 紀錄。
3. `GET /api/v1/earnings-call/calendar`:
   - 參數：`year` (int), `month` (int), `market` (`TW` | `US` | `ALL`)。
   - 返回日曆格子資料（每日場次計數、重要標的預告）。
4. `POST /api/v1/earnings-call/sync`:
   - 手動或排程觸發即時資料同步與 DuckDB 更新。

### 5. 金融終端前端 UI 整合

- **操盤看板催化劑專區 (Catalyst Section)**：
  - 於總經流動性卡片下方，新增「重磅法說會倒數雷達」。
  - 顯示未來 7 天內權值股法說會（如台積電、聯發科、輝達、蘋果），點擊即打開簡報或個股抽屜。
- **個股詳情抽屜 (Stock Drawer)**：
  - 新增「法說會資訊」Tab。
  - 列出歷次法說會時間、營運指引重點、簡報下載按鈕 (PDF) 與直播連結。
- **獨立法說會行事曆視圖 (Dedicated Calendar View)**：
  - 參考專業 Earnings Call 行事曆與 Bloomberg Terminal 排版。
  - 頂部包含三向開關（`全部 ALL` / `台股 TW` / `美股 US`）與時區開關（`CST` / `EST`）。
  - 卡片式與清單式兩種視圖切換，呈現公司標誌、代號、時間、EPS Surprise 對比柱狀圖與簡報下載按鈕。

## Risks / Trade-offs

- **[Risk 1] 上游交易所 OpenAPI 偶發連線超時或 302 重定向**
  → *Mitigation*: Fetcher 設置合規 User-Agent，啟用自動重試機制（3 次指數退避），並在連線失敗時自動讀取本地最近一次 snapshot 快取。
- **[Risk 2] 美股 Webcast 直播需跳轉第三方平台或需要登入**
  → *Mitigation*: 優先提取官方 Investor Relations (IR) 直播入口與各大財經入口（Yahoo Finance、YouTube 官方直播），並明確標示「官方 IR 外鏈」提示。
- **[Risk 3] 本機資源緊繃，全市場資料寫入可能造成資料庫鎖定**
  → *Mitigation*: 使用 DuckDB 批量事務寫入 (`executemany`)，抓取工作使用 Python 輕量非同步或單線程批次執行，控制記憶體在 100MB 以內。

## Migration Plan

1. **資料庫遷移**：啟動時由 `DuckDBManager` 自動執行 `CREATE TABLE IF NOT EXISTS earnings_calls`，無需手動遷移現有表格。
2. **排程整合**：在 `run_daily.sh` 中加入 `python -m data.earnings_fetcher --sync`，每日開盤前（08:00 CST）及美股收盤後（05:00 CST）各同步一次。
3. **回滾策略**：若新端點或抓取有異常，可由配置項 `ENABLE_EARNINGS_CALL_TRACKER=false` 停用前端頁籤與排程，現有看板與預測流程不受任何干擾。
