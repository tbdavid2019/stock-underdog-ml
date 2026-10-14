# earnings-call-tracker Specification

## Purpose
提供台股與美股全市場法人說明會（Earnings Call / Investor Conference）的一體化追蹤體系，涵蓋跨市場資料擷取、DuckDB 時序儲存、FastAPI 查詢介面與金融終端互動行事曆。

## Requirements

### Requirement: Cross-Market Earnings Call Data Ingestion
系統 SHALL 支援台股與美股法說會資料的多源自動化擷取，涵蓋 TWSE OpenAPI、TPEx OpenAPI、MOPS 重大訊息、SEC EDGAR 公告、Yahoo Finance 與 Investing.com，並在上游服務暫時無法連線時自動降級使用本地快取。

#### Scenario: Scheduled ingestion of Taiwan market earnings calls
- **WHEN** 系統執行每日法說會排程任務時
- **THEN** 系統自動向 TWSE OpenAPI 與 TPEx OpenAPI 請求最新法說會清單，解析會議日期、時間、實體地點、線上會議網址、簡報 PDF 連結與會議語言，並完成結構化儲存

#### Scenario: Scheduled ingestion of US market earnings calls and webcasts
- **WHEN** 系統執行美股市場法說會排程任務時
- **THEN** 系統向美股財報行事曆與官方 IR Webcast 來源抓取 S&P 500 與 Nasdaq 100 企業之財報會議時間、EPS 預估值與官方直播連結

#### Scenario: Upstream service failure and cache fallback
- **WHEN** 上游交易所或外部行事曆 API 連線超時或返回錯誤代碼時
- **THEN** 系統發出警告日誌，並自動回退讀取本地最近一次成功同步之快取資料，確保服務不中斷

### Requirement: DuckDB Earnings Call Storage and Indexing
系統 SHALL 在 DuckDB 中維護 `earnings_calls` 資料表，記錄市場類型、標的代碼、公司名稱、會議起訖時間、時區、會議型態、簡報連結、影音連結與 EPS 指標，並建立複合索引以加速檢索。

#### Scenario: Upserting earnings call records without duplicates
- **WHEN** 資料採集管道寫入抓取到的法說會記錄時
- **THEN** 系統依據市場代碼、股票代號與會議日期進行唯一性去重與更新，避免重複資料堆疊

#### Scenario: Querying earnings call storage by ticker
- **WHEN** 應用程式透過標的代號查詢法說會歷史時
- **THEN** 系統能以次秒級延遲自 DuckDB 返回該標的歷次法說會記錄、簡報下載網址與 EPS 數據

### Requirement: RESTful Query Endpoints for Upcoming Calls and History
系統 SHALL 提供符合 REST 規範的 FastAPI 端點，支援即將舉行的法說會清單過濾、特定標的法說會歷程查詢、行事曆月度聚合統計與手動觸發同步。

#### Scenario: Querying upcoming calls with market filter
- **WHEN** 客戶端向 `/api/v1/earnings-call/upcoming` 發送請求並指定 `market=US`
- **THEN** 系統返回未來排定之美股法說會清單，包含公司代碼、美東時間 (EST) 與台灣時間 (CST)、EPS 預期與 Webcast 連結

#### Scenario: Querying upcoming calls for all markets
- **WHEN** 客戶端向 `/api/v1/earnings-call/upcoming` 發送請求並指定 `market=ALL`
- **THEN** 系統返回依會議時間排序之台股與美股法說會合併清單

#### Scenario: Querying specific ticker earnings call history
- **WHEN** 客戶端向 `/api/v1/earnings-call/{ticker}` 發送 GET 請求（如 `2330.TW` 或 `NVDA`）
- **THEN** 系統返回該標的之歷史法說會紀錄、簡報 PDF 連結與未來已排程之法說會資訊

#### Scenario: Querying monthly calendar event distribution
- **WHEN** 客戶端向 `/api/v1/earnings-call/calendar` 請求指定年月份的事件分佈
- **THEN** 系統返回該月份每日台股與美股法說會場次統計與重點個股清單

### Requirement: Catalyst Hub Integration and Warning Indicators
系統 SHALL 於操盤看板整合台美股法說會催化劑預警，針對即將召開法說會之重磅權值股與自選觀察股顯示醒目標籤與倒數計時。

#### Scenario: Highlighting upcoming weight stock earnings call in catalyst zone
- **WHEN** 台股權值股（如台積電、聯發科）或美股科技巨頭（如 Magnificent 7）預計在 7 天內召開法說會
- **THEN** 操盤看板催化劑專區自動產生高優先級預警卡片，標明距會議天數、營運展望焦點與重要程度標籤

### Requirement: Stock Drawer Earnings Call Tab
系統 SHALL 在個股詳情抽屜（Stock Drawer）中提供專屬「法說會」標籤頁，直觀展示該標的之法說會歷程、官方簡報下載、直播 Webcast 入口與 EPS 預估比對。

#### Scenario: User inspects earnings call tab in stock drawer
- **WHEN** 使用者在個股抽屜中切換至「法說會」標籤頁
- **THEN** 介面渲染該股票最近期法說會時間倒數、官方簡報 PDF 預覽/下載按鈕、影音直播連結及歷史法說會摘要列表

### Requirement: Interactive Multi-Market Earnings Call Calendar View
系統 SHALL 提供專屬之法說會行事曆視覺化視圖，支援台股與美股切換、雙時區（台灣時間 CST 與美東時間 EST）即時切換、關鍵字搜尋與一鍵跳轉外部資源。

#### Scenario: User toggles market filter in calendar view
- **WHEN** 使用者在法說會行事曆頁籤點選「美股 (US)」標籤
- **THEN** 行事曆視圖即時切換為美股專屬排程，列出標普 500 等美股企業之財報法說會時間與直播連結

#### Scenario: User toggles display timezone
- **WHEN** 使用者切換時區顯示為「美東時間 (EST)」或「台灣時間 (CST)」
- **THEN** 行事曆中所有會議時間戳自動轉換為所選時區，並清晰標示時區代碼避免跨國時差混淆
