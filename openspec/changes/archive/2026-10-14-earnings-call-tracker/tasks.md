# Tasks

## 1. 資料庫模型與快取基礎建設 (DuckDB Storage & Cache Infrastructure)

- [x] 1.1 在 `data/duckdb_manager.py` 中新增 `earnings_calls` 資料表結構（包含市場 TW/US、標的代碼、時間戳 UTC、雙時區、簡報/影音連結、EPS 指標與權值股標記），並驗證資料庫初始化時表結構正確建立。
- [x] 1.2 在 `DuckDBManager` 中實作 `save_earnings_calls_batch` 與 `get_earnings_calls` 檢索方法，支援唯一鍵 upsert 去重與多條件過濾，並編寫單元測試驗證儲存與查詢正確性。
- [x] 1.3 建立快取目錄 `cache/earnings/` 與 JSON 檔案讀寫/降級備援機制，驗證上游離線時自動回退快取之行為。

## 2. 跨市場法說會資料擷取管道 (Multi-Market Ingestion Pipeline)

- [x] 2.1 實作 `data/earnings_fetcher.py` 之台股抓取器：解析 TWSE OpenAPI (`t100sb02_1`) 與 TPEx OpenAPI (`mopsfin_t100sb02_1`)，並驗證台股上市櫃法說會資料與簡報連結解析正確。
- [x] 2.2 實作 `data/earnings_fetcher.py` 之美股抓取器：整合 Yahoo Finance 與 Investing.com 財報行事曆，抓取 S&P 500 與科技巨頭法說會時程、EPS 預估值與 Webcast 直播網址，並驗證美股資料解析正確。
- [x] 2.3 實作台美股權值股標記（Weight Stock Tagger）與雙時區換算器（UTC / CST / EST），驗證時區轉換與夏令時適配無誤。
- [x] 2.4 在 `run_daily.sh` 中整合每日法說會排程同步任務，並驗證 CLI 指令 `python -m data.earnings_fetcher --sync` 可順利執行。

## 3. 後端 REST API 與時序端點實作 (FastAPI REST Service)

- [x] 3.1 於 `api/schemas.py` 定義法說會請求與響應 Pydantic 模型（`EarningsCallItem`, `UpcomingCallsResponse`, `CalendarSummaryResponse` 等），驗證資料驗證與 JSON 序列化。
- [x] 3.2 於 `api/routes/earnings_call.py` 實作 `GET /api/v1/earnings-call/upcoming` 端點，支援 `market` (TW/US/ALL)、`days`、`is_weight_stock` 篩選與倒數天數計算，並驗證 API 返回結構。
- [x] 3.3 於 `api/routes/earnings_call.py` 實作 `GET /api/v1/earnings-call/{ticker}` 端點，提供指定個股歷史與未來法說會清單與簡報下載連結，並驗證個股查詢正確性。
- [x] 3.4 於 `api/routes/earnings_call.py` 實作 `GET /api/v1/earnings-call/calendar` 與 `POST /api/v1/earnings-call/sync` 端點，並在 `api/main.py` 完成路由註冊與健康檢查驗證。

## 4. 前端終端介面與互動功能實作 (Frontend Terminal UI)

- [x] 4.1 在操盤看板頂部催化劑專區（Catalyst Hub）加入「台美重磅法說會預警雷達」卡片，以高對比 Badge 標示 7 天內權值股法說會倒數與市場標籤，並驗證卡片渲染與點擊互動。
- [x] 4.2 在個股詳情抽屜（Stock Drawer）中新增「法說會資訊」標籤頁，渲染該股票歷次法說會時間、官方簡報 PDF 下載入口、直播 Webcast 連結與 EPS 對比，並驗證抽屜標籤切換與資料載入。
- [x] 4.3 實作獨立法說會行事曆視圖（Earnings Call Calendar View），包含「全部 / 台股 / 美股」市場標籤切換、雙時區（台灣 CST / 美東 EST）切換、搜尋過濾與外部連結導覽，並驗證視覺佈局符合金融終端高資訊密度規格。
- [x] 4.4 在前端腳本中封裝法說會 API 調用與前端狀態管理，驗證網路異常提示與資料快取狀態同步。

## 5. 跨市場端到端驗證與文檔同步 (End-to-End Verification & Documentation)

- [x] 5.1 執行端到端資料流驗證：自 TWSE/TPEx/美股來源擷取資料、寫入 DuckDB、透過 FastAPI 查詢至前端行事曆渲染，驗證跨市場資料流完整性。
- [x] 5.2 更新系統文件，同步更新 `docs/CHANGELOG.md` 與 `README.md`，說明法說會追蹤功能、API 使用方式與排程機制。
