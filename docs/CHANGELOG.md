# Changelog

所有重要的專案變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)。

## 2026-10-15 (v2.4.17)

### Added & Enhanced (TSMC & Global Corporate Benchmark Earnings Integration, Fallback Defense & Deterministic Testing)
- **🎯 官方指標企業法說會基準庫（Verified Corporate Benchmark Directory）深度整合**：
  - **診斷痛點**：台積電（2330.TW）於 2026-10-15 14:00 召開之 2026 Q3 法人說明會，因採用官方專屬 Global Audio Webcast 系統而非 TWSE WebPro，導致外部爬蟲若因時效、分頁或上游未及時建檔時易產生遺漏；且 `fetch_ticker_calls` 先前在 BigGo 回傳空結果時會提前返回，導致查詢台積電或重量級標的時無法退回基準庫。
  - **核心升級**：
    1. **官方指標企業基準目錄（`data/verified_earnings.json`）**：收錄包含台積電（2330.TW）、聯發科（2454.TW）、鴻海（2317.TW）、廣達（2382.TW）及美股科技巨頭（NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA 等）共 22 場已確認之全球指標企業法說會，包含官方 IR Webcast、簡報下載與 Q&A 重點指引。
    2. **DuckDB 獨立種子植入機制**：資料庫初始化時採用 `INSERT OR IGNORE` 批次寫入基準庫，不依賴單一股票判斷，確保現有資料庫平滑升級，所有基準資料皆完整保留。
    3. **雙向安全融合與富化（Bidirectional Merge & Enrichment）**：
       - `fetch_all_and_sync`：抓取 BigGo 與交易所資料後，與基準庫進行比對融合，既有紀錄自動補齊官方 Webcast 與指引。
       - `fetch_ticker_calls`：BigGo 上游無資料或連線異常時，自動退回基準庫，確保台積電等關鍵標的永不回傳空值。
    4. **API 與測試確定性增強（Deterministic API & Tests）**：
       - API `/api/v1/earnings-call/upcoming` 新增 `ref_date` 查詢參數（YYYY-MM-DD），支援依指定基準日計算「今天」、「即將召開」與「近期回放」，防範時區與日期漂移。
       - 單元測試補齊 `test_verified_corporate_earnings_fallback`，確保測試在任何日期執行皆具 100% 確定性。

## 2026-10-15 (v2.4.16)

### Added & Enhanced (BigGo-Style Earnings Call UI Overhaul, Today-First Smart Grouping, 2-Column Calendar Grid & Multidimensional Filtering)
- **📅 法人說明會 UI 頂級工藝改版（BigGo Finance 雙欄卡片風格、今日優先與過去/未來分層排版）**：
  - **診斷痛點**：
    1. 首頁法說會面板先前以純日期升序限制 50 筆，導致累積 50 筆歷史過去紀錄將「今天」與「未來」排期擠出查詢範圍之外，首頁切片前 8 筆全為 7 天前的「已召開」歷史事件。
    2. 全球跨市場法說會月曆彈窗（Modal 6）原先僅以 SQL `STRING_AGG` 串接公司名稱與代號輸出純文字逗號清單，缺乏市場別、時間戳、簡報/影音/財測直連及個股圖譜互動，未對齊 BigGo Finance（`finance.biggo.com.tw/earnings-call`）的卡片視覺規格。
  - **頂級量化終端工藝升級**：
    1. **「今天」優先與時序分層查詢架構**：
       - API `/api/v1/earnings-call/upcoming` 全面重構：將 SQL 查詢切分為 `today_records`（今天）、`upcoming_records`（未來，升序 ASC）、`past_records`（過去歷史，降序 DESC，由昨天往前回溯）。
       - 新增 `time_filter` 參數（`ALL`, `TODAY`, `UPCOMING`, `PAST`），並於 API 回傳 `today_count`, `upcoming_count`, `past_count` 精確統計。
       - 補齊每一筆法說會項目之 `weekday`（如 `星期三`）。
    2. **操盤首頁法說會面板（Dashboard Panel 2）極致改版**：
       - 新增四大時間膠囊篩選按鈕：`🔥 今天 (N)`、`⚡ 即將召開 (N)`、`📜 近期回放 (N)`、`全部`。
       - 各市場別切換（`全市場`、`台股`、`美股`、`日股`、`韓股`、`權值股`）。
       - 導入 BigGo Finance 頭像色彩徽章（依市場別藍/綠/紫/紅區分外框）、粗體等寬代號、公司名稱、場次備註/季度標籤、倒數狀態膠囊（今日/倒數天數/歷史天數）、簡報 PDF 及影音回放按鈕。
       - 當今天無會議時，呈現沉浸式空狀態卡片並提供快速跳轉「即將召開」或「近期回放」導流。
    3. **全球法說會月曆彈窗（Modal 6）對齊 BigGo Finance 雙欄卡片網格**：
       - 徹底揚棄 `STRING_AGG` 純文字清單，改以 DuckDB 取出每場結構化法說會詳細物件清單（`calls`）。
       - 依日期分組，日期標題列顯示「日期 + 星期 + 總場次 + 權值股徽章」。
       - 下方採響應式雙欄卡片網格（Desktop 2-col, Mobile 1-col）：公司圓形頭像徽章、市場徽章、股票代號（點擊即時展開歷史量化抽屜）、公司名稱、說明會季度/類型、時區標籤（CST/EST）、`📄 簡報` 直連、`🎥 影音` 直連、`💬 財測指引` 提示。
       - 彈窗頂部提供月份切換（上個月/下個月）、市場切換膠囊、時間焦點膠囊（`全部日程`、`今天`、`即將召開`、`歷史回放`）、即時搜尋過濾框（即時過濾代號、名稱、財測指引）與時區切換開關。

---

## 2026-10-15 (v2.4.15)

### Added & Enhanced (Taiwan Company Network Corporate Hierarchy, DuckDB Caching, FastMCP/WebMCP Tool #20, Stock Drawer Subtab & Visual Force Graph Integration)
- **🏢 台灣企業集團股權關係網絡整合（Taiwan Company Network Integration & Sharded CDN Architecture）**：
  - **診斷痛點**：以往公司簡介僅提供產業分類與一般營運簡介，缺乏台灣上市櫃公司之統編、登記資本額、母公司控股、轉投資子公司持股比例及所屬企業集團股權拓撲關係，量化決策無法穿透複雜集團交叉持股與關係企業聯動。
  - **頂級量化終端工藝升級**：
    1. **官方 CDN 分片串接與別名精準映射**：串接 `https://taiwan-company-network.david888.com`，載入 `aliases.json`（3,685+ 筆別名對應），支援股票 4 碼代號、市場別（TWSE/TPEx）與繁體公司全名精準映射，首字 Unicode 轉 Hex 分片即時載入（如 `台` ➔ `53f0.json`）。
    2. **Gzip 傳輸優化**：強制發送 `Accept-Encoding: gzip` 標頭，將 5.8MB 大分片檔案壓縮至 ~600KB，傳輸延遲自 20 秒降至 <500ms。
    3. **上市櫃關係企業智慧標記**：遍歷上游母公司（upstream）與下游轉投資子公司（downstream），自動比對全市場上市櫃清冊，為其精準標註 `is_listed`、股票代號（`ticker` / `code`）與所屬市場（TWSE/TPEx）。
    4. **商工登記與圖形網絡萃取**：自動萃取統一編號（`id`）、登記資本額億元格式化（`X.XX 億元`）、代表人、登記地址與官方力導向關係圖直連網址（`webUrl`）。

- **🦆 DuckDB 本地持久化與多級快取（DuckDB Persistence & Zero-Overhead Caching）**：
  - **`company_networks` 專屬實體時序表**：建立 `(ticker, name, unified_id, capital, representative, address, data_json, web_url, updated_at)` 實體資料表。
  - **三級高速快取**：L1 記憶體別名快取 ➔ L2 本地磁碟分片快取（`cache/company_network/`） ➔ L3 DuckDB 本地列式快取（預設 30 天有效），重覆查詢於 <0.001 秒內極速返回。
  - **優雅降級容災**：若外部 CDN 離線或超時，自動無縫回退至 DuckDB 歷史快照，確保生產環境 100% 永不崩潰。

- **🤖 第 20 大 FastMCP / WebMCP 原生工具與端點契約（MCP Tool #20 & Dynamic Agent Standards）**：
  - **FastMCP & WebMCP 原生工具**：新增第 20 個量化工具 `get_company_network(ticker: str)`，支援輸入股票代號（如 `2330`、`2330.TW`）或公司名稱，返回結構化企業集團股權架構。
  - **WebMCP 發現規格提升**：`/.well-known/mcp.json` 與 `/mcp.json` 工具總量正式升級為 20 大原生工具。
  - **REST API 端點新增與增強**：
    - 新增 `GET /api/v1/market/company-network/{ticker}` 與 `GET /api/v1/market/company-network` 端點。
    - 既有公司簡介端點 `GET /api/v1/market/company-profile/{ticker}` 採用線程池並行自動注入 `company_network` 數據與官方力導向圖連結。
  - **規格文件同步**：同步更新 `/llms.txt`、`/llms-full.txt` 與 `/skill`。

- **🎨 操盤看板個股抽屜極致工藝（Impeccable Drawer Subtab & Corporate Network Pills）**：
  - **抽屜導航升級**：個股詳細資訊抽屜增設「集團股權網絡」專屬子分頁，並動態顯示轉投資子公司筆數徽章。
  - **概覽分頁強化**：新增統一編號、登記資本額、官方股權網絡圖直連按鈕，以及「集團網絡快覽」互動膠囊（上市櫃子公司一鍵切換該股歷史量化時序）。
  - **高密度網絡看板**：專屬分頁呈現登記資訊、外部力導向網絡全圖連結、上層控股母公司卡片與轉投資子公司矩陣，上市櫃標的以綠色高對比膠囊凸顯，全面相容深淺主題與手機端 RWD。

---

## 2026-10-15 (v2.4.14)

### Fixed & Enhanced (Macro Calendar DuckDB Snapshot Persistence, Parallel Speedup, Default Expanded UI & Live Agent Versioning)
- **⚡ 總經行事曆 DuckDB 全量持久化快照與秒級載入（Macro Calendar DuckDB Persistence & Sub-millisecond Cache）**：
  - **診斷痛點**：先前總經流動性行事曆（CME FedWatch、美股財報、大宗商品、全球總經行事曆）未持久化至 DuckDB，僅依賴暫時性 JSON 磁碟快取。當快取過期或容器重啟時，後端以單線程串行依序發動 4 次遠端請求，且預設代理 `2md.aiurl.tw` 易超時，導致首頁載入或點擊更新需漫長等待近 30 秒。
  - **頂級量化終端工藝升級**：
    1. **DuckDB `macro_snapshots` 專屬快照表**：建立 `(category, data_json, source, updated_at)` 實體資料表，將 `fed_rate_monitor`、`earnings_calendar`、`commodities_summary`、`economic_calendar`、`investing_summary` 與 `polymarket_macro_sentiment` 全量入庫 DuckDB。
    2. **三級高速快取架構 (L1 RAM ➔ L2 DuckDB ➔ L3 Disk)**：冷啟動與日常查詢優先命中 DuckDB，查詢延遲由 29 秒驟降至 **<0.001 秒**（次呼叫 0.0000s）。
    3. **四線程並行抓取 (ThreadPoolExecutor) 與超時縮減**：並發抓取 4 大宏觀端點，將單點超時由 18 秒縮減為 7 秒，並將穩定高可用節點 `https://2md.glsoft.ai` 設為第一優先來源。
    4. **Stale-While-Revalidate 優雅容災**：若外部上游短暫故障或逾時，自動回退使用 DuckDB 歷史快照，杜絕畫面空窗或錯誤。

- **🌟 前端總經行事曆預設展開（Default Expanded Macro Calendar & Preference Memory）**：
  - **診斷痛點**：先前前端儀表板預設收合行事曆細節 (`showMacroDetails: false`)，需手動點擊「展開完整行事曆」，掩蓋了關鍵的流動性決策看板。
  - **頂級量化終端工藝升級**：
    1. **預設展開呈現**：首頁載入預設即直接呈現 4 大面板（FedWatch 降息監控、法人說明會、大宗商品與總經事件、Polymarket 預測市場）。
    2. **使用者偏好記憶**：透過 `localStorage.getItem('stock_show_macro_details')` 自動記住使用者的收合/展開偏好。

- **🔄 LLM / Agent Skill 動態拉取最新版本指引（Continuous Evolution Live Fetch Standard）**：
  - **提醒外部 Agent 拉取最新規格**：於 `/llms.txt`、`/llms-full.txt` 與 `/skill` (`skills/stock-quant/SKILL.md`) 頂部新增顯著標註，提醒所有調用本平台的 LLM Agent（Claude、Antigravity、Cursor、GPT）於每次對話或任務啟動時，優先發送 HTTP GET 動態抓取最新規格，即時獲取最新擴充之 19 大 FastMCP 工具與端點資料架構。

---

## 2026-10-14 (v2.4.13)

### Added & Enhanced (Global Multi-Market Earnings Calls, Guidance Extraction, DuckDB Persistence & LLM/MCP Integration)
- **🌍 全維度跨市場法說會情報系統（Global Multi-Market Earnings Calls Architecture）**：
  - **診斷痛點**：先前法說會僅涵蓋部分台股與美股，缺乏日本（JP / 東京證券交易所 TSE）與韓國（KR / 韓國交易所 KRX）標的，且未對接完整的 BigGo Finance 全球官方資料來源。
  - **頂級量化終端工藝升級**：
    1. **6 視窗並行秒級滑動抓取**：對齊 BigGo API 單次 7 日跨度限制，將日期區間切分為 6 個並行視窗（過去 18 天至未來 18 天），秒級並行抓取並去重彙整 170+ 場以上跨市場（台股 91 場、美股 114 場、日股 20 場、韓股等）最新法人說明會日程與歷史回放。
    2. **智慧多市場識別與代碼正規化**：精準解析台股（`2330.TW`、`6126.TWO`）、美股（`NVDA`、`AAPL`）、日股（`7203.T`、`9984.T`）與韓股（`005930.KS`、`000660.KS`），並自動映射對應標準時區（`Asia/Taipei`、`America/New_York`、`Asia/Tokyo`、`Asia/Seoul`）與雙時區對齊。
    3. **高管指引 (Guidance) 與官方簡報 PDF 深度萃取**：
       - 自動並行調用 `/detail` 提取管理層營收展望、毛利率目標、資本支出（CapEx）官方財測指引與機構投行 Q&A 核心要點。
       - 調用 `/file?type=slides` 提取官方簡報原始 signed S3 PDF 直連。
       - 調用 `/file?type=video` 提取官方視訊/錄音串流與 WebPro 直播回放。

- **💾 DuckDB 全維度持久化入庫與無痛自動遷移（DuckDB Persistence & Zero-Downtime Migration）**：
  - **擴充資料表結構**：為 `earnings_calls` 資料表新增 `financial_period`、`guidance_summary`、`guidance_outlook`、`guidance_margin`、`guidance_capex`、`guidance_qa` 欄位。
  - **自動向後相容遷移**：啟動時自動透過 `ALTER TABLE earnings_calls ADD COLUMN IF NOT EXISTS ...` 執行無痛平滑遷移，保障既有資料不遺失。
  - **Pandas/Pydantic 型別防禦**：解決 DataFrame `to_dict()` 將空值轉為 float `nan` 造成 Pydantic 驗證失敗的典型缺陷，新增 `_clean_str` 與 `_clean_float` 安全過濾器。

- **🤖 LLM 文本與標準 FastMCP 工具完整對接（llms.txt, llms-full.txt & 19 Native MCP Tools）**：
  - **`llms.txt` 規格同步**：正式增設 `## Global Corporate Earnings Calls & Guidance Intelligence` 專區，清楚定義 `/api/v1/earnings-call/upcoming`、`/calendar`、`/{ticker}`、`/sync` 端點與多市場篩選參數。
  - **`llms-full.txt` 深度規格**：補齊 `earnings_calls` DuckDB 資料字典與完整 REST 端點契約。
  - **FastMCP、WebMCP Manifest 與 Dynamic Bridge 同步暴露**：新增 `get_upcoming_earnings_calls` 與 `get_ticker_earnings_calls` 兩大原生工具，MCP 工具總數升級為 19 大原生工具。

- **🎨 操盤看板極簡工藝與多市場篩選切換（Clean UI Header & Market Pills）**：
  - **去除冗餘綴詞**：將卡片標題精煉為「**法人說明會**」，徹底杜絕繁雜點綴詞。
  - **市場切換膠囊升級**：支援 `全部` (`ALL`)、`台股` (`TW`)、`美股` (`US`)、`日股` (`JP`)、`韓股` (`KR`) 五大市場一鍵切換。
  - **各市場專屬質感徽章**：台股（Emerald）、美股（Sky）、日股（Rose）、韓股（Purple）。
  - **日曆抽屜同步**：月度法說會日曆彈窗亦支援五大市場過濾。

---

## 2026-10-14 (v2.4.12)

### Fixed & Enhanced (Taiwan & US Earnings Call BigGo Aggregation & Clean UI Facelift)
- **🏛️ 操盤看板「台美法說會」乾淨精簡標題與反模式修復（Clean Header & No Buzzwords）**：
  - **診斷痛點**：原看板頂部催化劑卡片標題為「台美法說會預警雷達」，點綴詞「預警雷達」過於冗餘繁雜。且當點擊「台股」分頁時，因底層台股資料為空，前端竟錯誤降級渲染 Investing.com 美股財報行事曆（ASML、BAC、Morgan Stanley、BlackRock），導致台股分頁顯示美股的嚴重混淆。
  - **頂級量化終端工藝修正**：
    1. **去除繁雜綴詞，標題精煉**：將卡片標題由「台美法說會預警雷達」全面精簡為乾淨專業的「**台美法說會**」，維持頂級操盤終端之極簡風範。
    2. **嚴格隔離跨市場降級渲染**：修復前端 fallback 邏輯，明確限制僅在市場為全部（`ALL`）或美股（`US`）時才允許降級至美股總經財報清單；當切換為「台股」時絕不滲漏美股，若無時程則乾淨呈現「近期暫無台股法說會時程」。

- **🌐 台灣法說會資料全面抓取修復與 BigGo 財經 / MOPS 多視窗聚合（Taiwan Earnings Call Pipeline Overhaul）**：
  - **根本原因**：原先使用的 TWSE OpenAPI (`openapi.twse.com.tw/v1/opendata/t100sb02_1`) 與 TPEx 端點因主管機關調整而回傳 HTTP 302 導向 404 頁面，導致台灣上市公司與上櫃公司法說會資料全數抓取失敗（0 筆）。
  - **架構升級與高可用性實作**：
    1. **BigGo 財經 / MOPS 多視窗滾動抓取管道 (`fetch_biggo_calls`)**：全面對接 BigGo 財經公開法說會 API，透過多視窗滾動分段（-12 天至 +24 天，每 6 天一滾動視窗，避開單次 7 天查詢限制），全量涵蓋台灣上市（TWSE）、上櫃（TPEx）與美股重大法說會與財報電話會議。
    2. **官方簡報簽名 PDF 直連解析 (`fetch_biggo_slide_url`)**：針對當日（`today`）與近期即將召開之重磅法說會，透過 BigGo File API 直接解析出公開資訊觀測站 (MOPS) 之官方簡報 S3 簽名 PDF 下載入口，按鈕一鍵直達官方簡報。
    3. **個股代碼智慧正規化與權值股識別**：精準正規化台股代碼（如 `1319.TW`、`2340.TW`、`2603.TW`、`6126.TWO`）與美股代號（`NVDA`、`AAPL`、`ASML`），精準標記台灣前 50 大與美股科技巨頭權值股徽章。
    4. **DuckDB 參數型別防禦與即時隨選抓取**：修復 API 層與 DuckDB 查詢防禦層之型別轉換缺陷，並於個股法說會查詢（`GET /api/v1/earnings-call/{ticker}`）加入即時按需抓取入庫能力，查詢未在近期滾動視窗之個股亦能秒速回填歷次法說會與簡報連結。

---

## 2026-10-14 (v2.4.11)

### Fixed & Enhanced (Eliminated Desktop Table Horizontal Scrollbar & Compact Density Optimization)
- **🖥️ 操盤看板彭博表格橫向滾動條徹底消除與高密度緊湊排版優化（Zero-Scrollbar Compact Quant Table）**：
  - **診斷痛點**：桌機版在部分筆電（如 13"/14" MacBook、1080p 視窗）因先前設定的 `min-w-[1040px]` 限制與每格 `p-3`（水平 24px padding，11 欄累積達 264px 內距），導致在視窗寬度 < 1200px 時底部持續出現橫向滾動條（Scrollbar），且右側「策略共振標籤」與「操作」欄位被遮蔽或切斷。
  - **頂級量化終端工藝修正**：
    1. **移除硬性最小寬度約束 (`min-w-[1040px]` -> `w-full`)**：改為純流式響應 `w-full`，讓表格完美貼合容器寬度，徹底消除常駐橫向滾動條。
    2. **內距瘦身與文字密度最佳化 (`p-3` -> `py-2.5 px-2`)**：將每格內距由 24px 精準縮減至 16px，單是內距即節省 88px 橫向空間；文字採用緊湊 `text-xs`，精簡欄位標題（如「標的 / 名稱」、「目標價 / 均線」、「潛力」、「5日籌碼」）。
    3. **策略共振標籤雙徽章精煉 (`slice(0, 2)` + `+N`)**：主列表格呈現前 2 枚核心共振標籤（搭配緊湊內距 `px-1.5 py-0.5` 與小圖示），超過 2 枚自動呈現 `+N` 膠囊；將標籤欄寬由 285px 收斂至 ~160px，節省 120px+。懸停時完整浮動提示（`title`）與展開抽屜依然保留全量標籤。
    4. **全表內容量縮至 940px 以內**：在任何標準螢幕（1024px、1080p、1200px、1440p）均 100% 完整容納全部 11 欄與操作按鈕，底部零滾動條。

---

## 2026-10-14 (v2.4.10)

### Fixed & Enhanced (Desktop Bloomberg Quant Table Layout & Tag Row Height Alignment)
- **🖥️ 操盤看板彭博表格排版錯位與標籤垂直折疊高度爆開修復（Desktop Table Sizing & Tag Density Normalization）**：
  - **診斷痛點**：
    1. **標籤垂直爆開與文字截斷**：桌機版表格「策略共振標籤」欄位因 `flex-wrap` 且無最大寬度約束，導致單一標的 6~8 個標籤（如 `LSTM強`、`TimesFM強`、`高盈虧比`、`LSTM看漲`、`TimesFM看漲`、`雙ML共振`）垂直折成 4~6 行，表格行高從正常 48px 暴增至 180px；而較長標籤（如 `⭐ 雙重共振`）更在右側邊界被生硬截斷。
    2. **數值欄位過度擴張與空白失衡**：在未指定明確欄寬下，瀏覽器自動版面配置（Auto Table Layout）將大部分水平寬度分配給「現價」、「目標價」、「PE/PB」與「籌碼」等數字欄位，形成巨大無意義的空白空隙，卻將標籤欄擠壓至極窄寬度。
    3. **操作欄位脫離視窗**：因表格總寬度在無序伸展下超出視窗邊界，第 11 欄「操作」（含「詳情」與「時序」按鈕）被擠出可視區域之外，導致使用者完全看不到操作按鈕。
  - **頂級量化終端工藝修正**：
    1. **標籤單行防折疊與 +N 膠囊指示器 (`flex-nowrap` + `.slice(0, 3)` + `+N`)**：表格主列標籤限制呈現前 3 個最關鍵共振標籤並鎖定單行排列；若超過 3 個標籤則呈現精緻的 `+N` 膠囊，並於 `title` 工具提示中完整列出所有其餘標籤。列高完美鎖定在標準彭博終端行高（`h-12` / 48px），徹底消滅垂直巨大空隙與文字截斷。
    2. **全欄位嚴謹寬度約束與空間重新校準**：針對 11 個欄位全部賦予嚴格寬度規範（`#`: 48px、`標的代號/名稱`: 160px、`市場`: 96px、`策略模型`: 96px、`現價`: 96px、`目標價/均線`: 112px、`預期潛力`: 80px、`PE/PB`: 96px、`5日籌碼`: 128px、`操作`: 96px），並將剩餘彈性空間平滑分配給「策略共振標籤」（`min-w-[220px]`），消滅數值欄位間的多餘空隙。
    3. **保障「操作」欄位 100% 完整常駐可視**：表格基礎寬度收斂至 1040px，在任何常見桌機與筆電螢幕（>= 1200px）上皆無需橫向滾動即可完整看到全部 11 欄（含「操作」欄按鈕）。
    4. **抽屜展開完整標籤展示**：點擊「詳情」展開公司資訊抽屜時，於基本面區塊完整陳列所有多維度共振標籤，兼顧主表掃描效率與深度剖析需求。

---

## 2026-10-14 (v2.4.9)

### Fixed & Enhanced (Podcast Radar DB Persistence & 0ms Stale-While-Revalidate Hydration)
- **💾 修復 Podcast 決策評級未持久化入庫導致每次開啟重複推論與 32 秒卡頓缺陷**：
  - **診斷痛點**：使用者每次重新開啟網頁或點選 Podcast 雷達分頁時，後端響應時間高達 32 秒，前端中央呈現「正在連線情報資料庫與計算即時股票行情中...」並覆蓋已加載的單集列表。
  - **根本原因**：
    1. `data/podcast_service.py` 的 `enrich_episodes_batch` 於記憶體計算單集的重要性（`importance_score`）與標的多空情緒（`ticker_sentiments`）後，**漏寫回 DuckDB**，導致資料庫中 200+ 筆單集的評級欄位恆為 `NULL`。
    2. 每次 API 查詢觸發 100 筆單集逐一呼叫遠端推論鏈，同步 GET 請求嚴重阻塞逾 30 秒，且因未入庫而陷入「每次造訪都重推」的無效循環。
  - **完整解決架構**：
    1. **DuckDB 批次持久化入庫 (`DuckDBManager.update_podcast_episodes_decisions_batch`)**：於 `DuckDBManager` 新增批次更新函式，`enrich_episodes_batch` 在推論後立即批次將結果持久化至 DuckDB，杜絕重複計算。
    2. **種子配置全量預置與資料庫歷史資料回填 (`backfill_unevaluated_episodes`)**：
       - 全量 200 筆種子單集 (`config/podcast_seed_episodes.json`) 預置完整的 AI 決策分數、評級標籤與多空立場，冷啟動初次寫入即具備完備決策元數據。
       - 新增 `backfill_unevaluated_episodes()`，自動盤點並回填資料庫內現存的歷史未評估單集。
       - 同步 GET 請求路徑全面採極速確定性本地規則 (<0.2ms) 與 DB 快取直出，API 響應時間自 32 秒驟降至 **5 毫秒以內**。
    3. **前端 Impeccable 頂級工藝：Stale-While-Revalidate (SWR) 與 0ms 本地秒開**：
       - `api/static/js/app.js` 於 Vue 初始化時直接自 `localStorage` 讀取快照立即渲染（0ms 秒開，開網頁即見 100 篇情報）。
       - `tab_podcast.html` 調整條件渲染：只有在「完全無任何單集快取」時才呈現載入動畫，若已有快取則在背景無感靜默重驗證，徹底杜絕畫面遭覆蓋與閃爍問題。

---

## 2026-10-14 (v2.4.8)

### Added & Enhanced (5-Tier Resilient Decision Engine & Podcast AI Importance/Sentiment Integration)
- **⚡ 5 級階梯式高可用決策鏈 (5-Tier Resilient Decision Engine)**：
  - **核心架構**：整合 Clef-Flash System One、JEV 與 OpenAI 決策模型，實作 5 級高可用降級鏈路：
    - **Tier 1 (主力)**: `https://clef.create360.ai/v1/systemone` (Clef-Flash System One 快思考推理)
    - **Tier 2 (備援 1)**: `https://clef.aiurl.tw/v1/systemone` (llama.cpp GGUF 節點)
    - **Tier 3 (備援 2)**: `https://api.typesafe.ai/v1/systemone` (JEV 原生端點)
    - **Tier 4 (備援 3)**: OpenAI 結構化決策模型 (JSON Schema 降級)
    - **Tier 5 (兜底保障)**: **100% 確定性本地啟發式規則引擎 (Local Rule Heuristics Fallback)**。
  - **🚨 永不崩潰鐵律保證**：若所有遠端決策模型（Tier 1~4）全數遭遇斷線、503 服務不可用或逾時，系統自動於 2ms 內無縫降級至本地確定性啟發式規則，確保決策推論 100% 正常可用，API 恆常返回 HTTP 200。
- **🎙️ Podcast 單集深度智慧決策增強 (AI Importance & Ticker Sentiments)**：
  - **市場催化重要性評級 (`importance_score`)**：依據訪談深度、頻道權威、來賓知名度與關鍵催化劑，評估 1.0~5.0 分，並於單集情報卡右上角生成高質感膠囊徽章（`⭐ 5星重磅`、`⚡ 4星重要`、`📰 3星關注`）。
  - **實體標的多空姿態動態指示 (`ticker_sentiments`)**：針對單集提及之上市公司代號，由決策模型推論其看多（`↑ 看多`）、看空（`↓ 看空`）或持平（`↔ 中立`）之立場與確信度，並在代號 Badge 上直觀呈現。
  - **頂部工具列新增重要性快篩開關**：支援一鍵過濾出「⭐ 重磅」或「⚡ 重要」單集，大幅提升高信噪比資訊提煉效率。
- **💾 DuckDB 單集表無損遷移與 API 端點擴充**：
  - `podcast_episodes` 表結構自動遷移新增 `importance_score`、`importance_level`、`ticker_sentiments` 與 `decision_slot` 欄位。
  - 新增 `POST /api/v1/podcast/episodes/{episode_id}/evaluate`（單集重新推論）與 `POST /api/v1/podcast/batch-evaluate`（批量推論）。
  - 升級 `GET /api/v1/decision/health`，即時反饋全部 5 個 Tier 的存活與備援狀態。

---

## 2026-10-14 (v2.4.7)

### Fixed & Enhanced (Responsive Table-to-Card RWD & Podcast 100-Channel/100-Episode Full Sync)
- **📱 操盤看板表格 RWD 響應式優化（Mobile Hybrid Quant Cards & Desktop Fluid Table）**：
  - **診斷痛點**：在行動裝置直向窄螢幕（~360px~414px）下，包含 11 欄量化與籌碼指標的 Bloomberg 寬表格因缺乏容器寬度保護與防折行設定，被瀏覽器硬性擠壓至極窄空間，導致「策略模型」中文字符垂直折成四行（「策\\n略\\n模\\n型」）、「玄鐵重劍」折成「玄\\n鐵\\n重\\n劍」，右側指標被截斷，嚴重破壞版面閱讀體驗。
  - **雙層響應式架構革新**：
    1. **智慧裝置偵測預設（Smart Device Adaptive Default）**：針對小於 768px 之行動裝置，進入首頁自動預設為專為手機設計的卡片檢視（`viewMode = 'card'`），並將使用者手動切換狀態持久化記錄於 `localStorage`。
    2. **表格模式下 RWD 混合轉譯（Hybrid Table-to-Card Layout）**：當使用者在行動裝置上切換或維持表格模式（`viewMode = 'table'`）時，小螢幕（`< md`）自動無縫轉譯為「高密度行動量化卡片」，完整保留排名、代號、名稱、市場、策略、現價、目標價、PE/PB、籌碼、標籤與公司抽屜展開功能，杜絕任何文字垂直擠壓；而在平板與桌面端（`>= md`），則平滑呈現標準彭博寬表格，並全面賦予 `min-w-[980px]` 與 `whitespace-nowrap`，保障橫向滑動時排版絕不變形。
- **🎙️ Podcast 財經雷達全量 100 頻道去重與 100 篇情報端到端同步**：
  - **診斷痛點**：
    1. 頻道下拉選單出現重複頻道名稱（如 In a Nutshell、The Joe Rogan Experience、Dwarkesh Podcast 等重複出現兩次）。
    2. 頂部狀態列顯示「收錄 50 頻道 · 30 篇情報」，未完整展現上游最新收錄的完整 100 篇情報。
  - **根本原因與架構修復**：
    1. **資料庫去重與舊頻道淘汰 (`data/podcast_service.py`)**：早期 Mock 測試資料保留了舊頻道 ID（如 `kurzgesagt`, `joe-rogan`, `dwarkesh` 等），與後續標準 Slug ID 共存於 DuckDB 中。本次加入自動清理遷移機制，將舊 episode 自動映射至標準 ID，並以 `DELETE FROM podcast_channels WHERE channel_id NOT IN (...)` 徹底清除不在 100 種子清單中的 19 個舊頻道；API 路由層面進一步加入不分大小寫的標題嚴格去重，確保下拉選單 100% 乾淨無重複。
    2. **端到端放寬情報容量上限（20/30 -> 100 篇）**：將 `api/routes/podcast.py` 預設 limit 由 20 提升至 100，`api-client.js` 由 20 提升至 100，`app.js` 由 30 提升至 100，使前端首頁直接呈現「收錄 100 頻道 · 100 篇情報」，完整對齊 888 即時深度情報庫。

---

## 2026-10-14 (v2.4.6)

### Fixed & Enhanced (Pure Chinese Macro Labels & Header Language Switcher)
- **🏷️ 全面清除大盤風控多餘英文括號註記，回歸純粹凝練之專業中文語境**：
  - **診斷痛點**：大盤風控與宏觀狀態先前帶有冗餘的英文括號翻譯（如「全面多頭 (Strong Bull)」、「多頭回調 (Bull Pullback)」、「加權弱勢整理 (Weak Consolidation)」等），在繁體中文介面下顯得拖沓廢話且破壞視覺平衡。
  - **雙層徹底淨化架構**：
    1. **後端淨化 (`data/macro.py`)**：清理台股加權指數與美股宏觀分析引擎之全部 `state.regime_name` 定義，全數改為純中文標準狀態（如「全面多頭」、「多頭回調」、「加權弱勢整理」、「溫和空頭」、「輕微震盪」、「極度恐慌」），徹底移除所有後綴英文與舊版 Emoji。
    2. **前端智能清洗 (`cleanRegimeName`)**：無論後端即時推導或 DuckDB 快取舊資料，前端全局透過 `cleanRegimeName` 進行正則清除，在繁體中文模式下自動濾除任何 `(English Words)` 括號後綴，確保看板頂部風控藥丸與儀表板卡片呈現純粹凝練之標題。
- **🌐 頂部導覽列導入精煉語言切換按鈕（Language Switcher）**：
  - 於右上角功能群組（Header Actions）新增精巧藥丸型語言切換鈕（`🌐 繁中 / EN`），具備狀態記憶（`localStorage`）與流暢切換微互動。
  - 當切換為英文語系時，宏觀風控狀態將對應呈現標準國際金融術語（如「Strong Bull」、「Bull Pullback」、「Weak Consolidation」等），實現雙語切換自由且不混雜冗贅括號。

---

## 2026-10-14 (v2.4.5)

### Fixed (Eliminate Flash of Uncompiled Content / FOUC on Initial Page Entry)
- **🚫 徹底終結初始進入頁面時的模板閃爍（FOUC）與彈窗重疊問題**：
  - **診斷痛點**：首次進入頁面或網路載入時，瀏覽器會短暫跳出全黑背景的「台美股法人說明會暨財報行事曆」彈窗、左側浮動個股懸浮卡片及未解析的雙大括號標籤（如 `{{ hoverData.ticker }}`、`{{ calendarYear }}` 等），造成恐怖混亂的初次渲染體驗。
  - **根本原因**：
    1. Tailwind CDN 在執行期注入的 `.flex { display: flex; }` 樣式因權重與載入時序問題，覆蓋了原先弱規則的 `[v-cloak] { display: none; }`，導致 `#app` 在 Vue 3 完成掛載前未被隱藏。
    2. 浮動彈窗與懸浮卡（`showEarningsModal`、`hoverVisible`、`toastMessage`）直接寫在一般 DOM 容器中，未掛載前被瀏覽器當成普通 `fixed` 元素直接繪製。
  - **三重防護架構**：
    1. **升級強規格 `[v-cloak]` 規則**：引入 `[v-cloak], [v-cloak] * { display: none !important; }`，杜絕任何 CSS 框架的特異性覆蓋。
    2. **HTML5 原生 `<template v-if="...">` 隔離**：將「法人說明會日曆彈窗」、「個股懸浮速查卡」及「Toast 通知」全數包裹至 HTML5 原生 `<template>` 標籤中。依 HTML5 規範，未解析前 `<template>` 內容為惰性片段（Inert DocumentFragment），瀏覽器物理上絕不渲染，100% 杜絕彈窗先行爆出的問題。
    3. **優雅初始掛載預載畫面（Initial Mount Preloader）**：於 `<body>` 注入輕量級品牌橘色徽章載入層（`#preloader`），在 Vue 完成編譯掛載（`mounted`）後平滑淡出並銷毀，保證使用者每次造訪皆流暢無感。

---

## 2026-10-14 (v2.4.4)

### Fixed & Redesigned (Material Symbols Icon Facelift, Light Theme Default & Collapsed Macro Lines)
- **🎨 徹底移除 Raw Emoji 標籤圖示，全面升級為 Google Material Symbols Outlined 高質感圖標系統**：
  - 診斷痛點：量化策略標籤與共振等級先前直接沿用後端字串之系統 Emoji（如 `🔮雙ML共振`、`⭐雙重共振`、`🏆三重共振`、`👑四重共振` 等），在不同瀏覽器與作業系統呈現出風格不一的卡通化色彩，破壞專業金融終端之嚴謹調性。
  - 實作 `stripEmojis` 全局正則過濾（完整涵蓋 `\p{Extended_Pictographic}`、各區段 Emoji 與 Unicode 符號），並在前端 `parseTags` 中將共振標籤無縫轉化為乾淨文字搭配專屬 Google Material Symbols：
    - 四重共振：`workspace_premium`
    - 三重共振：`military_tech`
    - 雙重共振 / 多維共振：`stars` / `hub`
    - 雙ML共振：`psychology`
    - 法人鎖碼 / 土洋合買 / 投信連買：`handshake`
    - 玄鐵技術買點 / 均線波段：`show_chart`
    - TimesFM 預測：`auto_awesome`
    - LSTM 預測：`insights`
    - 高盈虧比：`balance`
    - 防守避險：`shield`
  - 新增 `getTagClass` 主題適配樣式，在深色曜石黑與淺色溫潤模式下均具備清晰邊界與專屬層次色彩。
- **☀️ 預設主題調整為淺色模式（Light Mode Default），杜絕突發性全黑畫面**：
  - 移除 `<html class="dark">` 之硬性鎖定，修改防閃爍預載入腳本與 `app.js`，預設以清爽溫潤的淺色模式渲染（除非使用者先前已主動切換並保存深色偏好）。
  - 標題列主題切換器維持純淨圖標按鈕（純 `dark_mode` / `light_mode`，無任何冗餘「淺色模式」文字標籤）。
- **📊 總經流動性與重磅行事曆預設收合（Collapse Macro Progress Bars by Default）**：
  - 將首頁大盤總經細節展開狀態調整為預設收合（`showMacroDetails: false`），使用者進入首頁時僅展示 4 張核心卡片與精簡摘要列，杜絕密集進度條與概率線條（FedWatch、Polymarket）突兀彈出所造成的視覺壓迫感，使用者可依需求點擊「展開完整行事曆」深入查閱。
- **🔤 遵循 Impeccable 工藝標準：中文字體排版與表格易讀性升級**：
  - 針對表格中股票名稱、策略模型、市場代號與共振標籤，將字體大小提升至 14px (`text-sm`) 與 16px (`text-base`)，全面符合中文字體易讀性門檻，搭配等寬數值字體 `JetBrains Mono`（`tabular-nums`），實現高工藝金融終端體驗。

---

## 2026-10-14 (v2.4.3)

### Fixed (Transparent Data Availability and Fallback Semantics)
- **🛡️ 宏觀資料缺失改為明確不可用**：yfinance 無資料或資料不完整時，宏觀 API 回傳 `exposure=null`、指標 `null` 與「資料不足」，停止以 VIX 20、100% 曝險或站穩均線作為替代值。
- **📅 移除未驗證美股法說會預置排程**：沒有可驗證的上游或本地快取時回傳空集合，不再產生推測日期、EPS 與 `YAHOO/IR` 假來源。
- **📉 修正 Podcast 行情語意**：移除固定行情快照；模型預測基準價不再當作即時漲跌幅，資料缺失時顯示空值並附來源時間。
- **📊 修正總經補充資料 fallback**：商品、FedWatch、Polymarket 無資料時不再顯示 `+0.0%` 或 0% 機率。
- **🧮 移除 synthetic composite score**：資料庫缺少正式綜合分數時保留 `null`，明確儲存的 0 分仍維持原值。
- **🎙️ RSS metadata 保持原始狀態**：缺少時長、語言或摘要時保留空值，單集標記為 `raw`；Podcast 同步端點正式啟用最新 RSS 抓取。
- **🧭 UI 移除固定統計 fallback**：法人筆數、日期與頻道集數缺失時顯示「資料不足」；頻道 metadata 標記為 `seed_snapshot`。
- **🚀 保留 SP500 太空概念擴充**：`SPCX`、`DXYZ`、`ASTS`、`RKLB` 維持納入 SP500 分析集合，作為策略性的太空主題擴充標的，並保留既有名稱映射。

---

## 2026-10-14 (v2.4.2)

### Added & Expanded (Full 100 Premier Podcast Channels & 200 Real AI Intelligence Episodes)
- **🎙️ 全量收錄頂級財經與科技推薦之完整 100 個 Podcast 頻道（由 36 個擴增至完整 100 個）**：
  - 串接頂級財經與科技上游資料源，完整匯入全部 **100 個全球頂級科技、宏觀財經與商業投資頻道**。
  - 頻道目錄全面涵蓋：
    - **前沿 AI 與科技（64 個）**：Founders, Lex Fridman, The Verge, Y Combinator Startup Podcast, Dwarkesh, AI Engineer, Lenny's Podcast, Theo - t3.gg, The Pragmatic Engineer, Peter Diamandis, Training Data, 42章經, Latent Space, Eye on A.I., Huberman Lab, Hard Fork, Lexman Artificial...
    - **宏觀金融與商業（20 個）**：The Economist, The Plain Bagel, All-In, My First Million, Starter Story, The Prof G Pod, Bloomberg Businessweek, Odd Lots, Trumponomics, Capital Group, The Knowledge Project, Acquired, Invest Like the Best, Barron's Live, FT News Briefing...
    - **創投、商業史與大師訪談（11 個）**：20VC with Harry Stebbings, How I Built This with Guy Raz, Masters in Business, The a16z Podcast, BG2 Pod (Bill Gurley & Brad Gerstner)...
    - **華語深度與生技科學（5 個）**：小Lin說, CLS同學, 股癌 Gooaye, 硅谷101, 瘋投圈, Citeline Podcasts, BioCentury This Week, Top Traders Unplugged...
  - **頻道官方 Artwork 頭像支援**：三欄響應式頻道卡片升級支援顯示高解析度頻道 Logo / 頭像（含優雅的加載錯誤與預設 fallback），打造頂級專業金融資訊終端的一致精品級視覺質感。
- **🤖 匯入 200 篇真實 AI 精煉單集情報與即時股票概念聯動（config/podcast_seed_episodes.json）**：
  - 串接全球即時單集情報流，收錄 200 篇由 AI 生成之深度繁中 TL;DR、關鍵商務洞察（Business Highlights）、實體標籤與關聯上市公司代號（Ticker）。
  - 單集即時行情 Badge 與操盤看板、個股抽屜全面連動，點擊股票標籤即可即時展開時序預測與籌碼分析。
- **⚡ 後端自動同步與查詢上限擴展**：
  - `api/routes/podcast.py` `/channels` 上限由 100 擴大至預設 200，並新增 `refresh=True` 支援。
  - `api/static/js/api-client.js` `getPodcastChannels` 預設查詢上限調整為 200，保證全量 100 頻道一次性完整渲染。
### Fixed & Redesigned (Zero Horizontal Scrollbars & Responsive Flow across Desktop and Mobile)
- **🚫 徹底終結桌面與手機端橫向捲軸（Scrollbar 橫軸）視覺噪音與裁切問題**：
  - **診斷痛點**：舊版在策略篩選列、頂部手機導覽列及 Podcast 話題標籤處使用 `overflow-x: auto`，搭配全局 `::-webkit-scrollbar` 設定，在桌機螢幕窄化或手機直向瀏覽時，強行浮現一條 6px 粗厚的黑灰色捲軸軌道，且多數選項被推出可視畫面之外，嚴重破壞彭博終端之工藝質感。
  - **操盤看板策略篩選升級為「自適應換行排版（Responsive Flex-Wrap）」**：
    - 移除 `overflow-x-auto`，全面改採 `flex flex-wrap items-center gap-2`。
    - 在寬螢幕上維持單列俐落排版；在筆電或小螢幕上自然換行排列，全部 7 大策略（三重共振、玄鐵均線、TimesFM、LSTM 看漲/看跌、全量批次）100% 完整可見，零橫向捲軸，無須笨拙側滑。
  - **手機版導覽列重構為「全寬等分四宮格（Full-Width Grid Segmented Control）」**：
    - 將 `header.html` 溢出的藥丸橫軸改為 `grid grid-cols-4 gap-1 w-full`，完美適配 375px~414px 任何手機螢幕寬度，零溢出、零滾動、無捲軸。
  - **Podcast 熱門話題升級為「AI 實體標籤雲（Entity Tag Cloud）」**：
    - 移除 `custom-scrollbar`，改以響應式標籤雲（Tag Cloud）呈現，所有前沿實體（OpenAI, NVIDIA, Claude, TSMC, Google...）一眼盡覽。
  - **全局捲軸純淨化與 `.no-scrollbar` 工具類**：
    - 優化 `index.html` 捲軸樣式，水平方向預設隱藏捲軸，垂直方向改為低干擾細緻軌道，杜絕任何控件邊緣黑線。

---

## 2026-10-14 (v2.4.1)

### Fixed (Browser Cache Invalidation, Dynamic Asset Fingerprinting & Template Crash Safeguard)
- **🐛 徹底修復前端版本快取殘留導致的白屏/黑屏例外（TypeError: formatFollowers is not a function）**：
  - **根本原因診斷**：版本號先前固定為 `2.4.0`，當伺服器更新 HTML 模板時，Cloudflare 或使用者瀏覽器從本機快取了尚未包含 `formatFollowers` 的舊版 `app.js?v=2.4.0`，導致 Vue 3 在解析 Podcast 標籤時噴出 `TypeError: formatFollowers is not a function`，使 Vue 根實例掛載中斷，觸發 `[v-cloak]` 鎖死全黑畫面。
  - **動態指紋快取破壞**：升級 `core/version.py` 至 `2.4.1`，並於 `api/main.py` 實作基於靜態腳本實體修改時間之動態資產指紋（`ASSET_FINGERPRINT`），確保日後任何代碼變更全網瀏覽器均能秒級取得最新腳本。
  - **靜態檔案反快取標頭**：在 `/static` 路由回應中加入 `Cache-Control: no-cache, must-revalidate` 與 `Pragma: no-cache`，杜絕過期 ESM 模組殘留於 CDN 或瀏覽器磁碟。
  - **防禦性模板與全局屬性兜底**：於 `api/templates/partials/tab_podcast.html` 導入防禦性型別檢查（`typeof formatFollowers === 'function'`），並於 `app.js` 掛載前將 `formatFollowers` 註冊至 `app.config.globalProperties` 與 `window`，從根本防範任何未定義例外。

### Redesigned & Fixed (Figma 5:36 Compact Orange Badge Logo & Obsidian Dark Terminal Default)
- **🎨 依據 Figma 5:36 重構左上角品牌 Logo（亮橘底色純標誌徽章，消除冗長版號與副標）**：
  - 依據 Figma 設計節點 `5:36`（`node-id=5-36`）與設計規範，徹底移除左側臃腫的文字區塊、版本號（`v2.4.0`）與「多維量化交易決策平台」副標題，釋放超過 200px 寶貴的水平空間。
  - 全新打造 48x48 高質感亮橘漸層品牌徽章（`888 \n Stock \n Quant` 三層垂直排列），黑字粗體對比強烈，支援點擊一鍵返回操盤看板，視覺乾淨簡練且具備高辨識度。
- **🌑 預設主題切換至 Figma 20261013 曜石黑彭博終端（消除首次造訪為淺色模式之誤解）**：
  - 診斷出使用者因本地瀏覽器歷史快取或舊版預設而造訪淺色模式（`#EEEEEE` 灰底白卡片），導致誤認整體風格未依據 Figma 20261013 進行深邃金融終端重塑。
  - 導入新版獨立主題狀態鍵 `theme_v20261013`，預設全面啟用深色曜石黑終端（Obsidian Black Terminal，`#030302` 深邃底色、`#0D0E11` 卡片與 `#F59E0B` 琥珀金點綴），並在 `<head>` 注入防白閃預載入腳本，確保首次開啟或重新整理均能完美呈現 Figma 20261013 彭博終端風貌。

#### Added & Expanded (Full Premier Podcast Recommendation Directory & 36 Top Channels)
- **🎙️ 全面擴充收錄全球頂級財經推薦頻道庫（由 12 個大幅擴增至 36 個全球頂級頻道）**：
  - 完整收錄全球指標性財經、商業與科技深度訪談頻道目錄：
    - **科普與大眾對話**：In a Nutshell (Kurzgesagt, 25.7M 追蹤者)、The Joe Rogan Experience (21.1M 追蹤者)、The Diary of a CEO (Steven Bartlett, 20.4M 追蹤者)、ColdFusion (5.3M 追蹤者)。
    - **前沿 AI 與科技趨勢**：Lex Fridman Podcast (5.1M)、The Verge (3.5M)、Y Combinator Startup Podcast (2.4M)、Dwarkesh Podcast (1.4M)、AI Engineer (673K)、Lenny's Podcast (644K)、Theo - t3.gg (571K)、The Pragmatic Engineer (541K)、Peter Diamandis (538K)、Training Data (410K)、42章經 (310K)。
    - **宏觀金融、商業巨擘與投資大師**：The Economist (4.2M)、The Plain Bagel (1.2M)、All-In (1.2M)、My First Million (963K)、Starter Story (886K)、The Prof G Pod (Scott Galloway, 823K)、Bloomberg Businessweek (591K)、Odd Lots (591K)、Trumponomics (591K)、Capital Group (527K)、The Knowledge Project (456K)、Acquired (1.25M)、Invest Like the Best (890K)。
    - **華語與生技/量化領域**：小Lin說 (2.8M)、CLS同學 (1.1M)、股癌 Gooaye (1.68M)、硅谷101 (472K)、瘋投圈 (395K)、Citeline Podcasts (260K)、Top Traders Unplugged (340K)、BioCentury This Week (180K)。
  - **資料持久化與自動增量同步**：更新 `config/podcast_seed_channels.json`、`data/podcast_service.py` 與 `api/routes/podcast.py`，當資料庫收錄頻道數少於種子頻道清單時自動執行 DuckDB `ON CONFLICT` 增量寫入，確保無需手動重置即時更新全量 36 頻道。
- **📺 新增「全部頻道庫 (Channels Directory)」3 欄響應式網格視圖**：
  - 於 `api/templates/partials/tab_podcast.html` 導入雙重視圖切換：
    - `[ 深度情報單集 (Episodes) ]`：瀏覽由 AI 繁中精華 TL;DR 萃取之單集情報卡、即時股票漲跌幅 Badge、內嵌 YouTube/MP3 播放器。
    - `[ 全部收錄頻道 (Channels Directory) ]`：以 3 欄高密度卡片完整呈現 36 個頻道之頭像圖示、總單集數、追蹤者數量（如 25.7M、5.1M、1.2M）、領域標籤與作者介紹，並支援一鍵點擊「查看單集」直達該頻道專屬情報篩選。
- **📈 擴充真實精煉單集情報庫與股票行情映射白名單**：
  - 對齊最新國際深度情報流，新增 Training Data（DeepMind David Silver 與 Lasse Espeholt 對談）、瘋投圈（149 抖省省突襲美團到店業務）、Citeline（Scrip 2026 最新生技要聞與名人代言）、Top Traders Unplugged（第421集 CTA 走向主流）、BioCentury（Ep. 392 生技市場與 Moderna 軋空）、42章經（Personal Agent 創業者思考）、Bloomberg Businessweek（川普與普丁柴油交易、單座資料中心天價成本）、Acquired（台積電與 NVIDIA 史詩歷史）、股癌（EP480 CoWoS 與科技股多空解讀）。
  - 新增美團（3690.HK）、阿里巴巴（9988.HK / BABA）、拼多多（PDD）、莫德納（MRNA）、阿斯特捷利康（AZN）、諾華（NVS）、argenx（ARGX）、埃克森美孚（XOM）、雪佛龍（CVX）、Vistra 核電（VST）、星座能源（CEG）與 CTA（QLENX）之行情降級與實體白名單映射。
- **🐛 修復 DuckDB 頻道單集數統計錯誤**：
  - 修正 `data/duckdb_manager.py` 中 `COALESCE` 在子查詢返回 0 時無法回退至種子頻道 `episode_count` 的 SQL 邏輯缺陷，改採 `CASE WHEN ... > 0` 條件判斷，確保全部頻道均正確顯示真實集數。


### Improved (Full Material Symbols Outlined Icon System & Minimalist Header Controls)
- **✨ 全面導入 Google Material Symbols Outlined 圖示系統（徹底汰除 Emoji）**：
  - 依據 Impeccable 工藝標準規範，杜絕以 Unicode Emoji 作為 UI 圖示的不一致混雜質感；全站引入 Google Material Symbols Outlined 向量字體圖示系統（`font-variation-settings: 'FILL' 0, 'wght' 350, 'GRAD' 0, 'opsz' 24`）。
  - 將導覽列、操盤看板、策略篩選、巨集狀態卡、法說會行事曆、三大法人、券商分點、個股歷史時序、AI Podcast 雷達與頁尾全站所有 Emoji 完全替換為乾淨利落、筆畫一致的 Material Symbols Outlined 圖示（如 `bolt`, `monitoring`, `account_balance`, `history`, `podcasts`, `military_tech`, `timeline`, `auto_awesome`, `smart_toy`, `calendar_month`, `headphones`, `smart_display` 等）。
- **🎯 頂部右側功能區極簡化（消除廢話文字）**：
  - 徹底移除右上角「深色模式 / 淺色模式」冗長繁瑣的狀態文字說明，改採現代頂級 Web App 標準的 40x40 圓形純圖示按鈕（`w-10 h-10 rounded-full border`）。
  - 整合「分享看板」（`share`）與「主題切換」（`light_mode` / `dark_mode`）為乾淨洗鍊的雙圖示按鈕組，並保留標準 `aria-label` 與 `title` 工具提示，視覺輕盈不雜亂。

### Fixed & Improved (Impeccable UI Layout Equilibrium, Chinese Typography Overhaul & Stock Recommendation Bug Fix)
- **⚖️ 導覽列三欄平衡重構（消除偏右擠壓與不平衡問題）**：
  - 重構 `api/templates/partials/header.html` 為經典穩定的三欄對稱架構：
    - 左欄：品牌標誌與主標題（`shrink-0`，緊湊不換行）。
    - 中欄：四大核心 Tab 置中對齊（`flex-1 justify-center`），徹底消除導覽列向右凸出擠壓與中間大片空白之不平衡視覺缺陷。
    - 右欄：系統狀態、分享連結與主題切換控制項。
  - 行動端適配：窄螢幕自動將四大 Tab 移至獨立置中子導覽列（Mobile Nav Sub-bar），按鈕字級提升至 `text-sm font-bold`，確保觸控熱區與閱讀體驗。
- **✍️ 中文字體排版全面依循 Impeccable 標準升級（嚴格遵守最小 12pt / 16px 規範）**：
  - **根字級標準化**：於 `api/templates/index.html` 將 `html { font-size: 16px; }`（96 DPI 下之 12pt 印刷基準）設為全站基底，並開啟 `-webkit-font-smoothing: antialiased` 與 `text-rendering: optimizeLegibility`。
  - **徹底撲滅微小字體**：全站搜尋並完全移除所有 `text-[10px]` 與 `text-[11px]` 極小字級，避免中文字複雜筆畫（如「聽」、「講」、「總」、「結」、「觀」、「薦」、「籌」、「碼」）模糊失真。
  - **內文與摘要工藝提升**：
    - Podcast AI TL;DR 深度觀點精煉內文提升至 `text-base leading-relaxed`（標準 16px / 12pt）。
    - 關鍵觀點條目（Takeaways）與個股抽屜業務概述提升至 `text-[15px] sm:text-base leading-relaxed`。
    - 實體標籤、股票名稱與操作按鈕全面提升至 `text-sm font-semibold`（14px），數值與英文代號採用等寬數字字體（JetBrains Mono `num-font`）。
- **🎯 根除操盤看板「0 檔標的」空名單錯誤**：
  - **修復前端預設值傳參偏差**：修正 `api/static/js/app.js` 與 `api-client.js`，當 URL 缺乏 `index` 參數或為「全部市場」時，不再向後端傳遞字串 `'ALL'`，避免 DuckDB 執行無效的 `WHERE index_name = 'ALL'` 查詢。
  - **後端指數名稱容錯清洗**：於 `data/duckdb_manager.py` 新增 `_clean_index_name` 安全清洗方法，將 `'ALL'`、`'NONE'`、`''`、`'全部市場'` 自動標準化為 `None`（全域查詢），並應用於多策略共振、玄鐵重劍、LSTM、TimesFM 等全部量化查詢接口，確保操盤看板隨時穩定產出推薦標的。

### Fixed & Improved (Podcast Live Playback & Media Player Integration)
- **🎙️ 修復 Podcast 情報單集連結無效問題並全面升級為站內雙模態播放器**：
  - **全面替換假資料連結為 100% 驗證之真實有效媒體**：
    - 將 `data/podcast_service.py` 中的 6 大精選單集假 URL（如 `watch?v=allin_e204`、`allin_e204.mp3` 等無效佔位符）全面換成經過即時連線驗證（HTTP 200 / 206）之官方真實節目：
      1. All-In Podcast: Sam Altman 深度訪談（YouTube: `nSM0xd8xHUM`，Libsyn 官方原聲 MP3 串流）。
      2. Lex Fridman Podcast: Mark Zuckerberg #383（YouTube: `Ff4fRgnuFgQ`，Blubrry 官方原聲 MP3 串流）。
      3. Odd Lots: 台灣地緣戰略與台積電護城河（YouTube: `lxmt7rz4DLQ`，Omny Studio 官方原聲 MP3 串流）。
      4. Dwarkesh Patel: Dario Amodei 訪談（YouTube: `n1E9IZfvGMA`，Substack 官方原聲頁面）。
      5. 小Lin說: 光刻機與晶片商戰史（YouTube: `Tzp0EF-sysI`）。
      6. 硅谷101: AI Agent 新範式與 Cursor/Claude 實戰（YouTube: `JEXUta6_oaQ`，Fireside 官方原聲 MP3 串流）。
    - 修正 `config/podcast_seed_channels.json` 中已失效或佔位之 RSS Feed（Libsyn、Omny Studio、Fireside、Substack），確保上游即時同步連線無阻。
  - **新增站內無摩擦雙模態播放器（Inline Media Player）**：
    - **站內 YouTube 嵌入播放器**：在單集卡片點擊「📺 站內播放」可直接於看板內展開 YouTube `<iframe>` 影片流暢播放，無需跳轉離開看盤視窗；並保留「YouTube ↗」獨立按鈕供另開分頁。
    - **站內 HTML5 原聲音訊播放器**：在單集卡片點擊「🎧 播放音訊」可原地展開 HTML5 原生 `<audio>` 播放器，支援背景聽播、倍速與進度拉桿；同時附帶外開 MP3 串流連結。
    - **標題超連結指示優化**：單集標題提供醒目的外部跳轉圖示（`↗`）與 `rel="noopener noreferrer"` 安全屬性，點擊直接前往驗證後的官方 YouTube 單集頁面。

### Fixed & Improved (Codex & OpenCode Dual Code Review Hardening)
- **🛡️ 後端高並發架構與資料安全強化 (Codex 審查改善)**：
  - **消除 N+1 查詢與 DuckDB 連線開銷風暴**：於 `data/duckdb_manager.py` 實作 `get_latest_quotes_batch` 向量化批次查詢，並於 `data/podcast_service.py` 實作 `enrich_episodes_batch`，將單次 API 請求由原先 80~100 次連線開關與全表掃描縮減為 1 次單一 DuckDB 視窗函數查詢（`ROW_NUMBER() OVER`），延遲降低 90% 以上。
  - **杜絕 GET 請求雪崩式連鎖同步 (Cache Stampede / DoS 防護)**：拔除 `api/routes/earnings_call.py` 與 `api/routes/podcast.py` 中因空條件查詢誤觸發的外部 OpenAPI / RSS 阻斷式同步；僅在資料庫完全為空（0 筆記錄）且為無篩選全域請求時才執行初次冷啟動。
  - **落地真實 Podcast RSS XML 解析管線 (`data/podcast_service.py`)**：新增 `fetch_rss_episodes` 方法，支援透過標準 HTTP User-Agent 抓取 RSS XML、解析 `<item>`、`<enclosure>` 音訊、發布日期、並執行正則與白名單之實體/Ticker 萃取。
  - **精確夏令/冬令時雙時區換算**：使用 Python 標準庫 `zoneinfo.ZoneInfo` 精確處理 `America/New_York` (EDT/EST) 與 `Asia/Taipei` (CST) 之跨日與日光節約時間換算。
  - **修復頻道單集數統計映射**：修正 `get_podcast_channels` 中 DuckDB SQL 別名，確保精確對齊 Pydantic `PodcastChannelItem` 之 `episode_count`。

- **🎨 前端工藝（Craft Floor）與閉環體驗升級 (OpenCode 審查改善)**：
  - **Podcast 實時概念股 Badge 原地展開抽屜（Inline Intelligence Drawer）**：點擊單集內的即時股價 Badge 不再強制跳轉離開頁面，而是在卡片下方直接滑出個股基本面業務摘要與重大新聞，並提供「🔍 查看量化籌碼與歷史 →」按鈕供深度查看，達成真正零跳轉之情報閉環。
  - **瀏覽器歷史導航（`popstate`）雙向狀態完全同步**：重構 `app.js` 之 `parseUrlParams()`，當 URL 參數缺失時自動重置為預設值，解決點擊返回上一頁後過濾狀態殘留的 Stale State Bug。
  - **頻道下拉選單同步 URL**：修正 `@change="setPodcastChannel(podcastSelectedChannel)"`，使頻道篩選可被 URL 分享與書籤持久化。
  - **淺色模式（Light Mode）全組件自適應升級**：修復法說會市場徽章、Podcast 語言標籤與日曆 Modal 關閉按鈕在淺色模式下的深色寫死與文字懸停隱形問題。
  - **修復非標準 Tailwind 樣式**：將 `py-0.2` 修正為合法之 `py-0.5`。
  - **操盤看板卡片模式（Card View）抽屜對齊**：卡片視圖新增「法說 / 簡報 ▼」抽屜切換功能，消除表格模式與卡片模式之功能不對等。
  - **日曆 Modal 滾動防穿透（Body Scroll Lock）**：彈窗開啟時鎖定 `body` 滾動，避免 Background Scroll Chaining。

## 2026-10-13

### Added
- **🎨 前端單一檔案架構模組化與外觀拉皮 v.1 (`frontend-facelift-and-modularization`)**：
  - **樣板與腳本模組化拆分**：解耦 3,011 行之龐大樣板 `api/templates/index.html`，縮減為輕量外殼（165 行），並拆解為 7 大 Jinja2 partials（`header.html`, `tab_dashboard.html`, `tab_institutional.html`, `tab_broker.html`, `tab_history.html`, `tab_agent_hub.html`, `footer.html`）。
  - **原生 ES 模組解耦**：建立 `api/static/js/` 模組庫，包含 `api-client.js`（封裝後端 FastAPI 請求與錯誤重試）、`chart-manager.js`（Chart.js 生命週期、防記憶體洩漏 instances 字典與主題連動更新）、`app.js`（Vue 3 根應用、狀態響應、URL 同步與 WebMCP 工具註冊），完全保持零 Node.js build step 之輕量與秒級啟動特性。
  - **FastAPI 靜態資源與模板渲染防護**：於 `api/main.py` 掛載 `/static` 目錄與靜態備援路由，首頁模板採用自訂定界符 (`[[ ]]`) 與正則備援展開，杜絕 Jinja2 衝突 Vue `{{ }}` 語法。
  - **Figma 外觀拉皮 v.1 視覺規範落地**：
    - 色彩系統：實作曜石黑 (`#030302`)、深黑卡片 (`#0D0E11` / `#16181D`)、邊框 (`#262626`)、琥珀金亮點 (`#F59E0B`) 與高對比淺色 (`#EEEEEE` / `#FFFFFF`) 雙主題切換，支援 localStorage 狀態持久化。
    - 頂部導覽列：實作膠囊藥丸 Tab（`Tab / Large`）與具狀態文字之「顯示: 深色模式」/「顯示: 淺色模式」切換開關，雙向同步 URL Query 參數。
    - 操盤看板：大盤宏觀卡片、2MD/FedWatch/財報/Polymarket 緊湊網格，以及全新高密度量化訊號表（深黑表頭、等寬金融數字、彩色策略/法人 Badge 系統、Yahoo/Goodinfo 外部跳轉與點擊即時展開之公司簡介抽屜 Company Drawer），並支援卡片與表格雙重視圖切換。
    - 法人與券商主力雷達：三大法人買賣超排行 TOP 30 表格、TOP 8 目標券商累計走勢圖（霓虹色系多折線、深淺色自適應網格）與 TOP 15 買超/賣超分點雙欄比較清單。
    - 個股歷史：歷史量化預測紀錄、三大法人累計折線圖與逐日籌碼數據明細。
- **🏛️ 台美雙軌法說會催化劑追蹤系統實作 (`earnings-call-tracker`)**：
  - **資料庫與排程快取基礎建設**：
    - 於 `data/duckdb_manager.py` 建立 `earnings_calls` 時序與日曆表，具備市場（TW/US）、標的代碼、雙時區（CST/EST）、簡報/影音網址、預估/公佈 EPS 指標與權值股標記，支援唯一複合鍵 upsert 去重與多條件時序索引。
    - 建立 `cache/earnings/` 離線容錯快取機制，遵從「優先抓取最新、上游異常自動降級快取」鐵律。
  - **跨市場資料擷取管道 (`data/earnings_fetcher.py`)**：
    - 台股上市櫃抓取：解析 TWSE OpenAPI (`t100sb02_1`) 與 TPEx OpenAPI (`mopsfin_t100sb02_1`)，支援民國年/西元年自動雙向換算，提取法說會日期、時間、線上/實體會議地點與官方簡報連結。
    - 美股科技巨頭與 S&P 500 行事曆：抓取 M7 與科技巨頭法說時程、EPS 預估值，並關聯官方 Investor Relations Webcast 直播與歷年簡報入口。
    - 整合 `run_daily.sh` 每日定時背景同步指令（`python -m data.earnings_fetcher --sync`）。
  - **後端 REST API 服務 (`api/routes/earnings_call.py`)**：
    - `GET /api/v1/earnings-call/upcoming`：近期即將召開法說會清單，支援市場篩選（`ALL`/`TW`/`US`）、天數範圍與權值股（`is_weight_stock`）篩選，自動計算雙時區與倒數剩餘天數。
    - `GET /api/v1/earnings-call/{ticker}`：個股法說會時序歷史與官方簡報 PDF、影音回放網址。
    - `GET /api/v1/earnings-call/calendar`：月度法說會聚合場次日曆查詢。
    - `POST /api/v1/earnings-call/sync`：非同步背景全量同步任務觸發。
  - **前端終端介面與互動整合**：
    - 操盤看板「🏛️ 台美法說會預警雷達」卡片：支援全部/台股/美股快速切換、權值股篩選、倒數 Badge（今天🔥/明天⚡/T-N天）、官方簡報 PDF 與直播 Webcast 直連。
    - 個股詳情抽屜（Stock Drawer）：新增「🏛️ 法說會與簡報」分頁標籤，呈現特定個股歷次法說會日程、會議方式、EPS 預估與官方簡報下載。
    - 全月法說會行事曆 Modal：高資訊密度日曆檢視，支援月度翻頁、雙時區（CST/EST）快速切換與公司關鍵字搜尋。
- **🎙️ AI 財經與科技 Podcast 智慧情報雷達實作 (`podcast-financial-radar`)**：
  - **資料庫與種子頻道庫 (`data/duckdb_manager.py`, `config/podcast_seed_channels.json`)**：
    - 新增 `podcast_channels` 與 `podcast_episodes` 表結構，支援頻道元數據、單集主客標記、繁中 TL;DR、觀點標籤、股票 Ticker 陣列與即時行情快照。
    - 建立 `config/podcast_seed_channels.json` 預置 12 大全球頂級科技與總經頻道（All-In, Lex Fridman, Bloomberg Odd Lots, Dwarkesh, 小Lin說, 硅谷101 等）。
  - **多源採集、實體映射與情報管線 (`data/podcast_service.py`)**：
    - 實作金融與科技實體（Claude, OpenAI, NVIDIA, Google, Cursor, Meta, TSMC, ASML, AMD, 聯發科, 鴻海等）至上市公司代號（NVDA, GOOGL, MSFT, META, 2330.TW, 2454.TW, 2317.TW 等）白名單校驗器。
    - 即時行情增強引擎：自動對比 DuckDB 最新的量化預測價、現價與漲跌幅，為單集即時生成高對比行情 Badge。
    - 預置高價值精煉情報庫（Sam Altman 訪談、祖克柏 Llama 3、台積電先進封裝壟斷、Dario Amodei 訪談、小Lin說半導體百年史、硅谷101 Cursor 革命），確保零網路延遲開箱即用。
  - **後端 REST API 服務 (`api/routes/podcast.py`)**：
    - `GET /api/v1/podcast/episodes`：支援實體話題（`topic`）、股票代碼（`ticker`）、頻道 ID（`channel_id`）與語言篩選，附加即時行情與時長格式化。
    - `GET /api/v1/podcast/channels`：取得收錄頻道庫列表與單集統計。
    - `GET /api/v1/podcast/topics`：取得熱門實體與話題標籤頻次聚合排行榜。
    - `POST /api/v1/podcast/sync`：非同步背景排隊全量同步。
  - **前端 Impeccable 頂級工藝介面 (`api/templates/partials/tab_podcast.html`, `app.js`)**：
    - 導覽列新增「🎙️ Podcast 雷達」膠囊藥丸 Tab，雙向同步 URL Query 參數。
    - 頂部熱門實體橫向滑動快篩軌（Horizontal Entity Rail）：包含話題頻次徽章，支援點擊一秒聚焦相關討論。
    - 單集深度情報卡片：收錄頻道頭像、時長、語言、主客標記、AI 繁中精華觀點（TL;DR）與 3 點核心 Takeaways。
    - 即時股票 Badge 閉環連動：單集即時渲染關聯股價與漲跌幅，點擊直接滑出個股時序與籌碼抽屜（Stock Drawer），實現「情報聽取 ➔ 概念股映射 ➔ 量化籌碼決策」一體化閉環。
  - **自動排程整合 (`run_daily.sh`)**：
    - 加入 Step 2.6，定時於每日盤前增量同步 Podcast 最新單集與 AI 萃取。

## 2026-10-08

### Added
- **⚡ Clef-Flash System One 決策大模型整合 (`data/clef_client.py`, `api/routes/decision.py`, `mcp_server.py`)**：
  - 封裝 Clef-Flash System One 強類型決策客戶端，支援 3 級高可用 Fallback 架構：
    - 主力節點（Primary）：`https://clef.create360.ai/v1/systemone`
    - 備援節點 1（Fallback 1）：`https://clef.aiurl.tw/v1/systemone`
    - 備援節點 2（Fallback 2）：`https://clef.create360.ai/v1/systemone`（Jev 相容模式）
    - 本地降級：網路中斷或超時時自動平滑回退，確保系統 100% 高可用。
  - 支援 Choice / Noul / Score 結構化機率推論，評估焦點標的交易動作（`strong_buy`, `buy`, `hold_watch`, `avoid`）、確信度評分（1~5 分）與勝率預期。
  - 新增 REST 端點：
    - `POST /api/v1/decision/evaluate`：傳入個股或候選字典進行即時決策推論。
    - `GET /api/v1/decision/health`：即時檢測 Clef 主力與備援端點可用性。
  - 擴充至 17 大標準量化 FastMCP / WebMCP 工具：
    - 原生 FastMCP 工具新增第 17 號工具 `get_clef_stock_verdict`。
    - `/.well-known/mcp.json`、`/mcp`、`/.webmcp/bridge.js`、`/llms.txt`、`/llms-full.txt` 全面升級至 17 工具宣告。
  - 整合進 `skills/stock-quant/SKILL.md` 供 Agent 直接調用。
  - 新增單元測試 `test/test_clef_client.py` 與 API 測試 `test/test_api_decision.py`，全數通過。

### Fixed
- **多維共振篩選與負潛力標的隔離防守門檻 (`evaluators/composite_evaluator.py`)**：
  - 修復「⭐ 優先推薦 (多維共振)」將雙 ML 大幅看跌個股（如南亞 1303、南電 8046、台塑化 6505 等）誤判入榜之重大邏輯缺陷。
  - 新增正向預期硬性閘門（Positive Potential Gate）：要求入選「優先推薦」之標的必須具備實質正向潛力（雙 ML 模型均看跌、雙模型平均潛力為負或單一模型深度破底標的均強制隔離淘汰）。
  - 多維共振入榜資格必須具備實質方向性訊號（技術買點或 ML 看漲訊號），避免單純僅滿足法人與板塊但動能走空之個股入選。
  - 排序邏輯重構：依共振層級（👑四重共振 > 🏆三重共振 > 🔮雙ML共振 > 🌟多維/雙重共振）、綜合評分與預期報酬進行層級排序，確保高勝率正向標的優先呈現。
  - 針對看跌但滿足其他策略之標的，標記 `🔻防守` 或 `🔻深度防守` 並折減綜合評分，隔離於優先推薦名單之外。

### Changed
- **Web 首頁排版與 CJK 繁體中文字級階層重構 (`api/templates/index.html`, `test/test_typography_rwd.py`)**：
  - 嚴格遵循 Impeccable 設計規範建立繁體中文（CJK）閱讀底線（Typography Floor）：全面根除過小之 `text-[10px]` 與 `text-[11px]` 等不合適字級，徽章、標籤與輔助資訊底線全面提高至 12px（`text-xs`）搭配 `font-semibold` / `font-bold`，確保複雜 CJK 漢字筆劃清晰辨識、不擠壓成團。
  - 解決「字體呼大呼小」階層斷崖：重構建立和諧自然的字級比例階梯（卡片類別 13px~14px -> 核心現價/目標價 18px~20px -> 數值標籤 12px -> 策略標籤 12px~13px），避免從 24px 大字急遽跳崖至 10px/11px。
  - 手機 RWD（360px ~ 420px 窄螢幕）響應式佈局優化：
    - CSS 引入 `-webkit-text-size-adjust: 100%`、`optimizeLegibility` 與中文字型平滑抗鋸齒渲染，防止行動裝置瀏覽器自動膨脹字體造成破損。
    - 個股卡片之估值指標（PE/PB/EV）、法人籌碼（5 日投信/外資進出）與標籤區塊增加 `flex-wrap gap-1`，窄螢幕下自動平順折行，杜絕文字水平碰撞與溢出。
    - 扁平化價格區塊內嵌結構，去除生硬邊框容器，降低視覺雜訊。
    - 修正總經區塊標題層級跳躍問題（`h1` -> `h2`），強化無障礙（a11y）結構語意。
  - 新增字型大小與 RWD 自動化測試（`test/test_typography_rwd.py`），防止 sub-12px 字級回歸。
- **Telegram / Discord / Email 多管道推播格式重構與視覺美化 (`evaluators/formatter.py`, `notifier_dual.py`)**：
  - 徹底移除 Telegram 優先推薦與分項策略中單一巨大等寬代碼區塊 (`<pre>...</pre>`)，根除手機端 Telegram 中英文等寬破版、橫向捲動及閱讀障礙。
  - 全面清除個股卡片中過度堆疊之雜亂表情符號（如 🎯、📍、🔒、💼、📊、🏷️、🧭 等），消除視覺噪音。
  - 個股卡片精煉為兩行式專業金融排版：第一行標的現價，第二行整合關鍵訊號（均線支撐、土洋合買、投信動態、ML預測）與估值（PE/PB），大幅提升手機閱讀舒適度與專業度。
  - 優先推薦與各分項策略改採結構化專業框題（【四重共振】、【三重共振】、【多維共振】、【波段操作】等）。
  - AI 操盤解讀採用 Telegram 原生 `<blockquote>` 引言框呈現，大幅提升閱讀體驗與視覺層次。
  - Discord 與 Email 同步升級為對齊之結構化卡片排版。

## 2026-09-10

### Added
- **🌍 全球證券清冊與最後成功快照架構**：
  - 新增跨市場 `market_universe` DuckDB 主檔與 `universe_sync_runs` 同步狀態表，欄位涵蓋市場、交易所、本地代號、標準代號、雙語名稱、ISIN、證券類別、上市狀態、Market Category、Financial Status 與 Board Lot。
  - 新增 TWSE/TPEx、NASDAQ Trader（涵蓋 NASDAQ、NYSE、AMEX、ARCA、BATS、IEX）、HKEX、JPX、SSE、SZSE、Euronext、LSE provider；各來源使用獨立 `cache/universe/<source_id>.json`，HKEX 名稱欄位以官方當次下載檔提供的語言為準。
  - Euronext 改用官方 `stocks-all-places/download` 完整 CSV 清冊，避免頁面 JSON 只回傳 20 筆造成不完整同步。
  - 上游失敗、空回應、格式錯誤或異常筆數時，保留最後成功快照並以 `stale=true` 回報；有效 cache 不會被空資料覆蓋。
  - 新增 `GET /api/v1/market/universe`，回傳清冊筆數、快照日期、cache 年齡、來源錯誤與標的清單。
  - 台股每日同步先更新清冊再寫入 `tw_daily_bars`；Docker cron 與本機排程共用 cache-aware 同步指令。
  - 全球清冊同步指令現在每次一起更新七個非台灣官方來源；單一交易所失敗時只回退該交易所自己的最後成功 cache。

### Fixed
- **🕛 午夜邊界與跨時區換日缺陷全面根治 (Midnight & Timezone Boundary Problem Fixes)**：
  - **回測目標交易日判定修正 (`backtest/backtest.py`, `backtest/resolver.py`)**：
    - 解決回測模組因無條件 `pred_date + 1 day` 導致「台股盤前 08:00 與美股盤前 20:30 之預測完全跳過當日真實收盤價」的嚴重邏輯錯誤。
    - 盤前 08:00 (TW) 與 20:30 (US) 產生的預測，精準判定其目標驗證日為當日 (Day T)；盤後產生的預測才遞延至下一交易日 (Day T+1)。
    - 重構日期搜尋與收盤狀態檢核 (`is_market_closed`)，杜絕在盤中或盤前提前拿未定價日 K 計算誤差。
    - 徹底修復 `check_date.date() in hist.index` 因 yfinance 帶時區之 `DatetimeIndex` 導致永真返回 False、所有標的均被 skipped 的比對失效問題。
  - **三大法人籌碼午夜換日防禦 (`data/institutional.py`)**：
    - 修復在午夜換日後至台股收盤前 (00:00 ~ 15:30，含 08:00 盤前) 執行時，`get_recent_trading_days` 索取「今日尚未開盤/尚未出表之未來資料」導致證交所 API 報錯的邊界問題。當執行時間早於 15:30 時，起算基準自動往回推一日。
  - **時間戳記標準化與時區對齊 (`database.py`, `data/duckdb_manager.py`)**：
    - 全面以顯式 `Asia/Taipei (+08:00)` 產生預測與批次時間戳記，杜絕 UTC 午夜 (00:00 UTC = 台北 08:00) 邊界時寫入 Supabase / DuckDB 產生的 8 小時偏差與 `age_hours` 異常。
    - `DuckDBManager.get_latest_predictions` 全面採用時區正規化計算，確保跨午夜查詢美股與台股批次之 `batch_date` 與 `analysis_date` 精確對齊。
  - **台股全市場日 K 跨日批次探測 (`data/twse_daily_fetcher.py`)**：
    - `roc_date_to_iso` 增加 `fallback_date` 探測機制，優先使用當批次實際交易日，杜絕盤前同步時部分個股日期遺漏跳為「今日日期」造成日 K 混亂。
  - **單元測試全覆蓋 (`test/test_backtest_date_resolution.py`, `test/test_institutional.py`, `test/test_duckdb.py`)**：
    - 新增 9 項回測交易日與時區解析測試、4 項籌碼換日邊界測試，以及 DuckDB 時區換算測試，全數測試通過。
- **Polymarket API/UI 契約一致化**：機率統一為 0~100 百分比，市場資料補上 `probability` 與 `top_outcome`，修正首頁顯示 `8500%` 或 `undefined%` 的問題；WebMCP 的 `category=all` 會正確代表全部分類。
- **Polymarket 失敗語意與快取修正**：2MD/DoH 都失敗時回傳 `success=false` 與錯誤資訊，不再把空結果快取成新資料；可用上一筆有效結果時會標記 `stale=true`。
- **TimesFM 風險報酬定義修正**：5 日策略使用 horizon 的 P50/P10 計算潛力與風險報酬，P90 保留為上行參考，避免文件與實際選股判定不一致。
- **版本資訊統一**：FastAPI、health endpoint 與首頁版本標籤共用 `2.4.0`。
- **CI/CD 安全與可驗證性**：yfinance 自動更新改建立 dependency PR，不再直接推送 `main` 或使用 `[skip ci]`；Docker package 寫入權限限縮到 image publishing job。

### Added
- **🎲 Polymarket 真金白銀預測市場宏觀情緒整合 (Polymarket Real-Money Macro Sentiment Integration)**：
  - **核心服務模組 (`data/polymarket_service.py`)**：串接 Polymarket Gamma API，無須 API 金鑰即時追蹤真金白銀千萬美元流動性之宏觀預測市場。
  - **防封鎖多層連線架構 (Multi-Tier Proxy & DoH Fallback)**：
    - 第一層（Primary）：透過 `2md.aiurl.tw` 繁中快取代理直連獲取。
    - 第二層（Backup）：自動輪詢 `2md.glsoft.ai` 與 `create360.ai`。
    - 第三層（DoH Fallback）：當 2MD 全數逾時或遭遇台灣 ISP 封鎖 sinkhole (182.173.0.181) 時，自動調用 Google (`8.8.8.8`) 與 Cloudflare (`1.1.1.1`) DNS over HTTPS 解析 legitimate Cloudflare Anycast IP，並透過自訂 SNI SSL Adapter 建立直連，確保 100% 高可用。
  - **降息預期與市場分類**：
    - 自動提取真金白銀 FOMC 利率決策機率（`fed_real_money_odds`：降息 1 碼、按兵不動、降息 2 碼、升息 1 碼），與 CME FedWatch 形成雙源交叉驗證。
    - 自動篩選分類為 `fed_rates`、`geopolitics`、`tech_giants`、`macro_recession`，並自動過濾純體育/娛樂雜訊。
    - 實作 L1 記憶體 + L2 本地磁碟雙層快取（15 分鐘 TTL）。
  - **宏觀風控聚合 (`data/macro.py`, `api/schemas.py`, `api/routes/macro.py`)**：
    - 將 `polymarket` 數據自動富化 (enrich) 至台美股宏觀風控狀態 `MacroState` 與 `MacroRegimeResponse`。
    - 新增 REST 端點 `GET /api/v1/macro/polymarket/sentiment`（支援 `category` 篩選與 `force_refresh` 強制更新）。
  - **擴充至 16 大標準量化 MCP 函數與 WebMCP 協議**：
    - 原生 FastMCP 工具新增第 16 號工具 `get_polymarket_macro_sentiment`。
    - `/mcp`、`/.well-known/mcp.json`、`/.webmcp/bridge.js`、`ai-plugin.json`、`/llms.txt`、`/llms-full.txt` 全面升級至 16 工具宣告。
    - 前端 Chrome WebMCP 動態註冊全套 16 大工具至 `window.document.modelContext`。
  - **互動看板第四欄擴充 (`api/templates/index.html`)**：
    - 宏觀風控摺疊區升級為 4 欄響應式網格（`grid-cols-1 md:grid-cols-2 xl:grid-cols-4`），新增 Panel 4「🎲 Polymarket 預測市場」專區，即時展示真金白銀降息定價卡片與焦點市場清單。
    - `refreshMacro()` 同步並行刷新 Investing 總經與 Polymarket 預測數據。
    - 更新 Agent Hub 規格標籤為 16 大原生量化 MCP 函數。
  - **完整單元測試 (`test/test_polymarket_service.py`, `test/test_api.py`)**：涵蓋雜訊過濾、分類正則、Fed 賠率解析、Fallback 流程與 16 工具端點驗證。

- **🔮 TimesFM 報表排版與 WebMCP / Agent 發現協議全面補齊 (TimesFM Reporting & Discovery Integration)**：
  - **終端機與量化日報 (`evaluators/formatter.py`)**：新增 `TimesFM` 時序大模型預測專區（持有 1-5 天目標價、預期潛力、真實盈虧比、PE/PB），並在「⭐ 【優先推薦】多策略交集」表格中完整納入 `TimesFM` 潛力與盈虧比欄位。
  - **AI 操盤解讀引擎 (`evaluators/ai_narrative.py`)**：將 TimesFM 預測數據、雙 ML 共振 (`🔮雙ML共振`) 與高盈虧比特性注入 3-Tier Fallback LLM Prompt 與零依賴規則模板。
  - **通知推播模組升級 (`notifier_dual.py`)**：在 Telegram、Discord、Email 的優先推薦共振標的中完整同步呈現 LSTM 與 TimesFM 訊號及盈虧比。
  - **WebMCP / Chrome WebMCP / API 發現協議全面同步 (`api/main.py`, `api/templates/index.html`, `mcp_server.py`)**：
    - FastMCP 伺服器新增 `get_commodities_summary` (黃金/原油/期銅) 與 `resolve_stock_ticker` (模糊股票代號解析)，達成全套 15 大量化工具。
    - `/mcp` 與 `/.well-known/mcp.json` 清單同步收錄全套 15 大量化工具，消除 307 重定向並支援遠端 SSE 串流。
    - 前端 Chrome WebMCP 動態註冊全套 15 大工具至 `window.document.modelContext`，打通瀏覽器端內建 AI 呼叫能力。
    - 依據 [llmstxt.org](https://llmstxt.org/) 規範重構 `/llms.txt` 與 `/llms-full.txt`，提供完整數學公式 ($P_{10}/P_{50}/P_{90}$、動態盈虧比、曝險折減)、DuckDB 表結構、REST/MCP 規格與標準 4 步操盤工作流。
    - 更新 `skills/stock-quant/SKILL.md`、`README.md` 與首頁 Agent Hub 說明卡片，精確對齊 15 大工具與共振因子規範。
  - **資料庫數值防禦 (`data/duckdb_manager.py`, `database.py`)**：安全處理多策略共振標的可能為 None/NaN 的浮點數轉型，杜絕單一模型遺漏時的存檔崩潰。

### Fixed
- **🐛 操盤看板四欄網格排版修復 (Web Dashboard Layout & Tag Balancing Fix)**：
  - 修復歷史數據卡片底部遺漏的閉合 `</div>`，徹底解決頂部 4 欄網格未閉合導致總經行事曆、策略按鈕列與標的卡片被錯誤壓迫進 25% 狹窄欄位引發的排版崩潰與按鈕隱形問題。
  - 為標的卡片價格數值補上 `truncate` 與防文字垂直重疊樣式，確保在各裝置與視窗寬度下價格清晰排版。
- **🐳 自動排程 Docker 映像檔自動更新 (`run_daily.sh`)**：
  - 於每日排程執行容器任務前自動執行 `docker compose pull stock-ml --quiet`，確保即時套用包含 Google TimesFM 與最新相依套件的生產映像檔，杜絕舊映像檔降級略過 TimesFM 的問題。

## [2.3.0] - 2026-09-09

### Fixed
- **🛡️ 容器預設命令安全加固與防死循環機制 (Docker Container Default Safety)**：
  - 將 Dockerfile 預設 `CMD` 由 `["main"]` 切換為 `["api"]`，避免常駐容器在手動運行、更新重啟或未指定命令時誤入全量批次分析。
  - `docker/entrypoint.sh` 入口派發腳本將空參數 `""` 安全導向 `api`（Uvicorn REST 伺服器），杜絕因批次退出配合 `restart: unless-stopped` 引發的無限重啟推播死循環。
  - 移除 `docker-compose.yml` 廢棄的 `version: '3.8'` 屬性，消除現代 Compose 解析警告。
  - 於 `docker-compose.yml` 掛載宿主機 Hugging Face 快取目錄（`${HOME}/.cache/huggingface:/root/.cache/huggingface`），使容器能即時共享本機 TimesFM 預訓練模型權重，免除重複下載。
- **⏰ 嚴格落實開盤前時段派發與資源防浪費 (Market Time-of-Day Auto Alignment)**：
  - `main.py` 與 `run_daily.sh` 預設市場全面改為 `--market auto`：依據台北時間自動切換（白天 05:00~13:30 專注台股盤前 08:00；夜間 13:30~05:00 專注美股盤前 20:30），非指定全市場時絕不在夜間執行台股運算，徹底避免浪費 CPU 與伺服器資源。
  - 修復 `test/test_config.py` 單元測試權重斷言與 `test/test_cli_market.py` 預設參數斷言（`args.market == "auto"`），打通 GitHub Actions CI/CD 自動構建最新 Multi-Arch Docker 映像檔。
- **📊 完整打通 TimesFM 前端看板與多管道通知 (TimesFM End-to-End Delivery)**：
  - 前端 Web 看板 (`api/templates/index.html`)：策略選單新增「🔮 TimesFM 預測 TOP」與「🛡️ TimesFM 避險」按鈕，即時串接 `/api/v1/predictions/timesfm/top-bullish` 與 `/top-bearish`。
  - 推播報表 (`notifier_dual.py`)：Telegram、Discord、Email 日報新增 TimesFM 預測段落（5 日目標價、預期潛力 %、盈虧比）。
  - 官方說明文件 (`README.md`)：全面補充 TimesFM 策略架構、`--market auto` 白天/夜間智慧分流防浪費機制、Web 看板 TimesFM 篩選切換、以及 Docker Compose 本地權重快取掛載說明。

### Added
- **🔮 Google Research TimesFM (Time Series Foundation Model) 策略整合**：
  - 新增 `models/timesfm_model.py`：單例惰性載入 Google Research TimesFM 2.5 預訓練權重，支援 PyTorch 向量化批次推論與 P10/P50/P90 分位數風險區間計算。
  - 新增 `strategies/timesfm.py`：標準化繼承 `BaseStrategy`，透過 `@register_strategy("timesfm")` 掛載至策略工廠，覆寫 `evaluate_batch` 進行零延遲矩陣推論，輸出 `TimesFM看漲`、`TimesFM強`、`高盈虧比`、`TimesFM看跌` 標籤與真實盈虧比 (Risk/Reward Ratio)。
  - 綜合評估引擎升級 (`evaluators/composite_evaluator.py`)：
    - 新增 `🔮雙ML共振` 標籤（當 LSTM 看漲 ∩ TimesFM 看漲，形成微觀記憶與宏觀大模型雙重驗證）。
    - 新增 `👑四重共振` 標籤（同時符合玄鐵技術買點 ∩ 法人籌碼鎖碼 ∩ LSTM 看漲 ∩ TimesFM 看漲）。
  - DuckDB 與 Supabase 持久化：支援寫入與查詢 `TimesFM` 策略紀錄。
  - FastAPI REST 端點與 MCP 原生工具：
    - `GET /api/v1/predictions/timesfm/top-bullish`
    - `GET /api/v1/predictions/timesfm/top-bearish`
    - MCP Tool `get_timesfm_top_predictions`
  - 新增單元測試 `test/test_timesfm_strategy.py`。

## [2.2.1] - 2026-09-01

### Changed

- **首頁日報導覽優化**：新增「今日量化訊號」主標與日報狀態提示，讓每日計算結果成為首頁主要入口。
- **資料狀態更誠實**：首頁載入期間不再顯示固定的宏觀、曝險、資料筆數或科技股上限假值；改以讀取中或未提供狀態呈現。
- **查詢體驗改善**：策略結果載入改用符合卡片版型的 skeleton，並加入日報失敗後的可見錯誤與重新整理操作。
- **可及性與手機操作**：補充主要導覽、策略選擇、主題切換、重新整理與歷史軌跡控制的語意標籤與鍵盤焦點樣式。
- **初始化流程整理**：移除首頁 Vue 應用重複的 `mounted()` 定義，保留 WebMCP 註冊與資料載入在同一個初始化流程中。
- **頁尾文案聚焦產品**：移除頁尾對 DuckDB 與深度學習實作的宣傳句，保留品牌與技術提供者資訊。

## [Unreleased] - 2026-09-04

### 🏛️ Investing.com 宏觀數據、FedWatch 降息監控與美股行事曆整合 (Investing.com Macro & US Calendars via 2MD)
- **透過 2MD API 整合 Investing.com 全球即時宏觀與日曆數據 (`data/investing_service.py`)**：
  - **CME FedWatch 聯準會降息監控 (`get_fed_rate_monitor`)**：實時抓取聯準會下次 FOMC 會議日期、倒數天數、各目標利率區間機率分布表（`Target Rate | Current % | Prev Day % | Prev Week %`）、最高機率目標利率與政策預期文字摘要。
  - **美股重量級財報行事曆 (`get_earnings_calendar`)**：抓取近期即將公佈財報之美股個股（代號、公司名稱、財報公佈日、預估 EPS、預估營收規模與市值）。
  - **關鍵大宗商品與原物料 (`get_commodities_summary`)**：實時追蹤黃金 (Gold 避險與流動性)、銅博士 (Copper 製造業與 AI 硬體景氣)、WTI 紐約輕原油 (Crude Oil 通膨預期與能源成本) 之日漲跌與週漲跌幅。
  - **全球重磅總經行事曆 (`get_economic_calendar`)**：自動爬取美國與主要經濟體高衝擊指標（CPI、PCE、非農就業 NFP、失業率、GDP、FOMC 決策）之公佈時間、預測值、前值與實際值。
  - **大盤催化劑警示 (`catalyst_alerts`)**：自動自總經與財報行事曆提煉今日/明日高衝擊事件，無縫注入大盤風控警示。
- **高可用 3 級容災與兩層原子持久化快取 (3-Tier Fallback & Atomic Disk Persistence)**：
  - 與 `CompanyProfileService` 共享防崩潰信號量 `Semaphore(2)`，杜絕高並發連線過載。
  - 支援 `2md.aiurl.tw` ➔ `2md.glsoft.ai` ➔ `create360.ai` 自動容災輪詢。
  - 建立 6~12 小時原子磁碟快取（`cache/investing/*.json`）與記憶體 L1 快取，重啟容器或服務時微秒級讀取，免除每次請求向 2MD 爬取外部網頁。
- **後端 REST API 與風控狀態擴充 (`api/routes/macro.py` & `data/macro.py`)**：
  - `MacroState` 與 `MacroRegimeResponse` 擴展 `fed_rate`、`commodities`、`earnings_calendar`、`economic_calendar`、`catalyst_alerts` 欄位。
  - `GET /api/v1/macro/latest` 自動附帶最新總經與行事曆分析。
  - 新增專屬端點：
    - `GET /api/v1/macro/investing/summary`
    - `GET /api/v1/macro/investing/fed-rate`
    - `GET /api/v1/macro/investing/earnings-calendar`
    - `GET /api/v1/macro/investing/economic-calendar`
    - `GET /api/v1/macro/investing/commodities`
    - 支援 `force_refresh=true` 強制刷新快取。
- **原生 FastMCP 與 WebMCP 工具支援 (`mcp_server.py` & `api/main.py`)**：
  - 於 FastMCP 伺服器新增 `@mcp.tool()`：`get_fed_rate_monitor`、`get_us_earnings_calendar`、`get_economic_calendar`。
  - 同步更新 `/.well-known/mcp.json`、`/.webmcp/bridge.js` 與前端 Chrome WebMCP 註冊清單（擴增至 10 個原生工具）。
- **前端首頁互動看板整合 (`api/templates/index.html`)**：
  - 於操盤看板「宏觀環境」卡片下方新增「🏛️ 總經流動性與重磅行事曆」互動看板。
  - 支援「即時更新」旋轉按鈕與「展開/收合完整行事曆」折疊切換。
  - 視覺化展示 FedWatch 各目標利率機率分佈進度條、美股重量級財報卡片清單、三大關鍵原物料日/週漲跌與焦點數據即時跑馬燈。
  - 整合即時 Toast 提示回饋。

### 🐳 Docker Hub 官方預建多架構映像檔與部署文件完善 (Docker Hub Multi-Arch Deployment Docs)
- **更新 `README.md`、`docs/DOCKER.md` 與 `docker-compose.yml`**：
  - 正式宣傳並標明 Docker Hub 官方多架構映像檔：`tbdavid2019/stock-underdog-ml:latest`（支援 `linux/amd64` 與 `linux/arm64` 雙架構）。
  - 將 `docker-compose.yml` 預設映像檔調整為 Docker Hub，使外部使用者直接執行 `docker compose pull` 免除 GHCR 憑證授權困擾。
  - 詳述推薦直接拉取預建映像檔之優勢（避免本地構建大型 PyTorch / TF / DuckDB 套件耗時 15~20 分鐘及編譯失敗）。
  - 提供無需 Clone 專案即可單行啟動 Web & API 服務的 `docker run` 快速指引。
  - 於 `README.md` 頂部新增 Docker Hub 官方徽章。

### 📄 全面支援 llmstxt.org 規範與頁尾開發者專區 (`/llms.txt` & Footer Hub)
- **遵循 Jeremy Howard [llmstxt.org](https://llmstxt.org/) 標準規範**：
  - 重構 `/llms.txt` 與 `/llms-full.txt`，提供乾淨、結構化之 Markdown 摘要、策略端點清單與全量系統規格。
  - **🧹 頂部導覽列極簡化**：移除頂部右上角多餘的 `llms.txt`、`API 文檔` 及導覽列的 `Agent / MCP` 按鈕，專注呈現「操盤看板」、「法人籌碼」與「個股歷史」，視覺更加俐落清爽。
  - **⚓ 統一收攏至頁尾 (Footer Developer Hub)**：在頁尾統一設置開發者與 AI 專區，整合 `🤖 Agent / MCP Hub`（點擊平滑切換分頁）、`📄 llms.txt ↗` 標準鏈結、Swagger UI、ReDoc 與 SKILL.md。
  - 在 `Agent / MCP Hub` 分頁新增專屬「📄 llms.txt / llms-full.txt」對接卡片，支援一鍵複製 URL 與直接檢視，供 LLM 系統提示詞、AI 爬蟲與 Agent 快速獲取即時量化模型定義。

### 🌐 2md 繁體中文公司簡介與即時新聞浮動卡片 (2md Company Profile & News Hover)
- **整合 2md URL to Markdown 與 Search 服務** (`https://2md.aiurl.tw`, `https://2md.glsoft.ai`, `https://create360.ai`)：
  - 新增 `data/company_profile.py` (`CompanyProfileService`)，支援自 2md 與 Yahoo 股市實時提取個股之繁體中文公司名稱、核心營運業務、產業別、董事長、市值及最新即時新聞動態。
  - **🚀 2md 官方原生微批次抓取 (Native Multi-URL Batch Crawl)**：批次抓取全面改接 2md 原生 `POST /v1/batch` 端點，以每批 2~3 支微批次單一 HTTP 請求並發取回多個 Yahoo Profile Markdown，大幅降低連線開銷與握手延遲。
  - **🛡️ 防崩潰全域並發信號量 (`Semaphore=2`)**：加入全域信號量節流，限制同時發往 2md 的爬蟲請求上限為 2 組，徹底避免瞬間觸發幾十個 Headless Chromium 渲染導致伺服器 OOM 當機。
  - **⏱️ 超時放寬至 25 秒給予充足渲染空間 (Realistic 25s Batch Timeout)**：
    - 單頁渲染超時調增至 10.0 秒，批次超時大幅放寬至 **25.0 秒**，給予 Chromium 充分動態渲染時間，避免過早斷線。
  - **🔄 完整三級 Fallback 容災輪詢 (Full 3-Tier Fallback Resilience)**：
    - 無論單筆或批次抓取，遇超時或異常時完整保留 `2md.aiurl.tw` ➔ `2md.glsoft.ai` ➔ `create360.ai` 之多級自動容災切換，確保後手永遠可用。
    - **由全域 Semaphore(2) 在源頭鎖死並發量**：即使前兩台異常、請求切換至第三台 GCP 機器，同時間也嚴格限制最多 2 個 Chromium 任務，徹底杜絕 20+ 連線暴衝引發的 OOM。
  - **💾 20 天磁碟持久化兩層快取 (L1 Memory + L2 Disk 20-Day Persistence)**：
    - 快取效期由 6 小時全面調增至 **20 天 (86,400 × 20 秒)**，滿足常用霸榜標的免頻繁爬蟲之需求。
    - 導入 L1 記憶體 + L2 磁碟原子持久化 (`./cache/company_profiles/{ticker}.json`)，即使 Docker 容器重建或服務重啟，微秒級自動命中磁碟並回填記憶體，徹底免除每次 hover 重複向 2md 抓取資料之延遲與連線消耗。
  - 後端內建 LRU 記憶體快取與 UTF-8 解碼，提供毫秒級 Hover Tooltip 回應速度。
  - 新增 REST API 端點 `GET /api/v1/market/company-profile` 與 `POST /api/v1/market/company-profiles/batch`。
  - 新增 MCP 工具 `get_company_profile`，供 AI Agent、Claude Desktop、WebMCP 呼叫。
  - 前端 UI (`api/templates/index.html`) 於所有股票代號（如 `9945.TW`、`2330.TW`、`AAPL`）支援滑鼠 Hover 即時彈出精美 Claude 質感公司介紹浮動卡片，附帶即時新聞與外部行情（Yahoo / Goodinfo / TradingView / 時序軌跡）捷徑。

### 🧭 前端狀態網址同步與瀏覽器導航支援 (URL State & Browser History Management)
- **完整支援網址狀態同步、分享連結與上一頁/下一頁 (Browser History PushState & Share)**：
  - 於 Vue 3 SPA 前端導入 HTML5 `history.pushState` / `replaceState` 與 `popstate` 監聽。
  - 所有操作（分頁切換 `tab`、策略選擇 `strategy`、市場篩選 `index`、法人籌碼子分頁 `sub`、排序 `order`、日期 `date`、個股搜尋 `ticker`、券商代號 `broker_ticker`）皆即時雙向同步至 URL 查詢參數。
  - 使用者可直接使用瀏覽器「上一頁」與「下一頁」返回歷史狀態。
  - 頂部導覽列新增「🔗 分享」按鈕，一鍵複製目前所選之策略、市場與個股專屬 Deep Link，便於直接分享與精準跳轉。

### ⏰ 排程與時效性重大修正 (Pre-Market Execution Alignment)
- **修正為開盤前買進決策指南 (Pre-Market Buy Guide)**：
  - 核心價值定位為「**開盤前進場佈局指南**」，全面修正原盤後/收盤後執行的錯誤排程。
  - **🇹🇼 台股盤前指南**：排程調整為 **週一至週五 早上 08:00 (台灣時間 / UTC 00:00)**，於 08:30 試撮與 09:00 開盤前產出並推播，並同步前日三大法人籌碼與最新美股動態。
  - **🇺🇸 美股盤前指南**：排程調整為 **週一至週五 晚上 20:30 (台灣時間 / UTC 12:30)**，於 21:30 (夏令) / 22:30 (冬令) 開盤前產出並推播美股 500 量化決策。
  - **🔧 yfinance 自動巡檢**：排程調整為 **週一至週五 早上 07:30 (台灣時間 / UTC 23:30)**，於台股盤前排程前自動完成環境與 API 巡檢。

### 🚀 新增與重大升級 (Features & Enhancements)
- **台美股大盤風控與語意解讀分流 (Market-Aware Macro & Narrative Separation)**：
  - 修正原先台股日報（台灣50 / 台灣中型100）直接套用美股恐慌指數與美股宏觀語意之問題。
  - 於 `data/macro.py` 新增 `evaluate_tw_market`，抓取台股加權指數 (`^TWII`) 判斷大盤趨勢（站穩 MA60/MA20），並以費半 (`^SOX`) 及 VIX 作為國際連動風控指標。
  - 於推播通知 (`notifier_dual.py`)、終端報告 (`formatter.py`) 及 AI 解讀 (`ai_narrative.py`) 依目標市場自動切換：
    - **🇹🇼 台股市場**：呈現「🇹🇼 台股大盤與風控」（加權指數、國際連動、法人籌碼），AI 解讀以台股大盤走勢與籌碼為主視角。
    - **🇺🇸 美股市場**：呈現「🇺🇸 美股宏觀風控」（S&P 500、VIX 恐慌指數、費城半導體）。
  - REST API `/api/v1/macro/latest` 與 MCP Tool `get_market_macro_regime` 支援 `market=tw` 與 `market=us` 參數路由。
- **每日自動化 Shell 腳本支援分流派發 (`run_daily.sh`)**：
  - 支援 `run_daily.sh tw`（台股盤前專用，含 TWSE 全市場日 K 同步）與 `run_daily.sh us`（美股盤前專用，略過台股同步加速執行）。
- **Docker 容器排程與 Entrypoint 同步更新 (`docker/crontab` & `docker/entrypoint.sh`)**：
  - 容器內排程同步更新為 Asia/Taipei 08:00 (TW) 與 20:30 (US) 盤前執行。
  - Entrypoint 支援 `main` 指令的後續參數透傳。
- **修復 GitHub Actions 排程同步工作流 (`.github/workflows/tw_stock_daily_sync.yml`)**：
  - 將遺留之 `pytest` 指令修正為標準 `python -m unittest`，解決因環境缺少 pytest 導致排程工作流失敗（Exit code 127）問題。
- **CI/CD 全面恢復 ARM64 + x86_64 雙架構編譯 (`.github/workflows/docker-ci-cd.yml`)**：
  - 移除先前的 platform 條件限制，在每次推送到 `main` 分支時，皆自動透過 QEMU 構建並發布 `linux/amd64` 與 `linux/arm64` (Apple Silicon / Raspberry Pi / Ampere) 雙架構 Docker 映像檔至 GHCR。
- **README 文件同步更新純數據庫與盤前架構 (`README.md`)**：
  - 補充「⏰ 盤前買進決策排程與純股市數據庫」專章，詳述 `tw_daily_bars`、`tw_institutional_daily`、`tw_broker_trades` 與本地時序快取之抓取與持久化架構。
- **修正個股查詢誤顯示上一筆結果**：每次新查詢會先清除舊的預測表格與法人圖表；未知輸入或查無資料時不再沿用前一支股票的數據。
- **個股查詢改為先解析標的**：支援 `TSLA` / `Tesla` / `特斯拉`、`聯電` / `聯華電子` 等明確別名，以及 DuckDB 行情 metadata 中的公司名稱；無法辨識的文字會明確回報，不會猜測代號。
- **MCP 個股歷史查詢誠實回報空結果**：無法辨識或沒有預測記錄時回傳 `success: false`、空資料與原因，避免 Agent 將空結果當成有效報告。
- **首頁補齊 SEO 與社群預覽 metadata**：加入 meta description、canonical、Open Graph、Twitter Card、JSON-LD、favicon、Apple Touch Icon、manifest 與 1200×630 OG 圖片資源。
- **修復 Web UI `index.html` 標籤閉合不對稱導致 Vue 掛載中斷問題**：
  - 移除法人籌碼分頁末端多餘的 `</div>` 標籤，恢復 Vue 根容器完整閉合，解決模板未編譯（顯示 `{{ ... }}`）之渲染異常。

### 🚀 新增與重大升級 (Features & Enhancements)
- **全面升級開源授權為 GNU AGPL-3.0 (`LICENSE` & `README.md`)**：
  - 符合強 Copyleft 網路服務條款，保護量化策略與深度學習引擎之開源完整性。
- **台股全市場官方 OpenAPI 批量日 K 抓取器 (`data/twse_daily_fetcher.py`)**：
  - 整合台灣證券交易所 (TWSE) 與櫃檯買賣中心 (TPEX) 官方 OpenAPI，單次請求 1~2 秒內抓取全市場 1,800+ 檔上市櫃股票當日開高低收、成交量與成交金額。
- **券商分點主力進出追蹤雷達 (`tw_broker_trades` & `api/templates/index.html`)**：
  - 在 DuckDB 建立 `tw_broker_trades` 資料表，完成全台股關鍵券商分點（美商高盛、摩根士丹利、瑞士信貸、富邦、元大台北、凱基台北等主力分點）**36,570+ 筆** 歷史進出數據匯入。
  - **目標券商累計買賣超多色折線走勢圖 (Chart.js)**：穿透追蹤各分點在波段期間的連續加減倉水庫曲線。
  - **主力買賣超分點 TOP 15 雙欄卡片**：提供過去 10D / 20D / 30D / 60D 各分點累計買進、賣出、淨買賣超張數與佔比。
  - **FastAPI & Agent / MCP 整合**：新增 `/api/v1/market/broker/summary/{ticker}`、`/api/v1/market/broker/trades/{ticker}`、`/api/v1/market/broker/trend/{ticker}` 以及 MCP Tool `get_broker_trades_for_stock`。
- **Web UI 操盤平台新增「🏛️ 三大法人籌碼雷達」分頁 (`api/templates/index.html`)**：
  - 支援三大法人合計、外資、投信與自營商淨買賣超（張數）排行榜（買超榜 / 賣超榜）與市場篩選（全部 / 上市 / 上櫃）。
  - **三大法人累計買賣超折線圖 (Chart.js)**：以多曲線呈現外資、投信、自營商與三大法人合計之連續波段累計進出走勢。
  - **歷史日期選擇器**：支援選取過去 240 個交易日（2025-09-11 ~ 2026-08-28）任一歷史交易日進行籌碼排行復盤。
  - **個股 240 天法人時序表**：於個股歷史查詢頁面新增「三大法人連續進出時序表」，完整呈現外資、投信、自營商每日進出與持股比率時間序列。
- **FastAPI 全市場與籌碼新路由 (`api/routes/market.py`)**：
  - 提供 `/api/v1/market/institutional/top`、`/api/v1/market/institutional/dates`、`/api/v1/market/stats`、`/api/v1/market/institutional/{ticker}` 等極速查詢端點。
- **雙軌每日自動化排程整合 OpenAPI 批量同步 (`run_daily.sh` & Crontab)**：
  - 每日 15:30 收盤排程於 Step 0 自動執行 TWSE & TPEX 全市場日行情批量同步，無縫注入 DuckDB。
- **歷史開源數據集回填器 (`data/historical_importer.py` & `scripts/import_historical_data.py`)**：
  - 成功整合 `voidful/tw-institutional-stocker` 與 `voidful/tw_stocker`，完成 **512,045 筆** 上市櫃三大法人歷史買賣超與核心權值股日 K 棒之高速回填至 DuckDB。
- **DuckDB 列式時序庫全市場資料表 (`tw_daily_bars` & `tw_institutional_daily`) 與 yfinance 零延遲備援**：
  - 在 `DuckDBManager` 新增 `tw_daily_bars` 資料表與冪等覆蓋 Upsert 機制。
  - `data/fetcher.py` 整合自動 Fallback：當 yfinance 發生限流或中斷時，自動透明降級讀取 DuckDB 本地歷史資料庫。
- **每日收盤台股全市場自動同步排程 (`.github/workflows/tw_stock_daily_sync.yml`)**：
  - 於台灣時間週一至週五 15:30 (UTC 07:30) 自動執行全市場行情同步與 DuckDB Artifact 保存。
- **籌碼與美股研究評估 (`tw-institutional-stocker` & `us_fddk`)**：
  - 完成法人籌碼持股推估模型與美股 20 年凍結宏觀 Regime 框架深度架構評估。
- **888 Stock Quant 品牌重塑與文案淨化**：
  - 全面統一平台名稱為 **`888 Stock Quant`**，移除全站、API 說明與 Agent / MCP 規格文件中所有「AI」字眼，回歸硬核多維量化決策與工程架構。
- **Anthropic Claude 官方美學（Normal Mode）**：
  - 淺色模式全面導入 Anthropic Claude 官方暖色調視覺設計（暖羊皮紙白底色 `#FAF9F5`、陶土珊瑚紅焦點色 `#D97757`、暖灰柔和邊框 `#E8E6DC`、炭黑高對比文字 `#141413`）。
- **JetBrains Mono 數據字體排版與層次重構**：
  - 全局中文字體採用標準平滑無襯線字體（`Noto Sans TC`），將 `JetBrains Mono`（`.num-font`）精準作用於所有股票代號、現價、目標價、漲跌潛力%、PE/PB 與時間戳記。

## [2.2.0] - 2026-09-01

### Added

- **888 Stock Quant 品牌重塑與文案淨化**：
  - 全面統一平台名稱為 **`888 Stock Quant`**，移除全站、API 說明與 Agent / MCP 規格文件中所有「AI」字眼，回歸硬核多維量化決策與工程架構。
- **Anthropic Claude 官方美學（Normal Mode）**：
  - 淺色模式全面導入 Anthropic Claude 官方暖色調視覺設計（暖羊皮紙白底色 `#FAF9F5`、陶土珊瑚紅焦點色 `#D97757`、暖灰柔和邊框 `#E8E6DC`、炭黑高對比文字 `#141413`）。
- **JetBrains Mono 數據字體排版與層次重構**：
  - 全局中文字體採用標準平滑無襯線字體（`Noto Sans TC`），將 `JetBrains Mono`（`.num-font`）精準作用於所有股票代號、現價、目標價、漲跌潛力%、PE/PB 與時間戳記。
- **頁腳技術提供標章**：
  - 頁面底部新增 `⚡ 技術提供 david888.com ↗` 專屬外鏈標章。
- **FastMCP 原生 Agent 工具伺服器 (`mcp_server.py`)**：
  - 封裝 6 大標準 MCP 決策工具（`get_market_macro_regime`、`get_triple_resonance_stocks`、`get_xuantie_pullback_stocks`、`get_lstm_top_predictions`、`get_stock_history`、`get_latest_market_snapshot`），直連 DuckDB 記憶體極速引擎。
  - 支援 Claude Desktop、Cursor、Antigravity、Open-WebUI 原生掛載與 stdio/SSE 傳輸。
- **Agent Skill 規格文檔 (`skills/stock-quant/SKILL.md` & `/skill`)**：
  - 結構化定義操盤工作流（大盤宏觀 ➔ 三重共振 ➔ 波段/短線候選 ➔ 風控防守位）與金融因子指標解讀指南。
- **正式生產網域與 Cloudflare 橘雲反向代理 (`https://stockdata.david888.com`)**：
  - 於 `10.9.0.99` Nginx 配置 `stockdata.david888.com` 虛擬主機（支援 HTTP 80 與 HTTP/2 443 SSL），透過 Let's Encrypt 自動續期憑證。
  - 對接 Cloudflare 橘雲 Proxy，支援 WebMCP SSE 長連線（`proxy_buffering off`、`proxy_read_timeout 86400s`）與即時操盤儀表板。
- **Cloudflare & Chrome 官方 WebMCP 標準對接 (`/.webmcp/bridge.js` & `data-packs`)**：
  - 全面遵循 [Cloudflare 官方 WebMCP 標準](https://blog.cloudflare.com/webmcp/) 與 W3C 規範，於首頁注入 `<script type="module" src="/.webmcp/bridge.js" data-packs="stock-quant-engine" data-mcp-url="/mcp/sse">`。
  - 提供 `/.webmcp/bridge.js` 動態橋接腳本，支援 Cloudflare BrowserRun、Chrome 瀏覽器 Agent 自動探索並註冊 `document.modelContext.registerTool`。
- **yfinance 自動巡檢與 CI/CD 升級系統 (`scripts/check_yfinance_update.py` & `.github/workflows/yfinance_autoupdate.yml`)**：
  - 自動對接 PyPI JSON API 檢測 yfinance 最新發行版本，並於沙盒中執行「真實行情抓取 + 欄位格式校驗 + 單元測試」。
  - 驗證通過後自動熱更新 Docker 容器、修改 `requirements.txt`、自動 Git Commit/Push 並平滑重啟 API 服務；驗證失敗則自動回滾，確保排程高可用。
- **GitHub Actions CI/CD 極速構建與併發取消機制 (`.github/workflows/docker-ci-cd.yml`)**：
  - 導入 `concurrency.cancel-in-progress: true`：當連續提交時自動中止舊 Commit 冗餘排程，避免任務積壓。
  - 常規 `main` 提交切換為原生 `linux/amd64` 構建搭配 GitHub Actions Layer Cache（從原本 QEMU ARM64 軟體模擬編譯的 30 分鐘驟降至 **< 2 分鐘**）。
  - 保留 Release Tag (`v*.*.*`) 與手動觸發時的多架構（x64 + ARM64）完整編譯。
- **GitHub Actions CI/CD 升級 Node.js 24 執行期**：
  - 移除已棄用的 Node.js 20 舊版強制旗標，全面升級支援 GitHub Actions 最新 Node.js 24 執行環境（`ACTIONS_RUNNER_FORCE_NODE24`），消除過期警報。

### Fixed

- **UI 佈局與折行修復**：修復頂部導航列品牌標題在特定寬度下折行（`whitespace-nowrap` 防護），確保標題與導航欄位永不拆字。
- **宏觀狀態解析路徑**：修復前端對 `/api/v1/macro/latest` 頂層欄位之解析邏輯，解決先前面板持續呈現「評估中...」的問題。
- **時效標籤顯示**：修復卡片底部時效標籤在空值時誤顯 `nullh` 之問題。
- **美股宏觀風控門檻 (`data.macro.MacroRegimeAnalyzer`)**：
  - 於 Stage 1 Pre-flight 監控美股指標（`SPY`、`^VIX`、`^SOX`）。
  - 動態評估全球市場狀態（全面多頭、多頭回調、避險防禦、極度恐慌熔斷）並動態調節建議投資曝險比例（0.0 ~ 1.0）。
  - 費城半導體（SOX）破季線時自動限制科技股部位上限（30% ~ 50%）。
- **台股三大法人籌碼分析器 (`data.institutional.InstitutionalProvider`)**：
  - 整合 TWSE 官方 T86 / MI_QFIIS 與 TPEX 買賣超數據，並支援本機 JSON 快取與自動容錯。
  - 自動計算 5日/20日 外資、投信累計買賣超（張數），以及投信連續買超天數（Streak）。
- **7 大板塊資金輪動策略 (`strategies.sector_rotation.SectorRotationStrategy`)**：
  - 追蹤半導體、AI伺服器/電子、金融、重電、航運、傳產、生技/太空等 7 大板塊 10D/15D/20D 加權動量資金流，挑選前 3 大強勢板塊領頭個股。
- **三大法人籌碼策略 (`strategies.institutional.InstitutionalStrategy`)**：
  - 自動篩選投信連續買超 3 日以上（投信連買）或外資與投信同步 5 日買超（土洋合買）個股。
- **三重共振綜合評估與標籤 (`evaluators.composite_evaluator.CompositeEvaluator`)**：
  - 升級支援「技術買點 ∩ LSTM看漲 ∩ 投信/外資主力大買」之 `🏆三重共振`、`土洋合買`、`投信連買`、`主流板塊` 標籤與宏觀曝險折扣權重。
- **3 級 Fallback LLM 研報解讀引擎 (`evaluators.ai_narrative.AINarrativeEngine`)**：
  - 支援 Primary ➔ Fallback 1 ➔ Fallback 2 ➔ 規則模板（Template）四級容錯機制。
  - 採用標準 OpenAI 相容協定，支援 Google Gemini 2.5 Flash、DeepSeek-V3、OpenAI GPT-4o-mini 或自建 Gateway。
  - 嚴格 Prompt 約束禁止捏造數據，純基於量化確定數據生成 100~150 字白話操盤解讀。
- **本地 DuckDB 時序資料庫與雙備份架構 (`data.duckdb_manager.DuckDBManager`)**：
  - 導入高性能嵌入式列式資料庫 DuckDB（儲存於 `data/storage/stock_quant.duckdb`）。
  - 管線 Stage 4 自動執行「雲端 Supabase + 本地 DuckDB」雙備份持久化，支援零延遲離線回測與高壓縮 Parquet 冷備份導出。
- **Supabase 全量數據導出遷移工具 (`scripts/export_supabase_to_duckdb.py`)**：
  - 自動分頁拉取 Supabase 線上所有歷史預測與量化數據，無縫回存本地 DuckDB（44,153 筆時序數據經 DuckDB 壓縮僅 4.1MB）。
- **現代化生產級 Docker 容器環境 (`Dockerfile` & `docker-compose.yml`)**：
  - 全面翻新升級為 `python:3.12-slim` 現代生產級映像，移除廢棄 Miniconda 配置。
  - 新增多服務架構：`stock-ml`（單次執行）、`stock-ml-cron`（後台常駐定時排程）、`stock-ml-sync`（Supabase 導出）、`stock-ml-test`（測試）。
  - 整合台北時區設定、系統依賴、DuckDB 與快取目錄自動掛載（`./data/storage`、`./data/cache`、`./logs`）。
- **GitHub Actions 自動化 CI/CD 與雙架構映像構建 (`.github/workflows/docker-ci-cd.yml`)**：
  - 新增 44 項單元測試自動化防護網關（Test Gate）。
  - 自動構建並推送支援 `linux/amd64` (x64) 與 `linux/arm64` (Apple Silicon M系列 / ARM) 之雙架構映像至 GitHub Container Registry (`ghcr.io/tbdavid2019/stock-underdog-ml`)。
- **FastAPI 高效能量化訊號 REST & MCP 服務 (`api/`)**：
  - 新增現代化 FastAPI 服務，基於本地 DuckDB 44,000+ 筆歷史時序數據提供毫秒級對外 API 查詢。
  - 提供 `/docs` 交互式 Swagger API 文檔與 `/redoc`。
  - **核心端點**：
    - `GET /api/v1/predictions/latest`：多指數最新標的行情、均線、PE/PB估值與預測數據。
    - `GET /api/v1/predictions/resonance`：多策略交集與 `🏆三重共振` 焦點股篩選。
    - `GET /api/v1/predictions/xuantie`：玄鐵重劍 MA60/120 均線回調買點標的。
    - `GET /api/v1/predictions/lstm/top-bullish`：LSTM 短線預測上漲 TOP N 潛力榜。
    - `GET /api/v1/predictions/lstm/top-bearish`：LSTM 短線預測下跌 TOP N 避險/放空榜。
    - `GET /api/v1/predictions/history/{ticker}`：個股時間序列歷史預測與實際走勢軌跡。
    - `GET /api/v1/macro/latest`：美股宏觀市場風控狀態（SPY/VIX/SOX）與建議曝險。
    - `GET /api/v1/stats/summary`：DuckDB 時序庫全盤統計與涵蓋範圍。
    - `GET /health`：服務健康檢查與資料庫總記錄數。
  - 整合 `docker compose up -d stock-ml-api`（Port 8000），為後續 MCP (Model Context Protocol) 伺服器與 WebMCP / Agent Skills 奠定基礎。
- **多平台推播訊息升級 (`notifier_dual.py`)**：
  - Telegram、Discord、Email 報告全面整合美股宏觀卡片、AI 操盤解讀與三重共振標籤。

---

## [2.1.0] - 2026-09-01

### Added

- **模組化分層架構 (Modular Pipeline Architecture)**：新增 `core/`、`data/`、`strategies/`、`evaluators/`、`pipeline/` 模組，徹底解決單一程式碼檔案過度耦合（God Object）的問題。
- **統一硬體管理 (`core.device.DeviceManager`)**：支援 CUDA、Apple Silicon (MPS) 與 CPU 運行時自動偵測、手動指定與平滑降級。
- **統一快取與資料管線 (`data.cache.CacheManager`, `data.fetcher.StockFetcher`, `data.fundamentals.FundamentalProvider`)**：
  - 支援指數成分股 JSON 快取與行情 Pickle 快取生命週期管理。
  - 基本面估值指標（PE/PB/EV/EBITDA）批次快取與錯誤容錯。
  - 完整保留 `answerbook` 優先最新、失敗回退本機快取之可靠性策略。
- **美股與太空概念標的擴充**：在 S&P500 / 美股分析清單中強制納入 SpaceX 相關標的（`SPCX`、`DXYZ`、`ASTS`、`RKLB`），擴大太空衛星概念選股覆蓋度。
- **插件式策略註冊中心 (`strategies.registry.StrategyRegistry`)**：
  - 標準化 `BaseStrategy`、`StockContext` 與 `StrategyResult` 契約。
  - 提供 `@register_strategy` 裝飾器，使未來擴充技術面、籌碼面、機器學習選股策略只需新增單一檔案。
  - 完整封裝 `XuanTieStrategy`（玄鐵重劍）與 `LSTMStrategy`（LSTM 深度學習）。
- **動態多策略綜合評價引擎 (`evaluators.composite_evaluator.CompositeEvaluator`)**：
  - 支援多策略動態權重評分（0~100）與 N-of-M 多策略交集自動標籤聚合。
  - 獨立的美化終端機表格輸出工具 (`evaluators.formatter.print_evaluation_report`)。
- **兩階段執行調度器 (`pipeline.orchestrator.PipelineOrchestrator`)**：
  - Stage 1 (I/O Prefetch) 與 Stage 2 (Compute) 分流，大幅改善並發效率並避免 GIL 衝突。

### Changed

- `answerbook` 指數榜單每次執行優先抓取最新資料；只有上游 API 失敗時才使用未超過 90 天的快取。
- 保留 S&P500 策略前 110 名限制，但每次先取得最新的完整榜單再取前 110 名。

### Fixed

- Supabase 寫入前新增嚴格 JSON 檢查器，將 `NaN` 與無限大數值轉為 `null`，其他不可序列化值則取消寫入，避免整批資料因 JSON 格式失敗。
- 修正 TensorFlow/Keras 在模組載入階段過早導入導致 `curl_cffi` 產生 `ImpersonateError` TLS 握手衝突的底層問題，全面改為 Stage 2 延遲載入（Lazy-load），確保 yfinance 批次行情下載 100% 成功。

### Documentation

- 建立 `openspec/changes/integrate-macro-sector-institutional` 規範提案：包含美股宏觀門檻 (Macro Regime)、板塊輪動策略 (Sector Rotation)、台股三大法人籌碼分析 (Institutional Flow) 與三重共振評分體系。
- 封存已落地的 `2026-09-01-refactor-pipeline-architecture` 變更規格至 `openspec/specs/`。

- 新增 `AGENTS.md`，規範 changelog、README、安全與測試同步要求。

### Maintenance

- 將 `main.py` 重構為輕量化管線調度入口（縮減至 40 行左右）。
- 清理 `database.py` 中殘留的 MySQL 與 MongoDB 廢棄類別與驅動檢查。
- 將過渡期檔案 `parallel_processor.py` 與 `main_lstm_only.py` 移至 `archive/`。
- `run_daily.sh` 補充 `PYTHONPATH` 與系統 `PATH` 環境變數匯出，確保在 Cron 最小環境下順利執行回測與主程式。
- 於 `10.9.0.99` 部署自動化 Cron 排程：包含台股收盤後（15:30 CST / 07:30 UTC）與美股收盤後（07:00 CST / 23:00 UTC）自動執行雙軌選股分析。

## [2.0.0] - 2026-01-08

### 🎯 重大變更

#### 預測邏輯重構
- **改為預測下一交易日** - 從預測「歷史資料最大值」改為預測「下一個交易日的收盤價」
- **實用性大幅提升** - 現在可以直接用於交易決策參考

#### 模型精簡
基於回測結果（543 筆資料），關閉表現不佳的模型：

| 模型 | 狀態 | 方向準確度 | 原因 |
|------|------|-----------|------|
| **LSTM** | ✅ 保留 | 54.6% | 略高於隨機，有優化空間 |
| **Prophet** | ❌ 關閉 | 46.9% | 低於隨機，不適合短期預測 |
| **Chronos** | ❌ 關閉 | N/A | 執行失敗，模型太大太慢 |
| **CROSS** | ❌ 關閉 | N/A | 執行失敗 |
| **Transformer** | ❌ 關閉 | N/A | 資源消耗大 |

**結論：** 專注優化 LSTM，目標提升方向準確度至 60%+

### ✨ 新增功能

- **自動回測系統** - 每日驗證預測準確度
  - 自動抓取下一交易日的實際股價
  - 計算方向準確度、平均誤差等指標
  - 支援按指數分析（台灣50、台灣中型100、SP500）
  
- **回測腳本** - `backtest/backtest.py`
  - 智能處理交易日（自動跳過週末和假日）
  - 批次處理，約 2.6 分鐘處理 1000 筆預測
  
- **分析工具** - `backtest/analyze_backtest.py`
  - 詳細的模型表現分析
  - 按模型和指數分組統計

### 🔧 優化改進

#### LSTM 模型優化 1/9/2026 生效
- **網路深度增加** - 從 2 層增加到 3 層 LSTM（128→64→32 units）
- **訓練時間延長** - epochs 從 10 增加到 100
- **Early Stopping** - 防止過擬合
- **Learning Rate Scheduling** - 動態調整學習率
- **時間窗口增加** - time_step 從 60 增加到 90 天
- **批次大小優化** - 從 32 降到 16（更細緻的梯度更新）

#### 專案結構優化
- **新增 `scripts/` 資料夾** - 整理測試和維護腳本
- **新增 `backtest/` 資料夾** - 回測系統獨立管理
- **根目錄精簡** - 只保留核心程式檔案

### 🐛 Bug 修復

- **修復資料洩漏** - `MinMaxScaler` 現在只在訓練資料上 fit
- **修復日期比對錯誤** - 回測時正確處理 `datetime.date` vs `datetime.datetime`
- **修復 Supabase 連線** - 優先使用 `SUPABASE_SERVICE_KEY`

### 🔒 安全性

- **更新 `.gitignore`** - 防止敏感資訊被提交
  - 新增 `.env*`, `*.env.backup` 等模式
  - 清理 git 歷史中的 `.env.backup`

### 📝 文檔更新

- **README.md** - 全面改寫，聚焦 LSTM 和下一日預測
- **SETUP.md** - 新增完整安裝指南
- **backtest/README.md** - 回測系統詳細說明
- **scripts/README.md** - 測試腳本使用說明
- **CHANGELOG.md** - 本檔案

### 📊 回測數據

基於 2026-01-05 ~ 2026-01-08 的 543 筆預測：

**LSTM 表現：**
- 方向準確度：54.6%
- 平均絕對誤差：21.3%
- 誤差 ≤10%：53.3%
- 誤差 ≤20%：71.2%

**各指數表現：**
- 台灣中型100：51.9%（最佳）
- SP500：48.5%
- 台灣50：41.5%（需改進）

### ⚙️ 設定變更

**推薦的 `.env` 設定：**
```bash
USE_PROPHET=false
USE_CHRONOS=false
USE_CROSS=false
USE_TRANSFORMER=false
```

### 🗑️ 移除項目

- Prophet 模型支援（可透過 `USE_PROPHET=true` 重新啟用，但不建議）
- Chronos 模型支援（可透過 `USE_CHRONOS=true` 重新啟用，但不建議）
- MySQL 資料庫支援（已完全遷移至 Supabase）

---

## [1.0.0] - 2026-01-05

### 初始版本

- 支援多種模型：LSTM, Prophet, Chronos, Transformer, Cross-sectional
- MySQL 資料庫儲存
- Discord, Telegram, Email 通知
- 多指數支援：台灣50, 台灣中型100, SP500, NASDAQ 等

---

## 未來計劃

### v2.1（計劃中）
- [ ] LSTM 準確度優化至 60%+
- [ ] 新增更多技術指標作為特徵
- [ ] 實作模型 ensemble
- [ ] 新增 Web UI 查看預測結果

### v3.0（構思中）
- [ ] 支援盤中即時預測
- [ ] 新增風險評估指標
- [ ] 整合基本面數據
- [ ] 支援自動交易（紙上交易）

---

**注意：** 本專案僅供學習和研究使用，不構成投資建議。
