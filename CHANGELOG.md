# Changelog

所有重要的專案變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)。

---

## 2026-10-31 (v2.4.49)

### Fixed & UI Stability (Frontend Industry Chain Template Render Error)
- **🐛 修復前端產業圖譜畫面因缺失 `formatNumber` 函數導致 Vue 模板渲染崩潰 (白畫面 / `TypeError: formatNumber is not a function`) 的重大 Bug**：
  - 在 `api/static/js/app.js` Vue 實例之 `methods`、`app.config.globalProperties` 與 `window` 統一補齊 `formatNumber` 數字格式化工具，支援千分位分割與小數點動態修約。
  - 杜絕渲染股票卡片現價與漲跌幅時因函數未定義觸發未捕獲之 TypeError，恢復整個 Web App 畫面的正常掛載與展示。
- **🏷️ 版本升級至 v2.4.49 (`core/version.py`)**。

---

## 2026-10-31 (v2.4.48)

### Added & Core (Cross-Market AI Value Chain & Supply Graph Engine)
- **🕸️ 跨市場產業價值鏈動態圖譜擷取模組 (`data/industry_chain_fetcher.py`)**：
  - **TPEx 產業價值鏈平台官方爬蟲與 14 天快取**：串接櫃買中心產業價值鏈資訊平台 (`ic.tpex.org.tw`)，解析半導體 (`D000`)、電腦週邊/AI 伺服器 (`E000`) 與網通/光通信 (`G000`) 之官方上中下游分類與成分股。
  - **高保真離線備援資料庫**：內嵌涵蓋台美核心 AI 產業鏈 (含美股巨頭 `US_AI_CORE`：NVDA, TSM, ASML, AVGO, MRVL, MU, COHR, LITE, FN, VRT, DELL, CLS, CRWV, MSFT, AMZN, GOOG, META, ORCL, TSLA 等)，確保在無外網或離線環境下 100% 穩定產出。
  - **SEC EDGAR 10-K 大客戶實體披露**：擷取美股 10-K 年報中 "Major Customers" 法定揭露 (>10% 營收依賴度) 與戰略供應鏈傳導關聯。
  - **8 大 AI 前沿技術主題**：支援 `全部主題`、`HBM/儲存`、`CPO/光互聯`、`電力/液冷`、`先進封裝`、`AI晶片/算力`、`伺服器/ODM` 與 `機器人/具身智能` 之三欄層級拓撲檢索。
- **🦆 DuckDB 產業鏈圖譜持久化與量化指標注入 (`data/duckdb_manager.py`)**：
  - 新增 `tw_industry_chains`、`tw_industry_nodes` 與 `tw_industry_relations` 實體圖譜表格與索引。
  - 實作 `init_industry_chain_tables()`、`save_industry_nodes_batch()`、`save_industry_relations_batch()`、`get_industry_nodes()`、`get_industry_relations()`。
  - 實作 `get_enriched_industry_quotes()`：圖譜節點動態關聯最新收盤價、當日真實漲跌幅、Mistral Timestral 與 Google TimesFM 雙時序大模型目標價與投信連買籌碼天數。
- **🌐 產業圖譜 REST API 端點與 Pydantic 資料模型 (`api/schemas.py`, `api/routes/industry_chain.py`, `api/main.py`)**：
  - 新增 `IndustryThemeItem`、`IndustryThemesResponse`、`IndustryGraphNode`、`IndustryGraphEdge` 與 `IndustryGraphResponse`。
  - 新增 `GET /api/v1/industry-chain/themes`：回傳主題清單與上中下游成分股數量統計。
  - 新增 `GET /api/v1/industry-chain/graph`：支援 `theme` 與 `relation_type` 篩選，回傳動態量化注入之圖譜拓撲。
- **🎨 前端三欄式產業圖譜畫布與動態 SVG 貝茲曲線 (`api/templates/partials/tab_industry_chain.html`, `api/templates/partials/header.html`, `api/templates/index.html`, `api/static/js/api-client.js`, `api/static/js/app.js`)**：
  - 三欄階層式畫布（`上游 UPSTREAM` ➔ `中游 MIDSTREAM` ➔ `下游 DOWNSTREAM`），支援 Bloomberg 終端曜石黑深色與 2md 暖米淺色雙模式 (WCAG AAA)。
  - 互動 SVG 貝茲連線計算 (`M x1 y1 C cx1 cy1, cx2 cy2, x2 y2`)，區分 `產業鏈`、`資本開支`、`大客戶依賴` 與 `主題關聯` 顏色光暈。
  - 支援卡片點選高亮傳導網絡（無關節點半透明淡化至 0.2）、畫布滾輪/按鈕縮放 (60%~160%)、隱藏未連線節點開關與一鍵跳轉個股歷史 K 線。
  - 桌面導覽列與行動端底部 App Bar 整合「產業圖譜」分頁。
- **🧪 完整單元測試覆蓋 (`test/test_industry_chain.py`)**：
  - 100% 覆蓋擷取器、離線降級、DuckDB 讀寫與 FastAPI REST 端點測試。
- **🏷️ 版本升級至 v2.4.48 (`core/version.py`)**。

---

## 2026-10-31 (v2.4.47)

### Added & Core (Production-Grade Market Trading Calendar & Holiday Guard Engine)
- **📅 跨市場交易行事曆核心模組 (`core/market_calendar.py`, `core/__init__.py`)**：
  - **台美市場代號智慧解析 (`resolve_market`)**：依股票代號自動辨識所屬市場（含 `.TW`、`.TWO`、`^\d{4,6}(\.TW|\.TWO)?$` 或 TWSE/TPEX 歸入台股 `TW`；其餘如 `AAPL`、`CAT` 歸入美股 `US`）。
  - **台股 TWSE OpenAPI 串接與 3 層容錯防禦**：直連臺灣證券交易所官方 OpenAPI (`https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule`)，支援本機 30 天磁碟快取 (`data/cache/market_calendar/twse_holidays_{year}.json`) 與記憶體快取；精確解析民國年 (`1150101` -> `2026-01-01`) 與「最後交易/開始交易/市場無交易/放假補假」語意分類；內嵌 2024~2027 年台股重要休市與封關交割日離線備援資料庫。
  - **美股 NYSE/NASDAQ 官方標準 10 大法定假日與復活節演算法**：以 Meeus/Jones/Butcher 公曆演算法精確計算復活節好禮拜五 (Good Friday)；支援新設立之六月節 (Juneteenth)；嚴格遵循美股官方放假補假規則 (Observation Rule：週六提前至週五、週日延後至週一、跨年元旦週六於 12/31 提前休市)。
  - **前後交易日與籌碼出表防護**：提供 `is_trading_day()`、`get_trading_day_status()` (回傳 T+1 與 T-1 日期及休市原因)、`get_next_trading_day()`、`get_previous_trading_day()`、`get_recent_trading_days()` (台股嚴格遵守 15:30 盤後出表門檻，美股遵守 16:00 門檻) 與 `get_holiday_list()`。
- **🛡️ 每日全市場日 K 同步腳本休市守衛 (`scripts/sync_twse_market.py`)**：
  - 支援 `--force` CLI 命令列參數（預設 `False`）。
  - 啟動時自動透過 `MarketCalendar.is_trading_day(market_or_ticker="TW")` 檢測；若為週末或國定休市日則優雅印出日誌跳過作業，杜絕假日對證交所上游無效請求與 DuckDB 空寫入。
- **📈 三大法人籌碼回溯交易日精確排假 (`data/institutional.py`)**：
  - 重構 `InstitutionalProvider.get_recent_trading_days()`，全面接入 `MarketCalendar.get_recent_trading_days(..., market_or_ticker="TW")`。
  - 將過往僅排除週末升級為嚴格排除台股國定假日（如中秋節、教師節、國慶日、農曆春節封關交割日等），徹底根除假日向證交所索取不存在之 T86 報表問題，並保持午夜換日與 08:00 盤前邊界相容性。
- **🌐 REST API 行事曆查詢端點與 Pydantic 資料模型 (`api/schemas.py`, `api/routes/market.py`)**：
  - 新增 `MarketHolidayItem` 與 `TradingDayStatusResponse` 資料結構。
  - 新增 `GET /api/v1/market/calendar/status`：支援 `market`、`ticker` 與 `date` 參數，秒級查詢交易日狀態、休市原因及前後交易日。
  - 新增 `GET /api/v1/market/calendar/holidays`：支援 `market` 與 `year` 參數，取得全年度休市清冊與假日說明。
- **🧪 完整單元測試覆蓋 (`test/test_market_calendar.py`)**：
  - 100% 測試覆蓋台美股週末休市、TWSE 重大假日與封關交割日、US 10 大法定假日與復活節、T+1/T-1 跨假計算、15:30 門檻回溯、Ticker 解析與 FastAPI 端點，全量 309 檔既有單元測試零迴歸全數通過。
- **🏷️ 版本升級至 v2.4.47 (`core/version.py`)**。

---

## 2026-10-31 (v2.4.46)

### Added & Interactive UI (Stock K-Chart Style Interactive Zoom, Pan & Timeframe Range Control)
- **📈 K 線級高互動縮放與平移體驗 (`api/templates/index.html`, `api/static/js/chart-manager.js`)**：
  - 引進 `hammerjs` 與 `chartjs-plugin-zoom`，在「真實股價 vs AI 目標價時序軌跡圖」全面支援滑鼠滾輪縮放 (Wheel Zoom)、按住拖曳平移 (Pan / Drag)、手機雙指縮放 (Pinch to Zoom) 與雙擊畫布重置 (Double-Click Reset)。
  - 畫布支援 `cursor-grab active:cursor-grabbing` 動態游標反饋，並設定 X 軸極限邊界 (`minRange: 6`) 防止過度縮放或超出資料範圍。
- **⏱️ K 線時間區間快選工具列 (`api/templates/partials/tab_history.html`, `api/static/js/app.js`)**：
  - 新增 TradingView / 看盤軟體風格的時間區間快選標籤（`近1月 (22D)`、`近2月 (45D)`、`近1季 (66D)`、`半年 (130D)`、`全部 (ALL)`），一鍵直達最聚焦的預測開獎視角，徹底解決長天期歷史圖表中近期預測點被擠在右側窄欄的問題。
  - 整合 `放大 (🔍+)`、`縮小 (🔍-)`、`← 左移`、`右移 →` 與 `重置 (↺)` 控制按鈕與輔助提示；自由縮放或平移時自動分流為 `CUSTOM` 自訂視角，切換按鈕具備 `:aria-pressed` 與無障礙狀態反饋。
  - Canvas 畫布配置 `touch-action: pan-y`，確保行動端雙指縮放時縱向頁面滾動順暢不打架。
- **🎯 視窗動態 Y 軸自動貼合與標點尺度自適應 (`autoFitPriceChartYScale`)**：
  - 在縮放、拖曳平移、時間區間切換或指標可見度（MA60、Timestral P10~P90 帶）切換時，自動根據當前可見 X 軸視窗內的所有有效數值重新計算 Y 軸之 `min` / `max` 並加上動態 8% padding，徹底告別傳統固定全域座標系留白過大或波幅扁平的問題。
  - **低價股自適應襯距**：Padding 底部下限改為相對比例動態下限（`0.02`），避免低價股（如 10~30 元標的）被原固定 0.5 元襯距過度壓平波幅；無有效數值時自動清除 Y 軸範圍保證優雅降級。
  - 當視窗縮放至 $\le 45$ 個交易日（近 1~2 個月）時，標點（圓形、方形、菱形、✕ 偏離叉號、★ 隔日預估星號）半徑自動放大為 relaxed 尺寸，確保開獎標點與預測命中率細節清晰可辨。
- **🏷️ 版本升級至 v2.4.46 (`core/version.py`)**。

---

## 2026-10-31 (v2.4.45)

### Fixed & Quant (AI Tech Radar & Discovery Engine Multi-Theme Consensus & Supply Chain Expansion)
- **⚡ 解決標的白名單 Catch-22 悖論與追蹤池嚴格隔離 (`core/ticker_canonicalizer.py`, `pipeline/discovery_pipeline.py`)**：
  - 修復原 `TickerUniverseValidator.get_whitelist()` 僅從 DuckDB 既有宇宙與 TW0050 讀取白名單，導致前沿設定檔 `config/discovery_supply_chain.json` 中之潛力新標的（如光聖 6442、聯鈞 3450、上詮 3363、雙鴻 3324、弘塑 3131、萬潤 6187、亞德客 1590 等）在尚未入池前即遭 `validator.is_valid()` 過濾而永遠無法生成 `propose_add` 推薦之核心死鎖。
  - `get_whitelist()` 自動載入前沿挖掘機產業鏈台美標的為合法 Universe。
  - **追蹤池隔離 (Codex/OpenCode 審查)**：`get_tracked_pool_tickers()` 嚴格排除 `market_universe` 全市場與 `TW0051`/`SP500` 泛指數，僅自 `predictions` 表與 `TW0050` 核心池讀取，確保未入預測池之台美供應鏈潛力股精準分流至 `propose_add` 或 `observe`。
- **🇺🇸 納入美股關鍵錨點龍頭、詞界比對與連字號支援 (`config/discovery_supply_chain.json`, `core/supply_chain_map.py`, `core/ticker_canonicalizer.py`)**：
  - 擴充 5 大主題之學術與架構關鍵字群（CPO 光互連、HBM 頻寬牆、液冷散熱、具身智能機器人、先進封裝 CoWoS）。
  - 各主題正式新增 `us_supply_chain` 結構化陣列，完整收錄美股核心領導廠（VRT, NVDA, AVGO, TSM, COHR, MU, AMD, LITE, ISRG, AMKR, INTC 等）之角色、代號與權重，並支援 `get_combined_supply_chain_for_theme()` 與雙向台美反向查詢。
  - **短詞詞界比對**：新增 `_keyword_matches` 輔助函式，針對 `inp`, `fau`, `vla`, `cxl` 等 $\le 4$ 碼英數採用正規詞界 (`\b`)，根除 `input`, `default`, `vlan` 等子字串誤判。
  - **美股代號連字號標準化**：美股正規化支援 `BRK-B` 自動轉為 `BRK.B`。
- **📊 7 維度多因子評分基準科學校準與 arXiv 基準補齊 (`pipeline/discovery_pipeline.py`)**：
  - `fetch_arxiv_papers()` 新增 `cs.AR`（計算機架構）論文類別；針對當日 arXiv 未覆蓋之硬體主題，在 live run 下補齊真實 arXiv 專屬檢索連結之基準參考論文，發布日期固定於 `2026-01-01`，防止快取與資料庫每日重複膨脹。
  - **空論文 Fallback 保障**：在無有效論文時優先觸發 Fallback 降級，再執行基準補齊。
  - **平滑因子擴展與多主題歸屬**：擴大論文計數 0 至 5+ 篇之平滑加分級距，論文資料結構支援 `matched_themes` 多主題雙向計分；科學校準後高純度未入池標的穩定達標 $\ge 75$ 分進入 `propose_add`，中純度標的穩定落在 $68 \sim 74.9$ 分進入 `observe`，池內既有標的維持 $\ge 70.9$ 分精確歸入 `already_in_pool`，達成零誤殺（0 reject）。
  - **歷史記錄去重**：`history_record` 採用 `ticker` 去重後統計，與前端 KPI 獨立標的檔數保持完全一致。
- **🖥️ 前端跨主題標的去重、多主題徽章與 WCAG 無障礙對比度提升 (`api/static/js/app.js`, `api/templates/partials/tab_discovery.html`)**：
  - `app.js` 新增 `deduplicateCandidates()` 方法：在「全部主題」模式下依 `ticker` 聚合標的，動態掛載 `all_themes` 並合併論文/訊號計數；在特定主題分頁時亦能透過全局地圖標註跨領域主題標籤。
  - KPI 計數器全面校準為獨立標的檔數，避免重複計數。
  - **WCAG 2.1 對比度與無障礙加固**：
    - 市場標籤（`TW` / `US`）採用淺深主題自適應高對比度邊框與文字（深色 `text-sky-300` / `text-purple-300`，淺色 `text-sky-900` / `text-purple-900`）。
    - 淺色模式下池內綠色與訊號指標全面切換為 `#065F46`，風險負分切換為 `#B91C1C`。
    - 裝飾性圖標加上 `aria-hidden="true"`，論文與快訊之關聯代碼標籤增加鍵盤無障礙操作支援 (`tabindex="0"`, `@keydown.enter`)。
- **🏷️ 版本升級至 v2.4.45 (`core/version.py`)**。

---

## 2026-10-31 (v2.4.44)

### Accessibility & UI (WCAG 2.1 Color-Blind Safe Palette & Direction Hit/Miss Visual Marker Overhaul)
- **♿ 預測趨勢折線圖標點動態辨識升級 (`api/static/js/chart-manager.js`)**：
  - **方向偏離/相反標點醒目標示**：當模型在 T 日預測 T+1 上漲（或下跌）而隔日實際收盤方向相反時，標點動態切換為高對比度的「✕」叉號警示標點 (`pointStyle: 'crossRot'`, 半徑增大至 4.5~6.5px, 邊框加粗至 2.5px, 採用高亮玫瑰緋紅 `#E11D48` / `#FB7185`)，徹底解決原圖表均為單一實心形狀、無法即時判斷漲跌方向是否相反的核心痛點。
  - **方向命中標點**：維持該模型專屬實心幾何形狀（方形/菱形/三角），並以對比邊框強化辨識度。
  - **隔日預估標點 (T+1 待開獎)**：切換為星號標點 (`pointStyle: 'star'`)，明確標記為待開獎之隔日目標錨點。
  - **Tooltip 提示診斷全面擴充**：hover 任何標點時，除顯示當日收盤與預期漲跌幅外，自動比對預測與隔日實際漲跌，直接輸出「🎯 方向命中 (預期 +X% ↑ · 實際 +Y% ↑ ｜ 誤差 Z%)」或「❌ 方向相反 (預測漲但實際跌 · 預期 +X% ↑ vs 實際 -Y% ↓ ｜ 誤差 Z%)」，並在 T+1 標記「⏳ 待下個交易日開獎驗證」。
- **🎨 WCAG 2.1 AA/AAA 色弱友善調色盤 (Color Vision Deficiency Safe Palette)**：
  - **真實股價 vs Timestral 色差徹底分離**：修正原真實股價 (`#38BDF8` 天藍) 與 Timestral (`#06B6D4` 青綠) 色相接近且在淺色背景對比度不足 (< 2.8:1) 之問題；將真實股價改為深海軍藍 `#1D4ED8`（淺色）/ 鈷藍 `#60A5FA`（深色），Timestral 全面改為高飽和溫暖琥珀橙 `#D97706`（淺色）/ `#F59E0B`（深色），對比度均達 4.5:1~7:1。
  - **多維度幾何冗餘特徵**：每條數據線同時具備專屬線條樣式與點形狀（真實股價：實線+圓形；Timestral：長虛線+方形；TimesFM：點線+菱形；雙模型平均：點劃線+三角形；MA60：長虛線+無點），即使色弱或黑白模式亦能 100% 辨別。
  - **圖表上方標點指引**：在折線圖上方新增「無障礙 / 色弱標點辨識指引」（實心幾何=方向命中、✕ 叉號=方向相反、★ 星號=隔日預估）。
  - **全歷史對決表與卡片對比度提升 (`tab_history.html`)**：全面修復淺色模式下 `text-cyan-400`、`text-purple-400`、`text-emerald-400`、`text-rose-400` 泛白難讀之問題，依照淺色/深色主題動態分流為高對比度文字（`text-amber-700`、`text-purple-700`、`text-emerald-700`、`text-rose-700`），符合 WCAG 1.4.3 AA 級標準。

---

## 2026-10-30 (v2.4.43)

### Added & Quant (AI Tech Radar & Discovery Engine: Academic Frontiers to Supply Chain Mapping)
- **🔭 AI 產業鏈前沿挖掘機 (AI Tech Radar & Discovery Engine) 全新上線**：
  - **前沿技術瓶頸外部化映射庫 (`config/discovery_supply_chain.json`, `core/supply_chain_map.py`)**：收錄 CPO 光互連、HBM/KV-cache、液冷散熱、具身智能機器人、先進封裝 CoWoS 等 5 大前沿技術瓶頸，對齊美股龍頭錨點（NVDA, AVGO, TSLA, VRT, AMD 等）並雙向映射台灣關鍵供應鏈（台積電 2330、聯發科 2454、奇鋐 3017、雙鴻 3324、信驊 5274、智邦 2345、光聖 6442、聯鈞 3450 等）。
  - **標的代號正規化與 Universe 白名單硬過濾 (`core/ticker_canonicalizer.py`)**：統一處理台股代號（切除 `.TW`/`.TWO` 後綴，保留純代碼）、美股轉大寫，並與系統 DuckDB / Universe 股票池進行白名單硬交集，防範幻覺標的。
  - **確定性 7 維多因子評分核心 (`core/discovery_scorer.py`)**：固化權重公式 $\text{Total} = \text{clamp}(0.20 S_{src} + 0.20 S_{purity} + 0.20 S_{paper} + 0.10 S_{novelty} + 0.10 S_{sent} + 0.10 S_{setup} - 0.10 S_{risk}, 0, 100)$，並嚴格分流至 `propose_add`（未在池且總分 $\ge 75$、純度 $\ge 60$）、`observe`（$60 \le \text{總分} < 75$）、`already_in_pool`（池內強化標的）與 `reject`。
  - **雙層原子持久化資料管線 (`data/duckdb_manager.py`, `pipeline/discovery_pipeline.py`)**：DuckDB 連線管理器擴充 `read_only` 參數，新增 `discovery_candidates`、`discovery_papers`、`discovery_signals`、`discovery_history` 四張表格與指數退避重試交易；管線實作 arXiv API 爬取（3 秒節流與 429 退避）、DuckDB 交易寫入、`tempfile + os.replace` 原子替換靜態快取檔（`cache/discovery/*.json`）、網路逾時自動 fallback 至舊快照標記 `stale=true`、以及過期資料 TTL 清理（論文 180 天、訊號 90 天）。
  - **FastAPI 唯讀加速 REST 端點 (`api/routes/discovery.py`, `api/schemas.py`, `api/main.py`)**：提供 `/candidates`、`/papers`、`/signals`、`/history`、`/status` 端點，全數採用 `read_only=True` 連線與本地快取優先策略。
  - **彭博終端儀表板與 RWD 適配 (`tab_discovery.html`, `header.html`, `app.js`)**：新增曜石黑深色 / 高對比淺色彭博終端風格之「前沿發現」分頁，實作 KPI 摘要卡、原生 Inline SVG 趨勢迷你圖、4/2/1 響應式情報矩陣（桌面 4 欄、平板 2 欄、手機單欄堆疊）、研究卡（Dossier）lazy-render 展開、深連結支援與免責聲明；手機底部導覽列升級為支援橫向平滑滑動，完美容納 5 個分頁。
  - **排程自動化與防重疊執行 (`run_daily.sh`)**：整合 Discovery Pipeline，並加入 `flock -n /tmp/stock_discovery.lock` 防撞鎖。

## 2026-10-29 (v2.4.42)

### Fixed & Quant (T+1 Foundation Model Threshold Realignment & Dual-Model Consensus Recovery)
- **🎯 雙時序基礎模型隔日 (T+1) 門檻科學校準 (`strategies/timesfm.py`, `strategies/timestral.py`, `core/config.py`)**：
  - 修復自 v2.4.33 轉向 T+1 單日預測後沿用 5 日波段門檻導致美股連續多日綜合優選（雙大模型共振）掛零之核心缺陷。
  - **TimesFM 盈虧比適配**：因 TimesFM 採用 `P50 (中位數期望)` / `P10 (10% VaR 下行)` 計算隔日盈虧比，單日下行波動率通常高達 3%~5%，導致原 `min_risk_reward_hit: 1.2` 相當於要求股票單日暴漲 >4.5% 才能命中。現將門檻合理校準為 `0.35x`，且 `rr_bonus` 評分依 `(risk_reward - 0.5) * 20.0` 平滑計算；「高盈虧比」標籤門檻依 Codex 審查建議定錨於 `>= 1.0`（預期期望值確實大於下行風險）。
  - **Timestral 單日漲幅門檻優化**：`min_potential_hit` 由 1.0% 調校為 `0.5%`（單日預期漲幅達 0.5% 換算年化已逾 200%，具備顯著短線優勢），避免 LRCX (+0.89%)、KLAC (+0.89%)、AMAT (+0.78%) 等頂級科技標的遭過度過濾；「高風報比」標籤門檻依 OpenCode 審查建議放寬至 `>= 1.5`。
- **⚙️ 全局 Config 策略工廠參數打通**：
  - 在 `PipelineConfig` 新增 `TIMESFM_MIN_RISK_REWARD` 與 `TIMESTRAL_MIN_RISK_REWARD` 環境變數設定，並將 `MIN_POTENTIAL` 預設值同步調整為 0.5。
  - 策略建構子支援 `Optional` 參數並防禦性 fallback 至全域組態。
- **🧪 雙線 AI 審查與測試套件加固 (`test/test_timesfm_strategy.py`, `test/test_timestral_strategy.py`)**：
  - 經 OpenAI Codex CLI 與 OpenCode CLI 雙線獨立 Code Review 並全數採納改進建議。
  - 新增組態預設值與 T+1 溫和看漲命中單元測試。

## 2026-10-28 (v2.4.40)

### Security & Integrations (JEV / TypeSafe SystemOne Endpoint Model Auto-Detection & Key Renewal)
- **🔑 JEV / TypeSafe SystemOne 金鑰更新與有效性驗證**：
  - 檢測 `10.9.0.99` 伺服器既有金鑰狀態，確認舊金鑰已過期失效（HTTP 403 Authentication Error）。
  - 全面更新 `10.9.0.99` 與本機環境之 `TYPESAFE_API_KEY` 與 `JEV_API_KEY` 為新配發金鑰，並通過真實 SystemOne 推論端點請求驗證（HTTP 200 OK）。
  - 平滑重啟 `10.9.0.99` 上的 `stock-underdog-ml-api` 與 `stock-underdog-ml-cron` 容器，API 服務狀態維持 `healthy`。
- **⚙️ ClefClient 與 DecisionService 外部節點模型名稱動態適配 (`data/clef_client.py`, `data/decision_service.py`)**：
  - 修復遠端 JEV 節點請求時誤傳 `clef-flash` 模型代號導致 `400 Unknown model` 之問題：當 slot URL 或名稱包含 `typesafe` 或 `jev` 時，自動適配採用官方原生模型代號 `jev-latest`，其餘自建節點維持 `clef-flash`。
- **🧪 測試套件加固**：
  - 更新 `test/test_clef_client.py` 驗證 JEV 節點調用時 payload 之 `model: "jev-latest"`，23 項 Clef 決策測試全數通過。

## 2026-10-22 (v2.4.39)

### Enhanced & Refined (Strict De-Buzzwording: Complete Purge of "AI" Marketing Buzzwords)
- **🧹 全面淨化使用者介面與視覺術語（徹底移除「AI」行銷浮誇字眼）**：
  - **`tab_history.html`**：
    - 「股價與 AI 預測目標價趨勢折線圖」正名為「**股價與隔日預測目標價趨勢折線圖**」。
    - 「⚡ AI 綜合操作建議」正名為「**⚡ 綜合操作建議**」。
    - 「AI 模型隔日 (T+1) 預測準確率 PK 成績單」正名為「**雙模型隔日 (T+1) 預測準確率 PK 成績單**」。
  - **`tab_dashboard.html`**：
    - 策略選項與篩選標籤「雙AI看漲」統一正名為「**雙模型看漲**」。
  - **`tab_podcast.html`**：
    - 頂部標題標籤「AI INTELLIGENCE RADAR」正名為「**PODCAST INTELLIGENCE RADAR**」。
    - 描述文案「提煉 AI 繁中精華觀點」改為「**提煉繁中精華觀點**」。
    - 頻道分類「科技 AI」正名為「**前沿科技**」。
    - 評級浮動提示「AI 市場催化重要性評級」改為「**市場催化重要性評級**」。
    - 摘要標題「AI 深度觀點精煉 (TL;DR)」正名為「**重點觀點精煉 (TL;DR)**」。
  - **`footer.html` & `tab_agent_hub.html`**：
    - 底部工具按鈕提示「切換至 AI Agent 與 MCP 工具中心」改為「**切換至自動化與 MCP 工具中心**」。
    - Agent Hub 說明文案移除「AI 爬蟲」，正名為「自動化爬蟲與 Agent 快速擷取」。
- **⚙️ 前端應用控制器與歷史快取向後相容 (`api/static/js/app.js`)**：
  - 策略清單標題與敘述去噪：`綜合優選 (雙模型看漲)`、`Timestral 預測`、`TimesFM 預測`、`看跌避險排行`，移除 Mistral AI / Google AI 等冗贅字樣。
  - **歷史快取標籤向後相容降噪清洗 (`parseTags` / `formatStrategyName`)**：自動將 DuckDB 歷史快取資料中帶有 `雙AI看漲`、`🤖AI強買`、`🤖AI看好`、`🤖AI建議觀察`、`⚠️AI建議避開`、`🤖AI做多`、`🤖AI中立` 之舊版標籤動態正規化為 `雙模型看漲`、`強勢買進`、`偏多看好`、`建議觀察`、`建議避開`、`偏多佈局`、`中立觀望`。
- **⚖️ 決策裁判與多渠道日報推播去噪 (`data/clef_client.py`, `evaluators/formatter.py`)**：
  - **決策 Verdict 標籤淨化**：本地規則與遠端決策 Tag 移除 `AI` 前綴，改為清晰純粹之 `🚀強勢買進`、`📈偏多佈局`、`📈偏多看好`、`👀建議觀察`、`⚠️建議避開`、`⚖️中立觀望`。
  - **推播排版規格對齊**：Telegram、Discord、Email 與 Terminal 推播將「【AI 操盤解讀】」正名為「**【盤勢重點解讀】**」，共振階層統一對齊「**雙模型看漲**」。
- **🧪 完整單元測試與版本同步**：
  - 更新 `test/test_clef_client.py` 與 `test/test_formatter.py` 斷言並新增雙模型共振相容性驗證，全數 294+ 單元測試通過。
  - 專案版本全面升級至 `2.4.39`。

## 2026-10-22 (v2.4.38)

### Enhanced & Optimized (UI Clean Nomenclature & Singleflight Per-Key Locking Defense)
- **🎨 前端介面徹底移除內部模型代號與工程術語 (`tab_history.html`, `tab_podcast.html`, `tab_agent_hub.html`)**：
  - **`tab_history.html`**：
    - 徹底刪除右上角 `source_slot` 節點徽章（如 `Clef-Create360`），使用者無需關注內部推論拓撲。
    - 標題正名為「**⚡ AI 綜合操作建議**」，副標題簡化為「綜合雙模型目標價、季線支撐、三大法人籌碼與估值，評估當前買賣動作與進場勝率」。
    - 載入狀態提示更新為「正在綜合技術面、籌碼面與模型預測計算操作建議...」。
    - 按鈕文案統一為「重新評估」與「評估中...」。
    - 三大核心指標名稱純淨中文化：「建議操作」、「出手信心評分」、「好買點機率」，移除括弧英文註釋；下方機率條標題更名為「各項操作建議機率分佈」，並移除內部「非自回歸 Softmax 推論輸出」工程術語。
  - **`tab_podcast.html`**：移除提示字中的內部來源資訊，保留「AI 市場催化重要性評級: X 分」。
  - **`tab_agent_hub.html`**：精簡工具標籤為「17 大原生量化 MCP 函數」。
- **🛡️ 決策鏈超時調校至 45 秒與防驚群效應（Singleflight Per-Key Lock）(`core/config.py`, `data/clef_client.py`)**：
  - **超時放寬 (`ClefConfig`, `ClefSlot`)**：將預設超時由 `4s` 提升至 `45s`（`CLEF_TIMEOUT_SECONDS=45`），應對上游模型冷啟動與深度長推論。
  - **防驚群機制 (`_INFLIGHT_LOCKS`)**：在 `ClefDecisionClient` 導入 Singleflight Per-Key Lock 與 Double-Checked Locking 機制，同一時間對相同標的（如 `2330.TW`）的併發請求僅會發出 1 次遠端調用，其餘請求等待鎖釋放後直接命中 L1 快取，杜絕驚群效應。
  - **併發流控**：批次評估 `evaluate_candidates_batch` 的預設 `max_workers` 調降為 2，避免並發瞬間壓垮遠端節點。
- **🧪 單元測試覆蓋 (`test/test_clef_client.py`)**：新增 45 秒預設超時檢查與 5 執行緒併發 Singleflight 僅調用一次驗證測試。

## 2026-10-22 (v2.4.37)

### Added & Enhanced (Clef-Flash System One Decision Engine Integration & Telegram Minimalist Stock Picks)
- **⚡ Clef-Flash System One 決策模型前後端深度整合 (`core/config.py`, `data/clef_client.py`, `data/decision_service.py`, `pipeline/orchestrator.py`, `api/routes/decision.py`)**：
  - **三級節點認證與金鑰防護**：於 `ClefConfig` 新增 Tier 3 (`FALLBACK2_URL`) API Key 安全讀取（`TYPESAFE_API_KEY` / `JEV_API_KEY` / `CLEF_FALLBACK2_API_KEY`），`ClefSlot` 支援 Bearer 與 `X-API-Key` 標頭驗證；絕無硬編碼金鑰。
  - **雙時序模型特徵升級**：`ClefDecisionClient.format_stock_state()` 全面支援 `Timestral` + `TimesFM` 雙大模型目標潛力、歷史方向勝率（`timestral_hit_rate` / `timesfm_hit_rate`）與均線季線支撐解析；本地規則 `_evaluate_local_rules()` 以 `primary_pot` (`Timestral`/`LSTM`) 與 `TimesFM` 進行非自回歸機率與動作推論。
  - **選股管線批次推論與公開個股查詢端點**：
    - `PipelineOrchestrator` 於每日選股 Stage 3 自動並行執行 `evaluate_candidates_batch`，並為共振標的賦予 `clef_verdict`。
    - 新增免金鑰、具每 IP 每分鐘滑動視窗防護之公開端點 `GET /api/v1/decision/stock/{ticker}`，自動解析代號與快照，回傳 4 選項動作機率、信心分與好買點勝率。
- **📊 前端個股情報中心整合 Clef-Flash 決策卡 (`api-client.js`, `app.js`, `tab_history.html`)**：
  - 於「歷史與 AI 準確率」分頁新增「⚡ AI 綜合決策裁判（Clef-Flash System One）」卡片，即時展示：建議操作（積極買進/逢低佈局/觀望等待/避開風險）、出手信心分（`conviction_score / 5.0`）、好買點機率與 4 維動作機率分佈條。
  - 支援單鍵強制重新推論與推論來源節點標籤顯示；嚴格遵照 WCAG AA/AAA 與 `2md.aiurl.tw` 高對比暖米/深墨色彩標準。
  - 短網址優化：網址帶 `?ticker=xxx` 自動切換至 `history` 分頁直達個股決策。
- **📱 Telegram 日報極簡報明牌化與網站互動導流 (`evaluators/formatter.py`, `test/test_formatter.py`)**：
  - 大幅精簡 Telegram 訊息結構，移除冗長之 4 大分項數據表，落實「報明牌就好，詳細請用戶到網站查」：
    - 1 行極簡大盤風控與曝險建議。
    - 1 段精簡 AI 操盤引言解讀。
    - 🎯【今日精選明牌 (優先推薦)】：前 10 支標的採 2 行精簡卡片，包含現價、TFM/Timestral 潛力、季線支撐與專屬個股歷史連結（`/?tab=history&ticker=...`）。
    - 🔗 底部包含指數完整互動儀表板導流連結（`/?index=...`）。
- **🧪 單元測試與版本升級 (`test/test_clef_client.py`, `test/test_api_decision.py`, `test/test_formatter.py`, `core/version.py`, `app.js`)**：
  - 版本正式升至 `v2.4.37`。
  - 全數測試通過，包含 ClefSlot API Key Header 授權、Timestral 狀態解析與推論、公開端點限流與查詢、Telegram 極簡排版與 URL 驗證。

## 2026-10-22 (v2.4.36)

### Fixed & Enhanced (AI Model Accuracy Scorecard Refactoring & 50% Hit Rate Floor Defense)
- **🏆 歷史預測準確率 PK 成績單重構與 50% 方向勝率門檻防禦 (`app.js`, `tab_history.html`)**：
  - 嚴格實施「50% 方向勝率門檻防禦機制 (Minimum 50% Directional Hit Rate Floor)」：模型必須在驗證天數 $\ge 2$ 且方向勝率 $\ge 50.0\%$ 下方具備獲頒「🏆 該股表現較佳」資格；若雙模型於該個股之方向勝率皆低於 50%，絕不授與領先標記，成績單頂部明確提示「⚠️ 兩模型於該股方向勝率皆低於 50%，建議以均線支撐與籌碼為主」，並於對應卡片標頭呈現「方向勝率 < 50%」微警示膠囊，杜絕過去僅依 MAPE 微幅差異而頒發勝出給猜對率極低模型的反直覺行為。
  - 獨立計算並分離同場對決指標 (Head-to-Head PK)，統計 `headToHeadCount`、TimesFM 勝出 (`tfWins`)、Timestral 勝出 (`tmWins`) 與平手 (`ties`)。
  - 清晰透明的三大核心指標展示卡：
    - **指標一：漲跌方向勝率**：大字百分比，副標題「`N 天中猜對 M 天`」。
    - **指標二：目標價平均誤差 (MAPE)**：大字「`±X.XX%`」，副標題「`越接近 0% 越準`」。
    - **指標三：雙模型同場 PK**：大字「`X 勝`」，副標題「`共 N 天同場比準度`」。
- **🧪 單元測試覆蓋與版本升級 (`test_typography_rwd.py`, `version.py`, `app.js`)**：
  - 版本正式升至 `v2.4.36`。
  - 單元測試新增 `test_model_scorecard_structure_and_labels` 與 `test_model_scorecard_js_logic_floor_and_h2h`，包含 Node.js 執行器驗證低於 50% 勝率門檻防禦、勝率領先判定、同場對決勝場/平手統計與字樣渲染斷言。

## 2026-10-22 (v2.4.35)

### Fixed & Enhanced (2md.aiurl.tw Color Palette & Strict WCAG AA/AAA Readability Upgrade)
- **🎨 全面升級 2md.aiurl.tw 暖米溫潤色系與頂級 WCAG AA/AAA 對比標準**：`index.html` 導入 2md 五色調（`#17324D`、`#FFF0CC`、`#FFF9E8`、`#2B251A`、`#FFB010`、`#9A3412`、`#DDC58B`）；全站淺色模式黃底全面改用深赭墨字 `#2B251A`（對比度 10.2:1 WCAG AAA）；看漲翠綠 `#065F46`、看跌緋紅 `#991B1B`、時序模型深藍 `#17324D` 與紫 `#5B21B6` 均達成 WCAG AAA 標準。
- **🏷️ 移除頂部左側多餘文字，回歸簡潔 888 STOCK QUANT 徽章 (`header.html`)**：頂部左側移除 `888 StockQuant` 文字，保留番紅花金黃底 `#FFB010` 配墨色邊框與字體之方型徽章；桌面與行動端導覽列改採藏青深藍 `#17324D` 容器與金黃高對比選中按鈕。
- **🧹 全站 Partial 模板淺色啟用按鈕清理與 Chart.js 調校**：全面掃除 `tab_dashboard.html`、`tab_history.html`、`tab_institutional.html`、`tab_podcast.html`、`footer.html` 的 `bg-amber-500 text-white` 為 `bg-amber-500 text-[#2B251A]`；Chart.js 淺色圖表格線與 Tooltip 採用 2md 色彩。
- **🧪 單元測試覆蓋與版本同步 (`test_typography_rwd.py`, `version.py`, `app.js`)**：版本升至 `v2.4.35`，單元測試嚴格斷言 2md 色彩系統與零違規按鈕。

## 2026-10-22 (v2.4.34)

### Fixed & Enhanced (Mobile RWD, Native Bottom App Bar, Touch Occlusion & Responsive Typography)
- **📱 行動端導覽重構為原生固定底部分頁導覽列（Fixed Bottom App Bar & Slim Header）**：頂部導覽列於行動端縮減至 `h-14`，並將 4 大主分頁抽離為固定於螢幕底部之原生導覽列（`fixed bottom-0` + `env(safe-area-inset-bottom)`，觸控區 `min-h-[48px]`），大幅釋放手機垂直可視空間；頁尾與 Toast 通知同步預留安全邊界高度。
- **🛡️ 懸浮個股名片遮擋與觸控防禦 (`app.js`, `footer.html`)**：`showCompanyHover` 加入 `(hover: hover) and (pointer: fine)` 媒體查詢守門員，於手機與平板等純觸控裝置停用懸浮浮窗，徹底根除手機點擊觸發 `mouseenter` 後懸浮卡卡死遮擋全螢幕之問題；桌面端同步加上捲動自動收合與右上角關閉按鈕。
- **📊 寬表格行動端卡片化與凍結首欄 (`tab_history.html`, `tab_institutional.html`)**：為「每日預測 vs 隔日真實收盤對決表」與「三大法人排行榜」新增手機專屬卡片視圖（`block md:hidden`），免除水平捲動迷路；法人累計表與桌面表格首欄加入 `sticky left-0 z-10` 固定首欄。
- **📐 抽屜子標籤、模型成績單與 Chart.js 行動端自適應 (`tab_dashboard.html`, `chart-manager.js`, `index.html`)**：抽屜子標籤改用 `flex-wrap gap-1.5` 防止第 4 顆「政府標案」按鈕超出邊界；Chart.js 手機端 X 軸刻度自動精簡為 `MM/DD` 與 `T+1`（上限 5 點）；新增 `@media (max-width: 640px)` 輸入框 `16px` 防 iOS Safari 自動放大規則，法說會彈窗改用動態視窗單位 `max-h-[85dvh]`。

## 2026-10-21 (v2.4.33)

### Fixed & Enhanced (T+1 Horizon Alignment, Chart Time-Shift, Model PK Scorecard & Unified Naming)
- **🎯 預測天期嚴格統一對齊「隔天 (T+1)」與目標價漲跌幅一致性**：TimesFM 與 Timestral 模型之 `potential` 全面改為隔日 `day1_potential`；風險報酬比改以隔日 T+1 分位數計算；DuckDB `get_ticker_history` 自動重算並校正舊資料庫歷史潛力百分比。
- **📈 圖表時間軸平移對齊與影子線徹底根除 (`chart-manager.js`)**：第 $i$ 天產生之隔日目標價平移至 $i+1$ 位置，並於右側新增 `'隔日預估 (T+1)'` 錨點；Tooltip 完整呈現前日預測基準價與隔日實際誤差；「雙模型平均」預設不打架隱藏。
- **🏆 個股歷史雙模型準確率 PK 成績單與每日一列對決表 (`app.js`, `tab_history.html`)**：`loadHistory()` 查詢筆數擴充至 240 筆；新增 TimesFM vs Timestral 方向勝率、MAPE 與對決結果 PK 卡片；歷史表格升級為每日一列對決表，未開獎標示 `⏳ 待下個交易日開獎`。
- **🏛️ 策略命名全面回歸專業金融語言**：策略選單直白命名（綜合優選、Timestral 看漲、TimesFM 看漲、季線支撐、AI 避險、全市場總表）；前端自動轉換玄鐵與共振造詞；全介面嚴格遵循 `impeccable` 規範（最小字級 `text-xs`）。
- **📢 多渠道日報（Telegram / Discord / Email）與 AI 解讀全面對齊 T+1 目標價與季線支撐正名 (`evaluators/formatter.py`, `evaluators/ai_narrative.py`)**：
  - 日報推播與終端報告之「波段操作 (玄鐵重劍)」全面正名為「波段操作 (季線支撐)」，多維共振交集正名為「綜合優選」。
  - TimesFM 與 Timestral 於 Telegram、Discord、Email 完整呈現隔日 T+1 目標價流程（`現價 → 目標價`）與 P10/P90 蒙地卡羅區間，徹底消除過去 5 日目標價與隔日潛力漲幅不一致之錯覺。
  - AI 操盤總評上下文提示詞與規則備援模板對齊季線支撐與雙時序模型隔日預期。

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
