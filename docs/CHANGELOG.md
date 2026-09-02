# Changelog

所有重要的專案變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)。

---

## [2.2.1] - 2026-09-01

### Changed

- **首頁日報導覽優化**：新增「今日量化訊號」主標與日報狀態提示，讓每日計算結果成為首頁主要入口。
- **資料狀態更誠實**：首頁載入期間不再顯示固定的宏觀、曝險、資料筆數或科技股上限假值；改以讀取中或未提供狀態呈現。
- **查詢體驗改善**：策略結果載入改用符合卡片版型的 skeleton，並加入日報失敗後的可見錯誤與重新整理操作。
- **可及性與手機操作**：補充主要導覽、策略選擇、主題切換、重新整理與歷史軌跡控制的語意標籤與鍵盤焦點樣式。
- **初始化流程整理**：移除首頁 Vue 應用重複的 `mounted()` 定義，保留 WebMCP 註冊與資料載入在同一個初始化流程中。
- **頁尾文案聚焦產品**：移除頁尾對 DuckDB 與深度學習實作的宣傳句，保留品牌與技術提供者資訊。

## [Unreleased] - 2026-09-02

### 🌐 2md 繁體中文公司簡介與即時新聞浮動卡片 (2md Company Profile & News Hover)
- **整合 2md URL to Markdown 與 Search 服務** (`https://2md.aiurl.tw`, `https://2md.glsoft.ai`, `https://create360.ai`)：
  - 新增 `data/company_profile.py` (`CompanyProfileService`)，支援自 2md 與 Yahoo 股市實時提取個股之繁體中文公司名稱、核心營運業務、產業別、董事長、市值及最新即時新聞動態。
  - 後端內建 LRU 記憶體快取與 UTF-8 解碼，提供毫秒級 Hover Tooltip 回應速度。
  - 新增 REST API 端點 `GET /api/v1/market/company-profile` (`/api/v1/market/company-profile/{ticker}`)。
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
