# Tasks

## 1. 契約、設定與代號白名單 (Contracts, Whitelist & Schema)

- [x] 1.1 建立 `config/discovery_supply_chain.json`：定義美股錨點、前沿技術瓶頸（CPO、HBM/KV-cache、液冷散熱、具身智能、先進封裝）與台灣關鍵供應鏈標的映射字典。
- [x] 1.2 建立 `core/ticker_canonicalizer.py`：實現台美股代號正規化（如 `2330.TW` -> `2330`，美股轉大寫）與 Universe 白名單校驗過濾，並編寫單元測試 `test/test_ticker_canonicalizer.py` 驗證大小寫、後綴與無效代碼過濾。
- [x] 1.3 擴充 `api/schemas.py`：定義 Discovery 候選、論文、訊號、歷史趨勢與狀態之 Pydantic 資料模型與 JSON Schema 規格。

## 2. 供應鏈映射與 7 維度評分核心 (Core Scorer & Mapping)

- [x] 2.1 建立 `core/supply_chain_map.py`：載入外部設定檔，實作前沿瓶頸至台美股標的之雙向查詢與緩存，並編寫單元測試 `test/test_supply_chain_map.py` 驗證對照完整性。
- [x] 2.2 建立 `core/discovery_scorer.py`：實作確定性 7 維多因子評分公式（權重矩陣、[0, 100] 邊界限制）與 4 級分流邏輯（`propose_add`, `observe`, `already_in_pool`, `reject`），並編寫單元測試 `test/test_discovery_scorer.py` 驗證邊界值與分級判定。

## 3. 資料管線、原子快取與 DuckDB 持久化 (Pipeline, DB & Atomic Cache)

- [x] 3.1 擴充 `data/duckdb_manager.py`：定義 `discovery_candidates`、`discovery_papers`、`discovery_signals` 與 `discovery_history` 表結構，提供 Transaction 批次寫入、連線重試與 TTL 資料清理方法。
- [x] 3.2 建立 `pipeline/discovery_pipeline.py`：實作 arXiv 爬取（3 秒間隔與 429 退避重試）、LLM 語意萃取、代號過濾、評分計算，以及 `tempfile + os.replace` 原子快取發布與異常 Fallback 快照機制。
- [x] 3.3 編寫管線與資料庫整合測試 `test/test_discovery_pipeline.py`，驗證交易寫入、快取原子替換、重跑冪等性與異常退避降級。

## 4. FastAPI REST 端點與快取服務 (API Endpoints & Status)

- [x] 4.1 建立 `api/routes/discovery.py`：以 `read_only=True` 連線與本地快取優先策略提供 `/candidates`、`/papers`、`/signals`、`/history` 與 `/status` 端點，支援條件過濾與分頁。
- [x] 4.2 於 `api/main.py` 掛載 discovery 路由，並編寫 API 整合測試 `test/test_api_discovery.py`，驗證各端點 HTTP 200 回應、stale 狀態標示與空資料處理。

## 5. 前端彭博終端儀表板與 RWD 適配 (Web UI & Navigation)

- [x] 5.1 建立 `api/templates/partials/tab_discovery.html`：以彭博深淺雙色主題實作 KPI 摘要卡、原生 Inline SVG 趨勢迷你圖、4/2/1 響應式情報矩陣（觀察候選、池內強化、arXiv 論文雷達、產業快訊）與免責聲明。
- [x] 5.2 更新 `api/templates/partials/header.html` 與 `api/templates/index.html`：手機端底部導覽列增加第 5 個 Tab 並支援橫向平滑滾動，前端支援個股研究卡 lazy-render 展開與深連結。
- [x] 5.3 驗證桌面端與手機端（Mobile RWD）之無縫渲染、鍵盤導覽無障礙性與 WCAG AA 對比度標準。

## 6. 排程自動化、觀測性與公開契約文件 (Scheduler, Flock & Docs)

- [x] 6.1 更新 `run_daily.sh`：整合 Discovery Pipeline，並加入 `flock -n /tmp/stock_discovery.lock` 防止重疊執行。
- [x] 6.2 更新 `docs/CHANGELOG.md`、`README.md` 與公開契約文件 `llms.txt`，說明 Discovery 前沿挖掘機功能與 API 規範。
