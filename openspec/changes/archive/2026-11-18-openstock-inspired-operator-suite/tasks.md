# Tasks

## 1. Phase 1 (P2 安全與併發底座): DuckDB 跨進程鎖、上游熔斷守衛與環境變數健檢

- [x] 1.1 新增 `core/duckdb_lock.py`（基於共用卷 `<db_path>.write.lock` 之 `fcntl.flock` 排他寫鎖與逾時 `DuckDBBusyError`），重構 `data/duckdb_manager.py` 將 `_init_schema()` 與寫入路徑納入跨進程鎖保護，並移除 `read_only=True` 失敗時反向升級為 `read_only=False` 之邏輯；先新增並執行 `python -m unittest test/test_duckdb_crossprocess_locking.py`（以 `multiprocessing` 模擬雙進程讀寫競爭）驗證零死鎖與優雅降級
- [x] 1.2 新增 `core/upstream_guard.py`（逗號分隔多金鑰輪詢池、單金鑰 `BoundedSemaphore` 併發閘門、60 秒失敗熔斷冷卻與零明文日誌遮罩）及 `core/netlimiter.py`（統一 `CF-Connecting-IP` ➔ `X-Real-IP` ➔ `client.host` 限流器與 Fail-Closed 金鑰驗證），一併修復 `api/routes/decision.py` 之 `/api/v1/decision/health` 加上 60 秒快取、限流與上游 URL/錯誤遮罩，並以 `python -m unittest test/test_upstream_guard.py test/test_decision_security.py` 驗證通過
- [x] 1.3 將 `KeyPoolGuard` 接入外部抓取模組（`data/investing_service.py`、`data/polymarket_service.py`），並新增 `scripts/check_env.py`（支援 `--json`、必填/選填降級說明/`WATCHLIST_API_KEY` 生產模式提醒/已棄用變數檢查、100% 遮罩敏感值）串接至 `scripts/install_new_machine.sh` 與 `.env.example`，執行 `python -m unittest test/test_check_env.py` 驗證通過
- [x] 1.4 更新 `docs/CHANGELOG.md`（同步 `CHANGELOG.md`）與 `README.md` 記錄跨進程鎖保護、多金鑰熔斷池與 `python scripts/check_env.py` 用法，確認文件不含任何敏感資訊

## 2. Phase 2A (P0 後端): DuckDB 自選股 Fail-Closed API 與跨市場即時交易時段（含日股午休）

- [x] 2.1 於 `data/duckdb_manager.py` 新增 `watchlist_items` 表與讀寫方法（代號正規化、上限 100 檔、`note <= 500` 字元、唯讀連線附帶最新預測摘要），以 `python -m unittest test/test_watchlist_duckdb.py` 驗證持久化與邊界拒絕
- [x] 2.2 新增 `api/routes/watchlist.py`（`GET / POST / DELETE /api/v1/watchlist` 與批次匯入，`DEPLOYMENT_MODE=production` 下強制 Fail-Closed 驗證 `WATCHLIST_API_KEY`，鎖忙碌回傳 HTTP 503）並於 `api/main.py` 掛載路由、於 `mcp_server.py` 註冊唯讀 `get_watchlist` 工具，以 `python -m unittest test/test_watchlist_api.py` 驗證公開唯讀、未帶金鑰寫入 401 及合法金鑰 CRUD
- [x] 2.3 於 `core/market_calendar.py` 新增 `get_live_sessions_summary()`（支援 `pre_market | regular | lunch_break | after_hours | closed` 五態，涵蓋東京 TSE 11:30–12:30 午休、美股夏冬令與提前收盤日）與交易日感知新鮮度 helper，掛載 `GET /api/v1/market/calendar/sessions`，並以 `python -m unittest test/test_market_live_sessions.py` 驗證東京 12:00 `lunch_break`、盤前、週末與國定連假新鮮度判定

## 3. Phase 2B (P0 前端): `⌘K` 全域指令面板、最近個股列、雙軌自選股與開休市時鐘

- [x] 3.1 升級 `api/main.py:get_asset_fingerprint()` 掃描 `api/static/` 與 `api/templates/` 全檔案最新 `mtime`；新增 `api/templates/partials/command_palette.html` 並擴充 `api/static/js/app.js` 與 `api-client.js` 實作 `⌘K / Ctrl+K` 全域搜尋指令面板（短字元本地過濾、`>= 2` 字元防抖查詢、`↑/↓/Enter/Esc` 導覽）與「最近瀏覽/釘選個股快捷列（上限 10 檔）」，以 `python -m unittest test/test_command_palette_and_tabs_ui.py` 驗證通過
- [x] 3.2 於 `api/templates/partials/tab_dashboard.html`、`tab_history.html` 與個股抽屜加入 `>= 44px` 星號收藏鈕（有金鑰同步 DuckDB、無金鑰自動退回 `localStorage`）、維度 1 `★ 我的自選 (N)` 篩選膠囊與純文字安全驗證之 `?watchlist_share=`（正則白名單、上限 100 檔、禁 `v-html`）分享匯入橫幅，並於 `api/templates/partials/header.html` 整合三市場即時開休市/午休倒數與交易日 `as_of`/`stale` 標章，執行 `python -m unittest test/test_typography_rwd.py`（若環境裝有 Impeccable CLI 則同步執行 `impeccable detect`）驗證雙主題 WCAG 對比度與 RWD 無折行
- [x] 3.3 更新 `docs/CHANGELOG.md`（同步 `CHANGELOG.md`）與 `README.md` 說明 `⌘K` 指令面板、雙軌自選股（含 `WATCHLIST_API_KEY` 保護說明）與跨市場交易時段指示器

## 4. Phase 3 (P1 決策閉環): 個股多源訊號一致度卡與原子冷卻量化告警

- [x] 4.1 新增 `data/signal_alignment.py`（實作模型、台股籌碼、30 日 Podcast、前沿雷達之 `+1 / 0 / -1 / None` 確定性公式與海外股籌碼 `applicable=False` 排除），於 `api/routes/predictions.py` 掛載 `GET /api/v1/predictions/alignment/{ticker}` 並於 `mcp_server.py` 註冊唯讀 `get_signal_alignment` 工具，以 `python -m unittest test/test_signal_alignment.py` 驗證一致看多、背離警示、美日股籌碼不扣分與過期排除
- [x] 4.2 於 `data/duckdb_manager.py` 新增 `quant_price_alerts` 表，實作 `data/alert_evaluator.py`（區分靜態目標價與動態滾動 `below_p10`/`above_p50`/`pullback_ma60`/`pullback_ma120`、`reference_model` 取值、以及先於 `flock` 交易寫入 `last_triggered_at` 成功後才呼叫 `notifier.py` 發送推播之原子防洗版機制），於 `api/routes/watchlist.py` 掛載 `/api/v1/alerts` 與強制金鑰驗證之 `POST /api/v1/alerts/check`、串接 `run_daily.sh`，並以 `python -m unittest test/test_quant_alerts.py` 驗證觸發、無 P10 拒建、未授權 401 與鎖失敗不洗版
- [x] 4.3 於個股分析抽屜整合「多源訊號一致度摘要卡」與「量化告警訂閱控制項」，以 `python -m unittest test/test_drawer_alignment_alerts_ui.py` 驗證 UI 結構與雙主題無障礙對比度，並同步更新 `docs/CHANGELOG.md`（同步 `CHANGELOG.md`）與 `README.md`

## 5. Phase 4 (P1 成長引擎): 獨立 `autoescape=True` 公開個股 `/s/{ticker}` SSR 頁與 Sitemap

- [x] 5.1 於 `api/main.py` 新增獨立 `jinja2.Environment(..., autoescape=True)` 之 `_render_ssr_template(template_name, context)` 與 `api/templates/ticker_seo.html`，掛載白名單驗證之 `GET /s/{ticker}`、`GET /sitemap.xml`（1 小時快取、實際交易日 `<lastmod>`）與 `GET /robots.txt`，實作 `< > &` Unicode 轉義之安全 JSON-LD、OpenGraph 與同業 Peers 導覽，並以 `python -m unittest test/test_public_ticker_seo.py` 驗證有效代號 200、惡意 `</script>` 字串 XSS 完全跳脫、無效代號 404 `noindex` 與合法 XML Sitemap
- [x] 5.2 更新 `llms.txt`、`docs/CHANGELOG.md`（同步 `CHANGELOG.md`）與 `README.md` 補充公開個股頁與 Sitemap 規格說明

## 6. 全系統整合驗證

- [x] 6.1 執行全套單元與跨進程回歸測試 `python -m unittest discover -s test -p 'test_*.py'` 及 `python scripts/check_env.py`，確認 0 failed 且無測試殘留暫存檔

## Workflow follow-up

- 完成 Wave 1 實作與 OpenCode + Codex 雙重 Code Review 收斂後，執行 OpenSpec archive 歸檔。
