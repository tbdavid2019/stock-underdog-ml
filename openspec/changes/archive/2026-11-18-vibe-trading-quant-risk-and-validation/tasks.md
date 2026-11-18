# Tasks

## 1. Phase 1 (價格底座與純量化核心): 多市場日 K 載入、Risk X-Ray、統計檢定與 ADMM 優化器

- [x] 1.1 於 `data/duckdb_manager.py` 新增離線 `global_daily_bars` 表與正確 `WHERE date <= ? ORDER BY date DESC LIMIT ?` 反轉對齊之唯讀載入 helper（整合 `tw_daily_bars`、`global_daily_bars` 與本地磁碟快取備援，全程零網路連線），並於 `data/fetcher.py` 批次下載路徑同步寫入海外調整後收盤價；執行 `python -m unittest test/test_local_bars_loader.py` 驗證最近視窗排序正確、缺漏標的列入 `unavailable_tickers` 且零外部連線
- [x] 1.2 新增零網路 I/O 之 `backtest/risk_xray.py`，實作雙口徑波動率（`annualized_volatility_total` 與 `annualized_volatility_risk_sleeve`）、支援負對沖值之歐拉波動貢獻拆解（`sum(PCV) == 1.0` 且 `sum(CV) == sigma_total`）、風險槽 HHI、分位數階梯樣本門檻（`N >= 60` 算 95% VaR/CVaR、`N >= 100` 算 99% VaR/CVaR）、台股原始收盤價分割異常跳空過濾守衛、跨市場 `local_currency_return_proxy` 雙警告（未避險 FX + 日曆錯位）與嚴格 `allow_nan=False`；執行 `python -m unittest test/test_risk_xray.py` 驗證現金稀釋恆等式、45 日視窗隱藏 99% 尾端指標、1拆4 斷崖過濾與跨幣別警告
- [x] 1.3 新增純函式模組 `backtest/statistical_validation.py`（與單筆撮合 `simulation.py` 解耦，接入驗證評估摘要器），以固定 `np.random.default_rng(seed)` 實作對數日報酬 Rademacher 符號翻轉檢定（`p_value_return`）、交易順序置換最大回撤檢定（`p_value_max_drawdown`）、Circular Block Bootstrap 年化 Sharpe 95% 信賴區間，以及零變異（`std <= 1e-12` 回傳 `degenerate_zero_variance`）與小樣本守衛；執行 `python -m unittest test/test_statistical_validation.py` 驗證同種子位元級重現性、全零/常數序列無 `NaN` 除零與小樣本保護
- [x] 1.4 新增純 NumPy 無重型求解器依賴之 `backtest/turnover_optimizer.py`，實作對角線收縮共變異數 + ADMM 非對稱買賣換手軟閾值（`gamma_buy` / `gamma_sell` 支援台股 `0.3%` 賣方證交稅）+ Dykstra 可行域投影 + 最終可行裁剪（預設 `max_single_weight = 0.10` 對齊 `RiskPolicy`，步長下界保護支援 `risk_aversion = 0`）；執行 `python -m unittest test/test_turnover_optimizer.py` 驗證高賣方懲罰壓低換手率、`risk_aversion = 0` 不發散、上下限殘差 `<= 1e-8` 及短序列無 `NaN` 降級
- [x] 1.5 更新 `backtest/README.md` 與 `docs/CHANGELOG.md`（同步 `CHANGELOG.md`）記錄雙口徑 Risk X-Ray、階梯樣本門檻、統計顯著性檢定與非對稱 ADMM 換手優化器之公式與限制

## 2. Phase 2 (資料管線韌性): 具冷卻與平行保護之隔離金絲雀探測道

- [x] 2.1 新增 `data/loader_canary.py` 與 CLI `scripts/check_loader_health.py`（支援 `--json`、`--sources`、`--force` 與預設 30 分鐘最小冷卻保護），以 `ThreadPoolExecutor(max_workers=4)` 與單源 `8s` 逾時對 TWSE、TPEx、yfinance、answerbook、Investing、Polymarket、BigGo 法說會與 arXiv 執行無快取隔離探測，分類三態健康碼並以 `tempfile` + `os.replace` 原子寫入白名單遮罩快照 `data/cache/canary_health_latest.json`；執行 `python -m unittest test/test_loader_canary.py`（mock `duckdb.connect` 斷言可寫連線數恆為 `0`）驗證冷卻重用、三態分類與 Token/內網 IP 100% 遮罩
- [x] 2.2 於 `run_daily.sh` 非阻塞串接金絲雀快照更新（探針失敗絕不影響主排程 exit code），並更新 `README.md` 與 `docs/CHANGELOG.md`（同步 `CHANGELOG.md`）說明 `python scripts/check_loader_health.py` 用法

## 3. Phase 3 (唯讀與受保護服務層): FastAPI 診斷端點、反 DoS 與 FastMCP 工具

- [x] 3.1 確認前置依賴 Wave 1（`core/duckdb_lock.py`、`core/netlimiter.py`、`watchlist_items`）就緒並內建 Fail-Closed 降級 shim；新增 `api/routes/risk.py` 掛載於 `api/main.py`：提供含每 IP 滑動視窗限流與 `BoundedSemaphore(2)` 保護之 `POST /api/v1/risk/xray`（`<= 100` 檔）、`GET /api/v1/watchlist/risk-xray`（表未就緒時回傳 `watchlist_unavailable`）、`POST /api/v1/risk/optimize-weights`（`<= 50` 檔），以及強制 API Key 認證與 `account_id` 正則檢驗之私有 `GET /api/v1/paper/accounts/{account_id}/risk-xray`（嚴格 `read_only=True` 讀取），並於 `api/routes/stats.py` 新增只讀本地快照之 `GET /api/v1/stats/canary-health`；執行 `python -m unittest test/test_risk_and_canary_api.py` 驗證超限 422、私有模擬帳戶未帶金鑰 401/403 防枚舉、公開 GET 零外網連線與 `allow_nan=False`
- [x] 3.2 於 `mcp_server.py`、`api/main.py`（`/mcp`、`/.well-known/mcp.json`、`/.webmcp/bridge.js`、`/llms.txt`、`/llms-full.txt`）、根目錄 `llms.txt` 與 `skills/stock-quant/SKILL.md`（對應 `/skill` 端點）同步註冊唯讀工具（`get_portfolio_risk_xray`、`get_canary_loader_health`）與新 REST API 文件，並於 `test/test_mcp_tools.py` 補齊唯讀工具驗證測試，同步更新 `scripts/README.md`、`docs/CHANGELOG.md`（同步 `CHANGELOG.md`）與 `README.md`

## 4. Phase 4 (操盤介面): 最小錨點風險 X 光健檢面板與統計顯著性標章

- [x] 4.1 新增獨立樣板 `api/templates/partials/risk_xray_panel.html` 與模組 `api/static/js/risk-xray.js`，以單一錨點（`<= 10` 行改動）接入 `tab_dashboard.html` 與 `app.js`，將當前 DuckDB 或匿名 `localStorage` 自選股代號陣列送往 `POST /api/v1/risk/xray`，支援雙口徑風險指標、負對沖 `PCV < 0` 視覺呈現、階梯樣本不足提示與統計顯著性標章；執行 `python -m unittest test/test_risk_xray_ui.py`（若環境裝有 Impeccable CLI 則同步執行 `impeccable detect`）驗證雙主題 WCAG AA/AAA 對比度（嚴禁白字疊 `#FFB010`）與零 `NaN` 破版
- [x] 4.2 更新 `docs/CHANGELOG.md`（同步 `CHANGELOG.md`）與 `README.md` 補充自選股組合風險 X 光健檢與統計顯著性標章使用說明

## 5. 全系統整合驗證

- [x] 5.1 執行全套單元回歸測試 `python -m unittest discover -s test -p 'test_*.py'`，確認全數測試 0 failed 且無暫存測試檔殘留

## Workflow follow-up

- 完成 Wave 2 實作與 OpenCode + Codex 雙重 Code Review 收斂後，執行 OpenSpec archive 歸檔。
