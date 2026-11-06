# Technical Design

## Context

實地盤點專案現有程式碼與平行推進中之 Wave 1（`openstock-inspired-operator-suite`），關鍵架構現況如下：
1. **既有風險與模擬模組**：`backtest/risk.py` 負責交易前限額守衛（預設單筆上限 `0.10`、相關群組聚類門檻 `0.80`）；`backtest/simulation.py` 僅提供單筆開盤撮合原語 `execute_open()` 與 Walk-Forward 切窗 `walk_forward_windows()`（並無完整組合回測迴圈）；`backtest/paper.py`（`PaperLedger`）儲存私有模擬帳戶持倉；`evaluators/forecast_performance.py` 與 `backtest/analyze_backtest.py` 負責彙整歷史驗證表現。
2. **歷史日 K 儲存現實落差**：`DuckDBManager` 目前僅有台股原始收盤價表 `tw_daily_bars`（未還原除權息與分割），而美股（如 `NVDA`）與日股（如 `7203.T`）在 `main.py` 執行時僅透過 `data/fetcher.py`（`auto_adjust=True`）寫入本地 `cache/stock_data/*.pkl`，尚未落入 DuckDB。此外，既有 `get_daily_bars_for_ticker()` 使用 `ORDER BY date ASC LIMIT ?`，若直接截取會誤拿最舊歷史而非最近視窗。
3. **跨 Wave 模組邊界**：`core/duckdb_lock.py`、`core/netlimiter.py` 與 `watchlist_items` 表皆屬 Wave 1 新增元件；Wave 2 必須明確宣告前置依賴並提供獨立測試/降級保護，同時將 `app.js` 與 `tab_dashboard.html` 的修改壓縮在最小錨點內。

## Goals / Non-Goals

**Goals:**
- 建立純本地唯讀多市場歷史價格載入契約（台股 `tw_daily_bars` + 海外 `global_daily_bars` / 本地磁碟快取備援），查詢強制 `WHERE date <= ? ORDER BY date DESC LIMIT ?` 反轉對齊，嚴禁公開 API 即時外呼 `yfinance`，並對未還原除權息/股票分割異常跳空提供偵測與防護。
- 實作零網路 I/O、100% 通過 `json.dumps(..., allow_nan=False)` 之 `backtest/risk_xray.py`、`backtest/statistical_validation.py` 與 `backtest/turnover_optimizer.py`。
- 嚴格保護私有模擬帳戶端點（Fail-Closed API Key 驗證 + 防帳戶枚舉），並對公開矩陣/優化計算端點施加維度上限、滑動視窗限流與 `BoundedSemaphore` 併發閘門。
- 實作具 30 分鐘冷卻保護、平行逾時控制、白名單結構與零生產 DuckDB 寫入之 `data/loader_canary.py`。

**Non-Goals:**
- **絕不引入任何真實券商實盤下單 SDK**，亦不引入 `cvxpy`、`ecos`、`osqp` 等重型 C/C++ 求解器套件。
- **絕不讓公開匿名 HTTP `GET` 端點即時觸發外部爬蟲**，亦不公開裸露未授權之 `PaperLedger` 私有模擬帳戶持倉。

## Decisions

### Decision 1: 多市場本地歷史日 K 契約與企業行為異常跳空守衛
- **決策**：
  1. **補齊海外離線日 K 表 `global_daily_bars`**：於 `DuckDBManager._init_schema()` 新增 `global_daily_bars(ticker VARCHAR, market VARCHAR, currency VARCHAR, date VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, is_adjusted BOOLEAN DEFAULT TRUE, updated_at TIMESTAMP, PRIMARY KEY (ticker, date))`。當 `data/fetcher.py` 批次抓取美股/日股行情（`auto_adjust=True`）時同步寫入該表；唯讀查詢 helper `load_local_close_series(tickers, as_of_date, lookback_bars)` 依序從 `tw_daily_bars`、`global_daily_bars` 及本地磁碟 `CacheManager.get_stock_data(..., max_age_hours=None)` 唯讀載入，**全程零網路連線**。查無足夠本地日 K 之標的列入回應 `unavailable_tickers` 清單，不拋出 500。
  2. **查詢排序修正**：所有 DuckDB 日 K 讀取強制使用 `WHERE ... AND date <= ? ORDER BY date DESC LIMIT ?`，取回最近 `lookback_bars` 筆後再於記憶體按 `date ASC` 排序，杜絕誤取多年以前舊資料。
  3. **原始收盤價企業行為跳空守衛（Corporate-Action Jump Guard）**：標註 `price_adjustment_mode`（`tw_daily_bars` 為 `"raw_close"`，`global_daily_bars` 為 `"adjusted_close"`）。針對 `"raw_close"` 序列（如台股），因為股票分割（如 1拆4、1拆10）或大額減資會在未還原收盤價產生單日 `|r_{i,t}| > 0.35`（遠超台股 ±10% 漲跌幅限制）的虛假斷崖，模組自動將超過市場法規單日異常門檻（預設台股一般股 `|r| > 0.25`、海外股 `|r| > 0.60`）之未驗證跳空日標記至 `corporate_action_suspects` 並自歷史報酬交集中剔除該異常單日，防止單次分割摧毀 95%/99% VaR 與共變異矩陣，同時於回應附帶未還原權息近似提示。

### Decision 2: `backtest/risk_xray.py` — 雙口徑歐拉拆解、階梯式尾端門檻與跨市場揭露
- **決策**：
  1. **雙口徑（Total Portfolio vs Risk Sleeve）嚴格一致定義**：
     - 設各風險資產對總帳戶權重為 $w_i \ge 0$，風險總權重 $W_{\text{risk}} = \sum_{i=1}^N w_i \le 1$，現金權重 $w_{\text{cash}} = 1 - W_{\text{risk}} \ge 0$；定義風險資產槽正規化權重 $\tilde{w}_i = w_i / W_{\text{risk}}$（當 $W_{\text{risk}} > 0$，滿足 $\sum \tilde{w}_i = 1$）。
     - **全組合口徑（Total Basis）**：組合日報酬 $R_t = \sum_{i=1}^N w_i r_{i,t}$（現金報酬視為 0），全組合年化波動率 $\sigma_{\text{total}} = \sqrt{252 \cdot w^\top \Sigma w}$，歷史損失序列 $L_t = -R_t$ 計算出之 `var_95`、`cvar_95`、`var_99`、`cvar_99` 完全對應全帳戶淨值波動（即 `var_total == W_risk * var_sleeve`）。
     - **風險槽口徑（Risk-Sleeve Basis）**：另行輸出 `annualized_volatility_risk_sleeve` $\sigma_{\text{sleeve}} = \sqrt{252 \cdot \tilde{w}^\top \Sigma \tilde{w}} = \sigma_{\text{total}} / W_{\text{risk}}$，以及僅針對風險槽計算之集中度 `hhi_risk_sleeve = sum(\tilde{w}_i^2)`（避免現金權重平方虛增集中度）。
     - **歐拉波動拆解恆等式**：當 $\sigma_{\text{total}} > 10^{-12}$ 時，第 $i$ 檔成分波動率 $CV_i = 252 \cdot w_i (\Sigma w)_i / \sigma_{\text{total}}$，百分比波動貢獻 $PCV_i = CV_i / \sigma_{\text{total}} = \tilde{w}_i (\Sigma \tilde{w})_i / (\tilde{w}^\top \Sigma \tilde{w})$。數學上**同時嚴格滿足** $\sum_{i=1}^N CV_i = \sigma_{\text{total}}$ 與 $\sum_{i=1}^N PCV_i = 1.0$！注意當某標的與組合呈負相關對沖時，$CV_i$ 與 $PCV_i$ 可為負值（代表降低組合風險），API 與前端必須原生支援負值呈現。
  2. **分位數級階梯樣本門檻（防止 30 日算 99% CVaR 偽數值）**：
     - 重疊有效交易日 $N < 30$：回傳 `status = "insufficient_history"`，所有風險指標不予硬算。
     - $30 \le N < 60$：計算波動率、歐拉 PCV、HHI 與高相關配對（門檻 `|corr| >= 0.75`，區別於 `backtest/risk.py` 交易前硬攔截聚類門檻 `0.80`，作為提前預警），但 `var_95 / cvar_95` 與 `var_99 / cvar_99` 均設為 `None` 並回傳 `tail_95_status = "insufficient_tail_samples"`、`tail_99_status = "insufficient_tail_samples"`。
     - $60 \le N < 100$：啟用 `var_95 / cvar_95`（附 `tail_95_sample_count`），而 `var_99 / cvar_99` 維持 `None` + `tail_99_status = "insufficient_tail_samples"`。
     - $N \ge 100$：同時啟用 95% 與 99% VaR / CVaR（附 `tail_99_sample_count`）。
  3. **跨市場幣別與交易日曆雙重揭露**：混合 TWD / USD / JPY 標的時強制標記 `currency_mode = "local_currency_return_proxy"`，拒絕輸出金額 VaR，並同時附帶 `fx_unhedged_warning`（未計入 USD/TWD、JPY/TWD 匯率波動）與 `calendar_mismatch_warning`（跨國國定假日交集造成部分日期跨度超過單一交易日）。

### Decision 3: `backtest/statistical_validation.py` — 純序列輸入解耦、零變異守衛與雙軌檢定
- **決策**：
  - **架構解耦**：`backtest/simulation.py` 僅負責單筆開盤撮合與切窗，因此 `backtest/statistical_validation.py` 設計為接收 `daily_returns: Sequence[float]` 與選用 `trade_returns: Sequence[float]` 之獨立純函式模組，並提供整合轉換器供 `evaluators/forecast_performance.py` 及回測分析腳本呼叫，不污染 `simulation.py` 單筆撮合職責。
  - **統一三大檢定規格與邊界保護**（全程使用 `np.random.default_rng(seed)`）：
    1. **零變異與常數序列守衛**：在任何檢定前，若有效樣本標準差 $\text{std} \le 10^{-12}$，立即終止隨機抽樣，回傳 `status = "degenerate_zero_variance"`、`p_value = 1.0`、`is_statistically_significant = False`，徹底杜絕 `0 / 0 = NaN` 或假性 `p = 0.001`。
    2. **對數日報酬 Rademacher 符號翻轉檢定（`p_value_return`）**：針對日報酬序列（需 $T \ge 30$，先轉為有限對數報酬 $z_t = \ln(1 + \max(r_t, -0.999999))$ 以滿足正負對稱性、避免 `-100%` 翻轉為 `+100%` 的有限責任失真），執行 $B = 1000$ 次隨機符號 $\epsilon_t \in \{-1, +1\}$ 檢定 $T^{(b)} = \text{mean}(\epsilon \odot z) / \text{std}(\epsilon \odot z)$，計算單尾 `p_value_return = (1 + count(T_sim >= T_obs)) / (B + 1)`。
    3. **交易順序置換最大回撤檢定（`p_value_max_drawdown`）**：針對已實現單筆交易報酬序列（需 $M \ge 10$，下界夾限 `> -0.999999`），執行 $B = 1000$ 次順序隨機排列重算複利淨值曲線最大回撤 $DD^{(b)}$，輸出 `p_value_max_drawdown` 與 `mc_95_worst_drawdown`。
    4. **Circular Block Bootstrap 年化 Sharpe 信賴區間**：針對日報酬序列（需 $T \ge 30$），以區塊長度 $L = \max(5, \lfloor T^{1/3} \rfloor)$ 抽樣 $B = 1000$ 次（每次若抽中零變異區塊則 Sharpe 記為 `0.0`），輸出 `[sharpe_ci_low, sharpe_ci_high]`。

### Decision 4: `backtest/turnover_optimizer.py` — 精確收斂之純 NumPy ADMM 非對稱換手優化器
- **決策**：廢除不具複合近端等價性（$\text{prox}_{g + I_{\mathcal{C}}} \ne \Pi_{\mathcal{C}} \circ \text{prox}_g$）之單步啟發式寫法，改用數學上保證收斂至全域凸最佳解之 **ADMM（Alternating Direction Method of Multipliers）** 求解分裂問題：
  $$\min_{w \in \mathcal{C}, \, z \in \mathbb{R}^N} \; \frac{\lambda_{\text{risk}}}{2} w^\top \hat{\Sigma}_{\text{shrunk}} w - \mu^\top w + \gamma_{\text{buy}} \|(z - w_{\text{prev}})_+\|_1 + \gamma_{\text{sell}} \|(w_{\text{prev}} - z)_+\|_1 \quad \text{s.t. } w = z$$
  - **為什麼選 ADMM？**
    1. **原生支援台股非對稱摩擦成本**：買進懲罰 `gamma_buy`（預設手續費 + 滑價，無量綱小數如 `0.0015`）與賣出懲罰 `gamma_sell`（含台股 `0.3%` 證交稅，如 `0.0045`；若傳對稱 `gamma` 則自動令兩者相等）在 $z$-update 中擁有**精確閉式非對稱軟閾值解（Asymmetric Soft-Thresholding）**！
    2. **零除零風險與嚴格可行域保證**：令懲罰參數 $\rho = \max(1.0, \lambda_{\text{risk}} \|\hat{\Sigma}_{\text{shrunk}}\|_2)$，因為二次項矩陣 $\lambda_{\text{risk}} \hat{\Sigma}_{\text{shrunk}} + \rho I$ 之最小特徵值恆 $\ge \rho \ge 1.0$，即使呼叫端傳入風險中性 $\lambda_{\text{risk}} = 0$，步長與線性系統亦絕對正定不發散！$w$-update 透過 Dykstra 投影至非空凸集 $\mathcal{C} = \{0 \le w_i \le u_i, \sum_{i \in g} w_i \le S_g, \sum w_i \le W_{\max}\}$（因 $w = \mathbf{0} \in \mathcal{C}$，可行域恆非空）。
    3. **參數對齊與最終可行裁剪**：預設 `max_single_weight = 0.10` 與 `backtest/risk.py:RiskPolicy` 保持一致（呼叫端可覆寫），並在迭代結束後對輸出施作一次確定性向下裁剪投影（`clip_to_feasible_polytope`），確保任何迭代步數下約束殘差恆 `<= 1e-8`，同時誠實回報 `solver_status`（`"converged"` | `"max_iterations_feasible"` | `"fallback_hold_previous"`）。

### Decision 5: `data/loader_canary.py` — 冷卻保護、平行執行、白名單遮罩與零 DB 寫入
- **決策**：
  1. **防封鎖最小冷卻間隔**：預設 `min_interval_sec = 1800`（30 分鐘）；若 `data/cache/canary_health_latest.json` 距今未滿 30 分鐘且未顯式傳入 `--force`，CLI 直接回傳現有快照並略過真實外網請求，防止排程重試或手動誤觸遭 TWSE / Investing 封鎖 IP。
  2. **平行探針與非阻塞排程**：使用 `ThreadPoolExecutor(max_workers=4)` 平行執行 8 個探針，單探針嚴格 `timeout_sec = 8.0`，整體巡檢控制在 16 秒內完成；於 `run_daily.sh` 中以非阻塞容錯呼叫（失敗僅記錄警告，絕不中斷每日預測主流程）。
  3. **白名單結構與零 DuckDB 寫入保證**：快照每個來源僅保留白名單欄位（`source`, `status`, `latency_ms`, `checked_at`, `http_status`, `missing_fields`, `sanitized_reason`），嚴禁輸出原始 Traceback 或帶 query 參數之完整 URL，且全程不開啟任何可寫 DuckDB 連線。

### Decision 6: API 權限分級、反 DoS 與最小錨點前端整合
- **決策**：
  1. **私有模擬帳戶端點強制 Fail-Closed 認證防枚舉**：`GET /api/v1/paper/accounts/{account_id}/risk-xray` 強制驗證 `account_id` 正則 `^[a-zA-Z0-9_-]{1,64}$` 並要求管理金鑰認證（復用 Wave 1 `verify_watchlist_write_access` / `WATCHLIST_API_KEY`），且以嚴格 `read_only=True` 重試模式讀取 `PaperLedger`（絕不退回 `read_only=False`）。
  2. **公開計算端點雙重反 DoS 與跨 Wave 安全備案**：`POST /api/v1/risk/xray`（`<= 100` 檔、`window` `30..504`）與 `POST /api/v1/risk/optimize-weights`（`<= 50` 檔）同時掛載每 IP 滑動視窗限流（優先使用 Wave 1 `core/netlimiter.py`，若獨立執行時模組未載入則自動啟用路由內建 Fail-Closed 本地限流器）與進程級 `threading.BoundedSemaphore(2)` 矩陣計算閘門（忙碌超時回傳 HTTP 503）；當 `watchlist_items` 表尚未初始化時，`GET /api/v1/watchlist/risk-xray` 優雅回傳 HTTP 200 + `status: "watchlist_unavailable"`。
  3. **前端最小耦合掛載**：所有 UI 模板與邏輯收斂於獨立 `api/templates/partials/risk_xray_panel.html` 與 `api/static/js/risk-xray.js`，前端一律以當前自選股代號陣列呼叫 `POST /api/v1/risk/xray`（天然同時支援 DuckDB 與匿名 `localStorage` 自選股），於 `tab_dashboard.html` 與 `app.js` 僅保留單一錨點引用（`<= 10` 行改動），並原生渲染負對沖 PCV 標籤。

## Risks / Trade-offs

- **[Risk] 台股原始收盤價遇到一般小額除息（如殖利率 3%~5%）仍會計入日報酬波動** $\rightarrow$ **Mitigation**：大額股票分割/減資斷崖（`|r| > 25%`）由異常跳空守衛自動攔截並揭露於 `corporate_action_suspects`；一般除息在回應 `price_adjustment_mode` 與前端說明列誠實標示「台股採證交所原始收盤價（已過濾分割異常跳空，含一般除息波動）」。
- **[Risk] 並行矩陣優化請求造成短暫 CPU 峰值** $\rightarrow$ **Mitigation**：50 檔資產上限 + 300 步 ADMM（單次 `< 5ms`）+ 進程級 `BoundedSemaphore(2)` + 每 IP 滑動視窗限流，徹底鎖死最大併發 CPU 佔用。

## Migration Plan

1. **Phase 1（本地價格底座與純量化核心）**：於 `DuckDBManager` / `fetcher.py` 補齊離線 `global_daily_bars` 與正確 `DESC LIMIT` 讀取 helper，實作純函式 `risk_xray.py`、`statistical_validation.py`、`turnover_optimizer.py` 並接入驗證摘要器。
2. **Phase 2（隔離金絲雀探測道）**：實作具 30 分鐘冷卻、平行逾時、白名單遮罩與零 DB 寫入斷言之 `loader_canary.py` 與 `check_loader_health.py`。
3. **Phase 3（唯讀/受保護 API 與 FastMCP 工具）**：確認 Wave 1 `duckdb_lock` / `netlimiter` 前置就緒（含安全降級 shim），掛載 `api/routes/risk.py`、`api/routes/stats.py` 與唯讀 MCP 工具。
4. **Phase 4（前端風險 X 光面板與顯著性標章）**：以最小錨點整合 `risk_xray_panel.html` 與 `risk-xray.js`，通過雙主題 WCAG AA/AAA 與負對沖 PCV 測試。
5. **回滾策略**：新增表與獨立路由均向後相容，移除路由掛載即可秒級回滾。

## Open Questions

- 無阻塞性開放問題；所有跨 Wave 依賴、價格表名稱、企業行為邊界與認證規則均已明確定義。
