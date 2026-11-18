# Proposal

## Why

在 Wave 1（`openstock-inspired-operator-suite`）建立跨瀏覽器自選股清單、量化價位告警與操盤工作台後，使用者雖然能快速篩選並追蹤一籃子標的，但仍缺少三個關鍵的機構級量化驗證閉環：(1) 無法透視「自選股組合或模擬倉」的尾端暴跌風險（VaR / Expected Shortfall）與單一持股波動貢獻度；(2) 現有歷史驗證與模擬僅輸出單一路徑報酬或準確率，無法檢定正期望值與最大回撤是否顯著優於隨機運氣（`p-value` 與信賴區間）；(3) 外部多源資料管線缺少隔離無快取的「金絲雀探測（Canary Health）」來區分上游斷線與網頁/API 欄位改版，且模擬換倉尚未納入非對稱買賣換手率懲罰以抑制台股賣方 `0.3%` 證交稅耗損。借鏡 MIT 開源專案 `HKUDS/Vibe-Trading` 的純函式量化驗證精華，本計畫以零實盤下單風險、零重型商業求解器依賴方式將上述四項核心能力移植落地。

## What Changes

- **多市場本地日 K 持久化與「組合風險 X 光片（Portfolio Risk X-Ray）」**：
  - 於 DuckDB 擴充 `global_daily_bars` 離線日 K 表（承接批次排程已抓取之海外調整後收盤價，與既有台股 `tw_daily_bars` 形成純本地唯讀價格底座，嚴禁公開 API 即時外呼 `yfinance`），並加入企業行為/除權息異常單日跳空過濾守衛。
  - 新增純收盤價矩陣計算模組 `backtest/risk_xray.py`（零網路 I/O、嚴格 `<= as_of` 零前視偏差、嚴格 `allow_nan=False`），以雙口徑（總組合口徑與風險資產槽口徑）計算年化波動率、階梯式樣本門檻之 **95% VaR/CVaR（`N >= 60`）** 與 **99% VaR/CVaR（`N >= 100`）**、風險槽 **HHI 集中度**、支援負對沖值之 **歐拉波動率貢獻佔比（Euler PCV）** 與高相關共移配對，並明確標示跨幣別未避險與跨市場休市日曆錯位警告。
- **策略與交易序列統計顯著性檢定（Sign-Flip、Order Permutation & Block Bootstrap）**：
  - 新增純函式模組 `backtest/statistical_validation.py`，提供固定隨機種子可重現之 **Rademacher 符號翻轉日報酬均值顯著性檢定（`p_value_return`）**、**交易順序置換最大回撤路徑檢定（`p_value_max_drawdown`）** 與 **Circular Block Bootstrap 年化 Sharpe 95% 信賴區間**，內建零變異（`std <= 1e-12`）與小樣本誠實守衛，整合進本地預測驗證與回測分析報告。
- **非對稱換手率懲罰多頭部位優化器（Asymmetric Turnover-Aware Allocation Optimizer）**：
  - 新增純 NumPy 無外部重型求解器依賴之 `backtest/turnover_optimizer.py`，採用 **ADMM（交替方向乘子法）+ Dykstra 可行域投影 + 最終可行域裁剪**，在非負多頭、單股上限（預設對齊 `RiskPolicy` 之 `0.10`）與產業群組上限約束下，精確支援買賣非對稱 L1 換手懲罰（`gamma_buy * ||(w - w_prev)+||_1 + gamma_sell * ||(w_prev - w)+||_1`，直接反映台股賣出 `0.3%` 證交稅摩擦）。
- **隔離無快取「上游金絲雀健康巡檢道（Upstream Canary Health Lane）」**：
  - 新增 `data/loader_canary.py` 與 `scripts/check_loader_health.py`，以平行執行緒與 30 分鐘最小冷卻保護對 TWSE、TPEx、yfinance、answerbook、Investing、Polymarket、BigGo 法說會與 arXiv 進行繞過快取且零生產 DuckDB 寫入之隔離探測，精準歸類 `healthy`、`upstream_unreachable_or_ratelimited` 與 `schema_drift_parse_error`，並輸出白名單欄位與敏感遮罩之原子 JSON 快照。

## Capabilities

### New Capabilities
- `upstream-canary-health-lane`: 以固定探針標的、最小冷卻間隔與平行隔離模式實測各資料源連通性與欄位結構完整性，區分上游網路/限流異常與解析器改版失效，且零生產資料庫寫入、零憑證洩漏。
- `turnover-aware-allocation`: 在多頭無槓桿、單股上限與產業上限約束下，以純 NumPy ADMM 演算法結合收縮共變異風險與買賣非對稱 L1 換手懲罰計算目標權重與單邊換手率。

### Modified Capabilities
- `portfolio-risk`: 擴充本地多市場日 K 底座與持倉後/自選股籃子之 Risk X-Ray 尾端風險診斷（雙口徑波動率、分位數階梯樣本門檻 VaR/CVaR、除權息異常跳空守衛、歐拉波動貢獻與跨市場幣別/日曆錯位揭露）。
- `historical-trading-simulation`: 擴充回測與驗證序列之確定性統計顯著性檢定（日報酬符號翻轉 `p_value_return`、交易順序置換 `p_value_max_drawdown`、Bootstrap Sharpe 信賴區間與零變異/小樣本守衛）。
- `fastapi-rest-service`: 新增受限流與併發閘門保護之唯讀 Risk X-Ray 端點、Fail-Closed 認證之私有模擬帳戶風險端點、換手率優化預覽端點、快取制 Canary 健康快照端點與唯讀 FastMCP 工具。
- `web-dashboard-ui`: 於自選股檢視與回測驗證介面以最小錨點耦合新增「組合風險 X 光健檢」互動面板（支援負對沖 PCV 與雙軌自選股）與統計顯著性標章。

## Impact

- **量化與驗證核心 (`backtest/`, `evaluators/`)**：新增純函式 `backtest/risk_xray.py`、`backtest/statistical_validation.py`、`backtest/turnover_optimizer.py`，與既有 `backtest/risk.py`（交易前守衛）互補，絕不引入實盤下單 SDK 或 `cvxpy` 重型編譯依賴。
- **資料管線與腳本 (`data/duckdb_manager.py`, `data/fetcher.py`, `data/loader_canary.py`, `scripts/check_loader_health.py`)**：新增離線 `global_daily_bars` 同步寫入與零 DB 寫入之隔離金絲雀快照 `data/cache/canary_health_latest.json`。
- **API、MCP 與前端 (`api/routes/risk.py`, `api/routes/stats.py`, `mcp_server.py`, `api/templates/partials/risk_xray_panel.html`, `api/static/js/risk-xray.js`)**：嚴格區分公開唯讀計算端點（含滑動視窗限流 + 併發 Semaphore）與私有模擬帳戶端點（強制 API Key 驗證防枚舉），並以最小錨點掛載降低與 Wave 1 之合併衝突。
- **跨 Wave 依賴順序**：排程於 Wave 1（`openstock-inspired-operator-suite`）落地 `core/duckdb_lock.py`、`core/netlimiter.py` 與 `watchlist_items` 後合流，且內建表未就緒時之 Fail-Closed 安全降級備案。
