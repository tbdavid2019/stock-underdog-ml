# Portfolio Risk Specification (Delta)

## ADDED Requirements

### Requirement: Local Multi-Market Historical Bars and Corporate-Action Jump Guard
系統必須（SHALL）透過本地唯讀儲存（台股 `tw_daily_bars`、海外離線同步之 `global_daily_bars` 與本地快取備援，強制按 `date <= as_of` 取最近視窗）載入收盤價，嚴禁於風險查詢期間發起外部即時網路補抓；針對未還原權息之原始收盤價序列，必須（MUST）偵測並剔除超過市場法規限幅之未驗證單日分割/減資異常跳空，同時記錄於 `corporate_action_suspects` 與 `unavailable_tickers`。

#### Scenario: Unadjusted stock split cliff filtered without distorting VaR
- **WHEN** 台股原始收盤價序列中存在單日跌幅超過異常門檻（如 1拆4 造成單日 `-75%` 斷崖）
- **THEN** 模組將該日期記錄於 `corporate_action_suspects` 並自尾端風險計算交集中剔除，避免單次分割偽造極端 VaR/CVaR

### Requirement: Dual-Basis Risk X-Ray and Tiered Tail Sample Thresholds
系統必須（SHALL）同時輸出全組合口徑（含現金稀釋之 `annualized_volatility_total` 與對應 VaR/CVaR）及風險資產槽口徑（`annualized_volatility_risk_sleeve`、`hhi_risk_sleeve` 與支援負對沖值之歐拉百分比貢獻 `PCV`），並強制執行分位數階梯門檻：重疊交易日 `N < 30` 回傳 `insufficient_history`；`N >= 60` 方計算 95% VaR/CVaR；`N >= 100` 方計算 99% VaR/CVaR（不足時回傳 `null` 與 `insufficient_tail_samples`），全程通過 `allow_nan=False`。

#### Scenario: Cash-diluted portfolio preserves both Euler sum and scaled total VaR
- **WHEN** 針對 `cash_weight = 0.30` 且重疊有效交易日 `N >= 100` 之多檔組合執行 Risk X-Ray
- **THEN** 風險資產 `sum(PCV)` 在誤差 `1e-6` 內等於 `1.0`、`sum(CV)` 等於 `annualized_volatility_total`，且 `var_95` 等於風險槽單獨計算 95% VaR 之 `0.70` 倍

#### Scenario: Window between 30 and 99 days withholds unreliable 99% tail estimates
- **WHEN** 組合有效重疊交易日為 `45` 日（介於 30 至 59 日之間）
- **THEN** 模組回傳有效波動率、HHI 與歐拉貢獻，但將 `var_95/cvar_95` 與 `var_99/cvar_99` 設為 `null` 並標記 `insufficient_tail_samples`

### Requirement: Cross-Market Currency and Calendar Mismatch Disclosure
當診斷對象同時包含不同計價幣別標的（如 TWD、USD、JPY 混合清單）且無匯率序列時，系統必須（MUST）拒絕輸出單一幣別金額 VaR，將 `currency_mode` 標記為 `local_currency_return_proxy`，並同時附帶未避險匯率波動警告（`fx_unhedged_warning`）與跨國休市日曆錯位警告（`calendar_mismatch_warning`）。

#### Scenario: Mixed TW and US watchlist discloses FX and calendar mismatch warnings
- **WHEN** 使用者對同時含 `2330`（TWD）與 `NVDA`（USD）之清單請求 Risk X-Ray
- **THEN** 回應標記 `currency_mode: "local_currency_return_proxy"`，並同時回傳非空之 `fx_unhedged_warning` 與 `calendar_mismatch_warning`
