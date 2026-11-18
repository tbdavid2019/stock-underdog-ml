# Web Dashboard UI Specification (Delta)

## ADDED Requirements

### Requirement: Watchlist Risk X-Ray Diagnostic Panel
前端必須（SHALL）以獨立樣板與腳本提供一鍵「組合風險 X 光健檢」互動面板，將當前 DuckDB 或匿名 `localStorage` 自選股代號陣列送往 `POST /api/v1/risk/xray`，清晰呈現雙口徑波動率、階梯式 95%/99% VaR 與 CVaR、風險槽 HHI、支援負對沖值之歐拉風險貢獻橫條、高相關配對與跨幣別/日曆錯位提示，且雙主題配色嚴格符合 WCAG AA/AAA 對比度規範（嚴禁白字疊琥珀金 `#FFB010`）。

#### Scenario: Anonymous localStorage watchlist and negative hedging PCV rendered cleanly
- **WHEN** 公開訪客以本地自選股點擊「組合風險 X 光健檢」且其中包含負相關對沖標的（`PCV < 0`）
- **THEN** 面板無需寫入權限即可渲染風險指標，並將負貢獻度標示為風險減縮項目而不破版

#### Scenario: Insufficient history or tail samples rendered without broken NaN labels
- **WHEN** 後端回傳 `insufficient_history` 或因視窗未滿 100 日將 `var_99` 標記為 `insufficient_tail_samples`
- **THEN** 面板顯示清楚的樣本門檻提示文案，絕不顯示 `NaN%`、`undefined` 或誤導性 `0.00%`

### Requirement: Statistical Significance Badge Presentation
前端必須（SHALL）在歷史模擬或績效驗證摘要區塊支援顯示統計顯著性檢定標章（包含 Bootstrap Sharpe 95% 信賴區間與 `p_value_return` 狀態，涵蓋顯著、未達顯著、樣本不足與零變異狀態），協助操盤者區分真實超額表現與短期隨機運氣。

#### Scenario: Non-significant, small-sample, or zero-variance state clearly distinguished
- **WHEN** 檢定結果之 `p_value_return >= 0.05` 或狀態為 `insufficient_samples` / `degenerate_zero_variance`
- **THEN** 介面以中性抑制色標示對應狀態，避免將少數幾筆幸運交易誤導呈現為高可靠度策略
