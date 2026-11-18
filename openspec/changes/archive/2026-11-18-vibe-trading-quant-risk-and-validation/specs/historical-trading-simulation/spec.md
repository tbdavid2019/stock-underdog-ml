# Historical Trading Simulation Specification (Delta)

## ADDED Requirements

### Requirement: Deterministic Statistical Validation and Zero-Variance Guard
系統必須（SHALL）提供與單筆撮合解耦之純函式統計顯著性檢定模組，以固定隨機種子（`random_seed`）對日報酬序列（`T >= 30`）執行對數報酬 Rademacher 符號翻轉檢定（`p_value_return`）與 Circular Block Bootstrap 年化 Sharpe 95% 信賴區間，並對已實現交易報酬序列（`M >= 10`）執行順序置換最大回撤檢定（`p_value_max_drawdown`）；當樣本標準差 `<= 1e-12` 或樣本不足時，必須（MUST）提前回傳 `degenerate_zero_variance` 或 `insufficient_samples` 且通過 `allow_nan=False`。

#### Scenario: Identical seed yields bit-identical p-values and confidence intervals
- **WHEN** 對同一組日報酬與交易損益序列以相同 `random_seed` 連續執行兩次統計檢定
- **THEN** 兩次輸出之 `p_value_return`、`p_value_max_drawdown`、`sharpe_ci_low` 與 `sharpe_ci_high` 完全一致，且隨機零均值雜訊路徑不會被系統性誤判為 `p < 0.05`

#### Scenario: Constant or all-zero return series avoids division by zero and NaN
- **WHEN** 傳入全零報酬序列或每筆報酬完全相同之零變異序列
- **THEN** 檢定模組不發生除以零或產生 `NaN`，回傳 `status: "degenerate_zero_variance"`、`p_value_return: 1.0` 與 `is_statistically_significant: false`

#### Scenario: Minimum sample guard prevents misleading significance claims
- **WHEN** 已完成交易筆數少於 10 筆或有效日報酬序列少於 30 個交易日
- **THEN** 對應檢定區塊回傳 `status: "insufficient_samples"`，將 `is_statistically_significant` 設為 `false` 且不填寫虛假 `p-value`
