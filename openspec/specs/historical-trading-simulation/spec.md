# historical-trading-simulation Specification

## Purpose
以當時可取得的訊號逐日重播交易，納入可成交限制、手續費、稅與滑價，提供與預測準確率分開解讀的策略淨績效，並保留資料與設定以重現每次模擬。

## Requirements

### Requirement: Chronological walk-forward evaluation
系統 SHALL 依時間切分校準與測試區間，模型選擇、排名與權重只使用決策前可取得資料，依預測步長隔離重疊標籤；保存股票池、資料、模型版本與逐期選擇，揭露預訓練資料截止未知及現今股票池重播的限制。

#### Scenario: Future winner would improve historical trades
- **WHEN** 某模型在測試區間結束後才成為績效第一
- **THEN** 該模型不得因此回溯成為區間起點選用模型，報告列出逐期當時選擇

### Requirement: Executable fills and explicit costs
系統 SHALL 使用訊號可用後的下一可成交 session 開盤模擬買入，以設定持有期收盤退出；依市場、商品與生效日期套用費用、稅、最低費用、滑價及數量規則。停牌、缺行情、漲跌停或容量不足須保守拒絕或按明示規則遞延。

#### Scenario: Forecast from closing data
- **WHEN** 收盤後產生買入訊號
- **THEN** 模擬採其後可交易開盤，報告含毛報酬、逐筆成本、淨報酬與未成交原因

#### Scenario: Opening liquidity is insufficient
- **WHEN** 開盤下單量超過決策前已知的保守開盤容量
- **THEN** 僅按已宣告容量部分成交並取消剩餘量；缺可靠容量資料則拒單，當日全天成交量不得回填當時容量

### Requirement: Reproducible performance report
系統 SHALL 保存不可變 run ID、設定與資料快照，輸出淨值曲線、淨報酬、最大回撤、Sharpe、成交數、勝率、換手率、曝險與同幣別基準；資料不足的統計須標示 unavailable。

#### Scenario: Identical replay
- **WHEN** 使用相同快照、模型輸出與設定重播
- **THEN** 逐筆成交、現金與績效相同，且可識別每筆交易所依據預測與風控結果

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
