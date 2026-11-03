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
