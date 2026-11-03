## Purpose

持續保存依每日訊號產生的模擬訂單、成交與帳戶變化，讓使用者追蹤策略在訊號出現後的表現，並透過一致的現金與部位帳本查核實際成本及風險影響。

## ADDED Requirements

### Requirement: Idempotent simulated order lifecycle
系統 SHALL 以帳戶、訊號與決策 session 識別模擬訂單，保存 submitted、rejected、filled、cancelled 狀態與原因；成交前執行風控，訂單與成交保持可追溯且重試冪等。

#### Scenario: Scheduler retries after fill
- **WHEN** 同一交易批次因中斷重跑
- **THEN** 已成交訂單保持單筆，現金、費用與持倉不重複扣加

### Requirement: Reconciled cash and positions
系統 SHALL 分市場及幣別維護多頭模擬帳戶，保存現金、股數、成本、已實現／未實現損益與每日淨值，處理拆股與配息事件並避免與還原價格重複計入。

#### Scenario: Corporate action and incomplete prices
- **WHEN** 持倉發生拆股、配息或當日缺價格
- **THEN** 帳本按事件調整且保留來源；缺價格以最後有效估值標記 stale，風控暫停新增曝險

### Requirement: Simulated operations boundary
系統 SHALL 清楚標示模擬帳戶與績效，僅使用與回測共用的成交及成本合約；帳戶建立、重設與執行採本機管理命令，公開服務提供唯讀查詢。

#### Scenario: Public user views account
- **WHEN** 使用者查詢模擬交易頁面
- **THEN** 能檢視帳本與淨值，公開 API 不提供重設、執行或真實券商下單功能
