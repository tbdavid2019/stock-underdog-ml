## Purpose

將既有宏觀曝險訊號與模擬帳戶的實際持倉連結，在送出模擬訂單前檢查單股、產業與相關性集中程度，提供可解釋的縮量或拒絕結果並保留決策紀錄。

## ADDED Requirements

### Requirement: Pre-trade exposure constraints
系統 SHALL 在新增模擬曝險前檢查現金、單股上限、產業上限、宏觀總曝險、成交量參與率與帳戶回撤停買門檻；縮量後仍不合交易數量規則時須拒絕，所有結果保存限額與原因。

#### Scenario: Several semiconductor picks
- **WHEN** 產業部位達設定上限且新訊號仍為同產業股票
- **THEN** 訂單縮量至限額或拒絕，即使各股預測均看漲亦保留限制

### Requirement: Correlation and data quality guard
系統 SHALL 以決策前報酬估計相關性群組曝險並揭露觀察視窗及樣本數；行情、宏觀、分類或相關性必要資料缺漏或過期時，暫停新增曝險並保留降低部位能力。

#### Scenario: Unknown sector or stale macro
- **WHEN** 新買入候選的必要風控資料不足
- **THEN** 產生資料品質拒單原因，資料恢復後才重新評估

### Requirement: Post-trade risk visibility
系統 SHALL 顯示部位、產業與群組曝險、現金、回撤、適用宏觀上限及違規原因，分帳戶幣別呈現，既有超限部位須列出下一 session 的減碼建議。

#### Scenario: Market movement creates limit breach
- **WHEN** 價格變動或宏觀降檔使現有部位超限
- **THEN** 顯示超限來源、暫停增加該風險並產生可追溯的減碼建議
