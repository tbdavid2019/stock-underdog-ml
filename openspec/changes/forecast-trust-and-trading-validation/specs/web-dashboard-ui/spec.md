## ADDED Requirements

### Requirement: Reliability and model performance views
系統 SHALL 提供股票可信度榜與模型戰績榜，支援市場、20/60/120 交易日、版本與步長篩選；顯示命中率、Wilson 下界、MAPE、基準改善、樣本、覆蓋與更新時間，區分正式排名、樣本累積及 shadow。

#### Scenario: Explore a reliable stock
- **WHEN** 使用者點選股票榜中的標的
- **THEN** 開啟個股頁並保留篩選脈絡，單股成績與榜單使用相同後端驗證結果

#### Scenario: Compare shadow candidate
- **WHEN** Chronos-2 尚在 shadow 或資料不足
- **THEN** 顯示候選身分與累積進度，與 active 模型共同樣本戰績並列，避免顯示為選股推薦

### Requirement: Multi-model history and health explanation
系統 SHALL 在個股歷史提供可依模型選擇的預測曲線與 PK，清楚顯示正式目標日期、pending、legacy 限制、停用原因及實際參與合成的模型與權重。

#### Scenario: Missing session and suspended model
- **WHEN** 某日實際價格缺失或模型被停用
- **THEN** 顯示缺失／停用狀態並保留原始預測，PK 與共振使用後端資格結果

### Requirement: Simulation performance and risk presentation
系統 SHALL 提供模擬帳戶與回測結果的淨值、淨報酬、最大回撤、成本、交易紀錄及部位風險，分幣別顯示並標記模擬、資料延遲及歷史重播限制；手機與深淺主題均能讀取必要資訊。

#### Scenario: Simulation has no fills
- **WHEN** 帳戶尚無成交或資料暫時過期
- **THEN** 顯示原因、最後更新時間及空狀態，績效不足處以 unavailable 呈現
