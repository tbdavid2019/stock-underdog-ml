## Purpose

持續以已驗證預測觀察各市場與版本的模型表現，區分樣本不足、資料延遲與品質退化，透過可追溯狀態轉換調整模型對選股的影響並保留恢復所需的影子證據。

## ADDED Requirements

### Requirement: Evidence based model health state
系統 SHALL 分市場、模型版本與步長輸出 insufficient_data、healthy、watch、degraded、suspended 狀態，並另列服務可用性及行情新鮮度；判斷僅使用當時已開獎資料及版本化門檻。

#### Scenario: Upstream outage without new actuals
- **WHEN** 上游斷線使當日尚無新開獎
- **THEN** 標示 stale／unavailable 並維持最後品質狀態，品質判斷不把缺資料視為猜錯

### Requirement: Auditable weight reduction and suspension
系統 SHALL 支援先觀測後啟用的自動降權政策，保存每次轉換證據、門檻、有效時間、原權重與有效權重；suspended 模型對目標合成與共振資格貢獻為零，原始影子預測持續保存。

#### Scenario: Persistently weak recent performance
- **WHEN** 模型達到政策規定的退化或停用證據
- **THEN** 有效權重按政策降低或歸零，API 與報告可解釋原因；其他健康模型可繼續供應訊號

### Requirement: Recovery hysteresis and cold start
系統 SHALL 要求停用後新增的獨立開獎樣本及連續恢復證據才恢復權重，版本更新重新累積樣本；所有合格模型權重為零時須輸出 unavailable 目標及觀望原因。

#### Scenario: One good day after suspension
- **WHEN** 停用模型僅新增一日良好成績
- **THEN** 維持停用並顯示恢復進度，達到完整恢復政策後逐級恢復

#### Scenario: No eligible model remains
- **WHEN** 所有模型均停用、失敗或尚未晉升
- **THEN** 系統保留技術面與籌碼資訊，模型合成目標為空且不產生模型共振資格
