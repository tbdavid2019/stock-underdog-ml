## ADDED Requirements

### Requirement: Forecast performance read endpoints
系統 SHALL 提供 GET `/api/v1/performance/stocks`、`/api/v1/performance/models`、`/api/v1/performance/tickers/{ticker}`、`/api/v1/models/health`，支援市場、版本、步長、視窗與 as_of 等適用篩選，回傳樣本、覆蓋、方法版本、產出時間與 stale 狀態。

#### Scenario: Invalid filter or insufficient evidence
- **WHEN** 視窗或步長無效，或有效樣本不足
- **THEN** 無效參數回傳驗證錯誤；證據不足回傳明確狀態與空排名，維持穩定分頁及排序

### Requirement: Simulation and risk read endpoints
系統 SHALL 提供 GET `/api/v1/backtests/{run_id}`、`/api/v1/paper/accounts/{account_id}`、其 `/orders`、`/equity` 與 `/risk` 子資源；僅公開經設定允許的模擬帳戶，所有查詢維持唯讀且不觸發推論、成交或帳戶重設。

#### Scenario: Query public simulated account
- **WHEN** 使用者查詢被允許公開的帳戶
- **THEN** 回傳模擬標記、成本與績效、資料時間及風控狀態；未知或未公開帳戶回傳 404

### Requirement: Verification provenance in history
系統 SHALL 在歷史預測查詢提供 forecast ID、模型版本、目標交易日、驗證狀態與正式開獎值，保留既有欄位相容性。

#### Scenario: Pending actuals in historical chart
- **WHEN** 目標日尚未正式開獎
- **THEN** API 標記 pending，圖表與 PK 不使用後續預測紀錄冒充實際行情
