# Spec Delta

## Purpose

提供外部資料與模型上游呼叫之多金鑰輪詢池、單金鑰併發節流閘門與短時失敗熔斷冷卻能力，並提供嚴格遮罩機密資訊之環境變數啟動健檢工具，確保新機部署與上游異常時具備高度穩定性。

## ADDED Requirements

### Requirement: Multi-Key Pool Rotation and Per-Key Concurrency Throttle
系統 SHALL 支援將逗號分隔之多把上游 API 金鑰解析為輪詢池以分散配額消耗，並限制單一金鑰同時在途請求數上限，防止突發併發請求導致上游連線掛起或限流封鎖。

#### Scenario: Rotating across multiple configured API keys
- **WHEN** 環境變數提供多把以逗號分隔之有效上游金鑰且系統連續發起多次請求
- **THEN** 系統依序輪詢使用不同金鑰發送請求，且日誌與錯誤訊息絕不輸出任何金鑰明文

#### Scenario: Enforcing per-key in-flight concurrency limit
- **WHEN** 單一金鑰之同時在途請求數達到設定上限
- **THEN** 後續請求排隊等待至名額釋放或逾時快速降級，不對同一金鑰無上限併發轟炸

### Requirement: Short-Lived Failure Cooldown Circuit Breaker
系統 SHALL 在上游端點發生逾時、連線失敗、HTTP 429 或 5xx 錯誤時啟動可設定秒數之熔斷冷卻期；冷卻期間內後續請求立即快速跳過該故障端點並回退至快取或次級備援，且抑制重複錯誤日誌洗版。

#### Scenario: Upstream endpoint enters cooldown after failure
- **WHEN** 某外部上游端點請求失敗進入 60 秒熔斷冷卻期，且期間內再次收到查詢
- **THEN** 系統在毫秒級內直接略過該故障上游、回傳附帶 `stale` 標記之有效快取或切換備援節點，不阻塞回應亦不重複刷寫相同錯誤堆疊

#### Scenario: Automatic half-open recovery after cooldown expires
- **WHEN** 故障上游端點之熔斷冷卻時間屆滿後收到新請求
- **THEN** 系統自動放行探測請求；若成功則恢復正常狀態，若仍失敗則重新進入冷卻期

### Requirement: Secret-Safe Pre-Flight Environment Doctor
系統 SHALL 提供環境變數健檢命令列工具，分類驗證「必填設定」、「選填功能設定（含未設定時之降級行為說明）」與「已棄用設定」，輸出修復指引並嚴格遮罩所有敏感憑證數值。

#### Scenario: Missing required environment variable detection
- **WHEN** 執行環境健檢工具時缺少必要設定或連線格式無效
- **THEN** 工具以非零退出碼結束，清楚列出缺失項目與設定範例，且已設定之金鑰僅顯示存在狀態而不印出任何明文或前後綴內容

#### Scenario: Optional variables missing with graceful degradation report
- **WHEN** 必填設定齊備但部分選填通知或外部 API 金鑰未設定
- **THEN** 工具以零退出碼通過檢查，並逐項說明對應功能將採取之安全降級模式（如停用特定推播或改用本地規則模板）
