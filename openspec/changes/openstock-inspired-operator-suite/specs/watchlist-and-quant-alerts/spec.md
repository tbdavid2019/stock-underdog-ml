# Spec Delta

## Purpose

提供跨進程安全之本地持久化與瀏覽器雙軌自選股管理，並支援結合時序預測分位數（P10/P50）、技術均線（MA60/MA120）與自訂目標價之量化條件監控與原子冷卻防洗版推播告警。

## ADDED Requirements

### Requirement: Persistent and Local Dual-Track Watchlist Management
系統 SHALL 提供持久化自選股清單之新增、查詢、移除與批次匯入能力（上限 100 檔正規化代號、備註上限 500 字元）；於生產環境未附帶有效寫入金鑰時預設拒絕後端寫入（Fail-Closed），並允許匿名訪客無縫退回瀏覽器本地儲存維護個人自選清單。

#### Scenario: Authorized admin persists watchlist across browsers
- **WHEN** 具備寫入權限之使用者將有效股票代號（如 `2330.TW` 或 `NVDA`）加入自選股並於另一瀏覽器讀取
- **THEN** 系統跨進程安全寫入持久化清單，且新視窗同步讀取已正規化之收藏狀態

#### Scenario: Production fail-closed write guard and invalid input rejection
- **WHEN** 生產環境請求未附帶有效寫入金鑰，或嘗試寫入無效代號、超過 100 檔上限、或超過 500 字元備註
- **THEN** 未授權請求回傳 HTTP 401，驗證失敗回傳 HTTP 400/422，既有持久化清單維持不變

### Requirement: Sanitized Watchlist Shareable Export and Import
系統 SHALL 支援將自選股清單編碼為可分享之 URL 查詢參數，嚴格執行代號白名單字元驗證、去重與最多 100 檔上限過濾，並以純文字安全渲染預覽供接收端確認匯入。

#### Scenario: Open shared watchlist link with malformed tokens
- **WHEN** 使用者開啟包含混合有效代號與超長或非法腳本字元之分享連結
- **THEN** 系統僅保留通過白名單格式驗證之前 100 檔有效代號進行純文字預覽，忽略非法片段且不自動覆蓋既有清單

### Requirement: Static and Dynamic Quantitative Alert Rules
系統 SHALL 允許建立、啟停與刪除自訂價位告警（高於或低於指定正數價格）與動態滾動量化告警（跌破最新有效模型隔日 P10 防守價、觸及隔日 P50 目標價、回測 MA60 或 MA120 均線），並明示綁定之參考模型或自動保守取值策略。

#### Scenario: Create rolling P10 downside defense alert
- **WHEN** 使用者為具備有效分位數預測之標的建立「跌破 P10 防守價」動態規則
- **THEN** 系統儲存該規則與參考模型設定，並於每次巡檢依最新有效交易日之 P10 分位數評估觸發

#### Scenario: Reject quantile alert when no quantile model forecast exists
- **WHEN** 使用者對尚無任何有效模型分位數（P10）資料之標的建立 P10 告警
- **THEN** 系統拒絕建立或明確標記為等待分位數資料，絕不以零值或空值誤觸發告警

### Requirement: Atomic Cooldown Multi-Channel Alert Evaluation
系統 SHALL 僅允許授權排程或持有有效管理金鑰之請求觸發告警巡檢，並確保先成功持久化更新冷卻時間戳後才發送 Telegram、Discord 或 Email 通知，防止資料庫鎖競爭造成重複洗版。

#### Scenario: Alert condition met with atomic cooldown persistence
- **WHEN** 授權巡檢發現啟用規則滿足觸發條件且已逾冷卻期，並成功寫入最新觸發時間戳
- **THEN** 系統發送附帶資料基準時間之多通道通知，並在冷卻期內抑制同一規則重複推播

#### Scenario: Abort notification if cooldown state write fails or caller is unauthorized
- **WHEN** 未帶管理金鑰呼叫巡檢端點，或巡檢期間跨進程寫鎖逾時無法更新冷卻時間戳
- **THEN** 未授權呼叫立即回傳 HTTP 401；寫鎖逾時則中止推播並回傳 HTTP 503，絕不在未記錄冷卻狀態下發送外部通知
