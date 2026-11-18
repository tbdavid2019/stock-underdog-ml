# Spec Delta

## ADDED Requirements

### Requirement: Global Command Palette Search and Keyboard Navigation
系統 SHALL 於所有儀表板分頁支援透過 `⌘K`（macOS）、`Ctrl+K`（Windows/Linux）或頂部搜尋鈕開啟全域指令面板，對短字元優先過濾本地快照、達門檻字數時防抖查詢後端解析器，支援方向鍵與 Enter 展開抽屜或跳轉個股歷史頁。

#### Scenario: Invoke command palette with debounced search
- **WHEN** 使用者按下 `⌘K` 或 `Ctrl+K` 輸入關鍵字（如 `台積` 或 `NVDA`）並按 `Enter` 或 `Esc`
- **THEN** 系統毫秒級呈現匹配標的，按 `Enter` 展開該標的抽屜或跳轉歷史頁並記錄至最近瀏覽列，按 `Esc` 關閉並還原焦點

### Requirement: Browser-Style Recent and Pinned Ticker Workspace Bar
系統 SHALL 於導覽區下方提供可橫向捲動之「最近瀏覽與釘選個股快捷列」，自動保留最近檢視標的（上限 10 檔）並支援單鍵切換、釘選與關閉。

#### Scenario: Switching between recently inspected stocks
- **WHEN** 使用者點擊快捷列中之任一歷史或已釘選標的膠囊標籤
- **THEN** 系統立即載入該標的之抽屜或歷史分析內容，無需重新搜尋或整頁重載

### Requirement: Dual-Track Watchlist Star Toggle and Market Filter Chip
系統 SHALL 於訊號表格、行動卡片、抽屜與歷史頁提供至少 44px 之星號收藏開關（有寫入金鑰時同步後端持久化清單，無金鑰訪客自動無縫存入瀏覽器 `localStorage`），並於篩選器提供 `★ 我的自選` 膠囊與純文字安全渲染之分享匯入橫幅。

#### Scenario: Filtering dashboard signals by watchlist
- **WHEN** 使用者點擊篩選器中之 `★ 我的自選` 按鈕
- **THEN** 系統即時將訊號清單過濾為目前生效之自選股集合，並可與方向、模型共識及籌碼條件交集篩選

### Requirement: Live Cross-Market Session Clock and Holiday-Aware Freshness Badge
系統 SHALL 於儀表板顯示台股、美股與日股即時交易階段（含開盤中、盤前倒數、日股午休、盤後、休市節日），並依據交易日行事曆顯示預測快照基準日與真實 `stale` 過期警示（連假休市期間不誤報過期）。

#### Scenario: Pre-market countdown, Tokyo lunch break, and holiday freshness display
- **WHEN** 使用者於盤前、日股午休或國定連假期間瀏覽儀表板
- **THEN** 系統準確顯示對應階段與倒數計時，且最新休市前交易日之有效快照顯示基準日期而不誤標為異常過期

### Requirement: In-Drawer Signal Alignment Card and Quant Alert Controls
系統 SHALL 於個股分析抽屜內呈現符合 WCAG AA 以上對比度之「多源訊號一致度摘要卡」與「量化告警快速訂閱控制項」（支援帶入有效 P10 防守價、P50 目標價、MA60 支撐或自訂價格），並於靜態資源或樣板變更時自動更新全站資產版本指紋。

#### Scenario: Subscribing to a P10 defense alert from stock drawer
- **WHEN** 授權使用者展開具分位數預測之個股抽屜並點擊「訂閱跌破 P10 防守告警」
- **THEN** 系統送出動態 P10 告警規則，並於抽屜內即時顯示啟用中告警狀態與參考模型標記
