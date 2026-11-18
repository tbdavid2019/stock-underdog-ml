# Web Dashboard UI Specification (Delta)

## MODIFIED Requirements

### Requirement: Financial Terminal Theme and Dual-Mode Switching
系統 SHALL 支援符合彭博終端調性之深色模式（曜石黑底色、琥珀金重點色）與淺色模式（高對比暖米白底色），提供切換開關與圖表連動更新，並同步 `<meta name="color-scheme">`、根節點 `color-scheme` 屬性與 `site.webmanifest` 品牌主題色。

#### Scenario: Toggling between Dark and Light mode
- **WHEN** 使用者點擊頂部深淺色切換開關
- **THEN** 系統切換 HTML root class、同步更新 `<meta name="color-scheme">` 與實際 `color-scheme`，切換全站色系，並通知 Chart.js 圖表實例重新渲染符合主題之網格與文字色彩

#### Scenario: Theme persistence
- **WHEN** 使用者變更主題偏好後重新整理頁面
- **THEN** 系統讀取本地儲存（localStorage）之主題設定，維持先前的深色或淺色模式狀態與對應 `color-scheme`

### Requirement: In-Drawer Signal Alignment Card and Quant Alert Controls
系統 SHALL 於個股分析抽屜（涵蓋桌面表格檢視、桌面卡片檢視與行動版卡片檢視三種版型）內呈現符合 WCAG AA 以上對比度之「多源訊號一致度摘要卡」與「量化告警快速訂閱控制項」（支援帶入有效 P10 防守價、P50 目標價、MA60 支撐或自訂價格），不得誤渲染於法說會行事曆等非標的抽屜彈窗內，且於靜態資源或樣板變更時同步更新全站與傳遞 ES 模組之資產版本指紋。

#### Scenario: Subscribing to a P10 defense alert from stock drawer
- **WHEN** 授權使用者展開具分位數預測之個股抽屜並點擊「訂閱跌破 P10 防守告警」
- **THEN** 系統送出動態 P10 告警規則，並於抽屜內即時顯示啟用中告警狀態與參考模型標記

#### Scenario: Alignment and alerts work consistently across all three drawer layouts without calendar leakage
- **WHEN** 使用者於桌面表格、桌面卡片或行動版卡片任一檢視展開抽屜切換至「訊號共振」或「價位告警」，或開啟法說會全月行事曆彈窗
- **THEN** 三種版型皆正確呈現並可操作對應子分頁內容，且法說會行事曆彈窗內不包含任何個股抽屜子分頁標記

## ADDED Requirements

### Requirement: Native Modal Dialog Focus and Scroll Containment
系統 SHALL 以常駐掛載之原生 `<dialog>` 搭配 `showModal()` 實作全域指令面板、告警中心、組合風險 X 光與法說會行事曆等阻斷式彈窗，支援原生 `Esc`、雙指標同區點放 light-dismiss、焦點捕捉與關閉後焦點歸還，並阻止開啟期間背景頁面捲動。

#### Scenario: Modal opens with focus containment and closes with focus restoration
- **WHEN** 使用者透過快捷鍵或按鈕開啟任一彈窗，並以 `Esc`、關閉鈕或 backdrop 點擊關閉
- **THEN** 彈窗開啟時焦點移入彈窗、背景頁面不隨滾輪或觸控捲動，關閉後 Vue 狀態同步重設並將焦點歸還至原觸發元件（若原元件已卸載則安全退回至主內容區）

#### Scenario: Dragging selection from inside modal to backdrop does not dismiss
- **WHEN** 使用者在彈窗內容區按下指標並拖曳放開於 backdrop 區域
- **THEN** 彈窗維持開啟不誤關閉，避免選取文字或操作控制項時意外遺失上下文

### Requirement: Keyboard Accessible Tables, Controls, and Heading Hierarchy
系統 SHALL 確保表格排序表頭採用 `<th :aria-sort><button type="button">`、水平捲動表格容器具 `tabindex="0"` 與具名 `role="region"`、頂部可點擊品牌標誌為原生鍵盤可操作控制項、表單 `<label>` 具備唯一 `for`/`id` 關聯，且每個啟用中分頁視圖維持單一可視主標題 `<h1>` 層級。

#### Scenario: Keyboard-only navigation across dashboard table, header, and forms
- **WHEN** 使用者僅以鍵盤 `Tab`、`Enter` 與 `Space` 操作品牌標誌、表格排序表頭、捲動區域與告警表單
- **THEN** 所有控制項具清晰 `:focus-visible` 焦點外框並觸發預期動作，且每個分頁僅存在單一可視 `<h1>` 標題

### Requirement: Verified CDN Scripts and Precompiled Styles Delivery
系統 SHALL 為所有版本固定之第三方 CDN `<script>`（Vue、Chart.js、Hammer.js、chartjs-plugin-zoom）設定 `integrity`、`crossorigin="anonymous"` 與 `referrerpolicy="no-referrer"`（動態 Google Fonts CSS 依 `referrerpolicy="no-referrer"` 載入並提供本機字型退回）；正式環境 HTML 改用預編譯靜態 CSS 取代 `cdn.tailwindcss.com`，統一靜態快取標頭並透過 import map 確保所有 ES 子模組隨資產指紋同步更新。

#### Scenario: Production HTML serves SRI-protected scripts, versioned ESM imports, and precompiled CSS
- **WHEN** 瀏覽器載入儀表板首頁與靜態資產
- **THEN** HTML 不含 `cdn.tailwindcss.com`，外部 CDN 腳本皆具 SRI 與匿名 CORS 屬性，且 `/static/` 下含 `?v=` 指紋之 CSS 與 ES 子模組在版本更新後立即取得最新資源

### Requirement: Unified Iconography and Precision Reduced Motion
系統 SHALL 收斂全站散落的手寫 inline SVG 與符號字元至單一圖示方案，逐步將 `html:not(.dark)` `!important` 規則替換為語意化 CSS Token 與精準屬性轉場；於 `prefers-reduced-motion: reduce` 偏好下僅停用空間位移、縮放與無限循環動畫，不全域套用 `0.01ms !important`。

#### Scenario: Reduced-motion user retains state transitions without continuous pulse
- **WHEN** 使用者作業系統開啟減少動態效果偏好
- **THEN** 呼吸閃爍與位移縮放動畫停用，而色彩、邊框與透明度等狀態回饋轉場仍可正常辨識

### Requirement: Scoped Locale Switching and Consistent Code Snippets
系統 SHALL 透過 `currentLocale` 雙語字典實際切換核心操作介面（頂部導覽列、市場時鐘標籤、總經概覽卡、6 維篩選器、訊號表格表頭與彈窗標題/操作鈕）並同步更新 `<html lang>`（後端原始新聞或公司摘要依來源語系保留）；同時讓 Agent Hub 頁面展示與複製按鈕共用同一份有效 Python 程式碼。

#### Scenario: Switching locale updates core UI labels and Agent Hub snippet stays valid
- **WHEN** 使用者切換 `繁中 / EN` 按鈕並重新整理頁面，或檢視與複製 Agent Hub Python 範例碼
- **THEN** 核心操作介面切換為對應語言並於重整後保留偏好；Agent Hub 畫面顯示與剪貼簿複製出完全一致且已解析實際 Origin URL 的合法 Python 程式碼
