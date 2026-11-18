# web-dashboard-ui Specification

## Purpose
提供基於金融終端高資訊密度規格的前端儀表板介面，支援深淺色雙主題切換、模組化導覽與多維量化/籌碼/走勢視覺化呈現。

## Requirements

### Requirement: Modular Dashboard View Routing and Tab Navigation
系統 SHALL 提供響應式頂部導覽列，支援「操盤看板」、「法人籌碼」、「個股歷史」與「分享」等頁面導覽切換，並自動同步瀏覽器 URL Query 參數與狀態歷程。

#### Scenario: User switches views via navigation tabs
- **WHEN** 使用者點選頂部「法人籌碼」或「個股歷史」頁籤
- **THEN** 系統即時切換至對應視圖，無需重新載入全頁，並將 URL 參數更新為 `?tab=institutional` 或 `?tab=history`

#### Scenario: Direct access via URL query parameter
- **WHEN** 使用者直接透過瀏覽器開啟帶有 `?tab=history&ticker=2330.TW` 的網址
- **THEN** 系統初始化時自動切換至個股歷史頁籤，並自動觸發台積電歷史資料載入

### Requirement: Financial Terminal Theme and Dual-Mode Switching
系統 SHALL 支援符合彭博終端調性之深色模式（曜石黑底色、琥珀金重點色）與淺色模式（高對比灰白底色），並提供頂部切換開關與圖表連動更新。

#### Scenario: Toggling between Dark and Light mode
- **WHEN** 使用者點擊頂部「顯示: 深色模式 / 淺色模式」切換開關
- **THEN** 系統切換 HTML root class，切換全站色系，並通知 Chart.js 圖表實例重新渲染符合主題之網格與文字色彩

#### Scenario: Theme persistence
- **WHEN** 使用者變更主題偏好後重新整理頁面
- **THEN** 系統讀取本地儲存（localStorage）之主題設定，維持先前的深色或淺色模式狀態

### Requirement: Quantitative Signal Dashboard Presentation
系統 SHALL 於「操盤看板」提供大盤宏觀狀態卡片、FedWatch/總經流動性事件網格，以及最新量化選股清單，支援多策略篩選（包括 Mistral Timestral 潛力股、雙大模型共振焦點與玄鐵均線波段）、標籤化顯示與公司簡介抽屜展開。

#### Scenario: Filtering signal candidate stocks
- **WHEN** 使用者於操盤看板點擊「雙大模型共振焦點」或「Timestral 潛力精選」等篩選標籤
- **THEN** 表格即時過濾顯示符合條件之候選股，並以高對比色顯示 Timestral P50 目標價與預期潛力百分比

#### Scenario: Expanding company details drawer
- **WHEN** 使用者點擊表格中之股票代號或名稱
- **THEN** 表格下方即時展開公司業務簡介抽屜，並提供 Yahoo 奇摩股市與 Goodinfo 外部連結

### Requirement: Institutional and Broker Radar Analysis Presentation
系統 SHALL 於「法人籌碼」提供三大法人買賣超排行清單、TOP 8 目標券商累計買賣超趨勢多折線圖，以及 TOP 15 主力買超與賣超分點雙欄比較清單。

#### Scenario: Viewing institutional top flow ranking
- **WHEN** 使用者選擇排序條件（三大法人合計、外資、投信）與買賣超方向
- **THEN** 系統顯示 TOP 30 法人買賣超排行，並以紅綠兩色清楚區分淨買賣金額與張數

#### Scenario: Analyzing broker accumulation radar
- **WHEN** 使用者輸入個股代號並指定交易日區間查詢券商分點
- **THEN** 系統繪製 TOP 8 券商多折線趨勢圖，並於下方渲染買賣超 TOP 15 分點明細表

### Requirement: Stock Time-Series History and Cumulative Institutional Flow
系統 SHALL 於「個股歷史」提供單一標的之歷史量化預測紀錄、股價與雙 AI 基礎模型（Google TimesFM 與 Mistral Timestral）預測目標價軌跡線圖、三大法人累計買賣超趨勢折線圖，以及每日籌碼明細表格。

#### Scenario: Querying stock prediction and institutional history
- **WHEN** 使用者於個股歷史輸入代號（如 `2330.TW`）並執行查詢
- **THEN** 系統展示該個股歷次預測目標價（含 TimesFM 目標價與 Timestral P50 目標價與 P10-P90 信賴區間）、預期潛力、法人累計折線圖與逐日籌碼數據

### Requirement: AI Tech Radar and Discovery View Presentation
The system SHALL provide a dedicated "前沿發現 (Discovery)" view in the Bloomberg terminal theme, presenting metrics KPIs, mini-trend sparklines, and a 4/2/1 responsive intelligence grid.

#### Scenario: Display responsive discovery intelligence matrix
- **WHEN** user navigates to the Discovery tab on desktop or mobile
- **THEN** system renders 4 sections (Candidates, In-Pool Reinforcement, arXiv Papers, Signals) adjusting from 4 columns on desktop to 1 column on mobile

#### Scenario: Expand candidate stock research dossier
- **WHEN** user toggles the "研究卡" action button on a candidate card
- **THEN** system lazy-renders an in-card dossier displaying score breakdown, thesis notes, and mapped supply chain anchors

#### Scenario: Navigate from candidate card to stock analysis
- **WHEN** user clicks on a tracked ticker within discovery items
- **THEN** system switches view or opens analysis drawer for that stock without page reload

### Requirement: Mobile Navigation Adaptation for Discovery Tab
The system SHALL adapt the fixed bottom mobile navigation bar to support horizontal touch scrolling when accommodating five tabs without clipping.

#### Scenario: Touch scroll mobile navigation bar
- **WHEN** user views the interface on a mobile viewport <= 375px
- **THEN** all five tabs remain accessible via smooth horizontal scrolling with active state indicator intact

### Requirement: Reliability and model performance views
系統 SHALL 提供股票可信度榜與模型戰績榜，支援市場、20/60/120 交易日、版本與步長篩選；顯示命中率、Wilson 下界、MAPE、基準改善、樣本、覆蓋與更新時間，區分正式排名、樣本累積及 shadow。

#### Scenario: Explore a reliable stock
- **WHEN** 使用者點選股票榜中的標的
- **THEN** 開啟個股頁並保留篩選脈絡，單股成績與榜單使用相同後端驗證結果

#### Scenario: Compare shadow candidate
- **WHEN** MOIRAI-2 尚在研究shadow或資料不足
- **THEN** 顯示候選身分與累積進度，與 active 模型共同樣本戰績並列，避免顯示為選股推薦

### Requirement: Multi-model history and health explanation
系統 SHALL 在個股歷史提供可依模型選擇的預測曲線與 PK，清楚顯示正式目標日期、pending、legacy 限制、停用原因及實際參與合成的模型與權重。

#### Scenario: Missing session and suspended model
- **WHEN** 某日實際價格缺失或模型被停用
- **THEN** 顯示缺失／停用狀態並保留原始預測，PK 與共振使用後端資格結果

#### Scenario: TimesFM version replacement
- **WHEN** 使用者檢視新版預測與舊版歷史
- **THEN** 新輸出顯示TimesFM 3.0、MOIRAI-2與Timestral的實際版本／用途資格；2.5僅在歷史篩選呈現，不列為現行執行模型

### Requirement: Simulation performance and risk presentation
系統 SHALL 提供模擬帳戶與回測結果的淨值、淨報酬、最大回撤、成本、交易紀錄及部位風險，分幣別顯示並標記模擬、資料延遲及歷史重播限制；手機與深淺主題均能讀取必要資訊。

#### Scenario: Simulation has no fills
- **WHEN** 帳戶尚無成交或資料暫時過期
- **THEN** 顯示原因、最後更新時間及空狀態，績效不足處以 unavailable 呈現

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

### Requirement: Watchlist Risk X-Ray Diagnostic Panel
前端必須（SHALL）以獨立樣板與腳本提供一鍵「組合風險 X 光健檢」互動面板，將當前 DuckDB 或匿名 `localStorage` 自選股代號陣列送往 `POST /api/v1/risk/xray`，清晰呈現雙口徑波動率、階梯式 95%/99% VaR 與 CVaR、風險槽 HHI、支援負對沖值之歐拉風險貢獻橫條、高相關配對與跨幣別/日曆錯位提示，且雙主題配色嚴格符合 WCAG AA/AAA 對比度規範（嚴禁白字疊琥珀金 `#FFB010`）。

#### Scenario: Anonymous localStorage watchlist and negative hedging PCV rendered cleanly
- **WHEN** 公開訪客以本地自選股點擊「組合風險 X 光健檢」且其中包含負相關對沖標的（`PCV < 0`）
- **THEN** 面板無需寫入權限即可渲染風險指標，並將負貢獻度標示為風險減縮項目而不破版

#### Scenario: Insufficient history or tail samples rendered without broken NaN labels
- **WHEN** 後端回傳 `insufficient_history` 或因視窗未滿 100 日將 `var_99` 標記為 `insufficient_tail_samples`
- **THEN** 面板顯示清楚的樣本門檻提示文案，絕不顯示 `NaN%`、`undefined` 或誤導性 `0.00%`

### Requirement: Statistical Significance Badge Presentation
前端必須（SHALL）在歷史模擬或績效驗證摘要區塊支援顯示統計顯著性檢定標章（包含 Bootstrap Sharpe 95% 信賴區間與 `p_value_return` 狀態，涵蓋顯著、未達顯著、樣本不足與零變異狀態），協助操盤者區分真實超額表現與短期隨機運氣。

#### Scenario: Non-significant, small-sample, or zero-variance state clearly distinguished
- **WHEN** 檢定結果之 `p_value_return >= 0.05` 或狀態為 `insufficient_samples` / `degenerate_zero_variance`
- **THEN** 介面以中性抑制色標示對應狀態，避免將少數幾筆幸運交易誤導呈現為高可靠度策略
