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
