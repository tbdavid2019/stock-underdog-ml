# Spec Delta: web-dashboard-ui

## MODIFIED Requirements

### Requirement: Quantitative Signal Dashboard Presentation
系統 SHALL 於「操盤看板」提供大盤宏觀狀態卡片、FedWatch/總經流動性事件網格，以及最新量化選股清單，支援多策略篩選（包括 Mistral Timestral 潛力股、雙大模型共振焦點與玄鐵均線波段）、標籤化顯示與公司簡介抽屜展開。

#### Scenario: Filtering signal candidate stocks
- **WHEN** 使用者於操盤看板點擊「雙大模型共振焦點」或「Timestral 潛力精選」等篩選標籤
- **THEN** 表格即時過濾顯示符合條件之候選股，並以高對比色顯示 Timestral P50 目標價與預期潛力百分比

#### Scenario: Expanding company details drawer
- **WHEN** 使用者點擊表格中之股票代號或名稱
- **THEN** 表格下方即時展開公司業務簡介抽屜，並提供 Yahoo 奇摩股市與 Goodinfo 外部連結

### Requirement: Stock Time-Series History and Cumulative Institutional Flow
系統 SHALL 於「個股歷史」提供單一標的之歷史量化預測紀錄、股價與雙 AI 基礎模型（Google TimesFM 與 Mistral Timestral）預測目標價軌跡線圖、三大法人累計買賣超趨勢折線圖，以及每日籌碼明細表格。

#### Scenario: Querying stock prediction and institutional history
- **WHEN** 使用者於個股歷史輸入代號（如 `2330.TW`）並執行查詢
- **THEN** 系統展示該個股歷次預測目標價（含 TimesFM 目標價與 Timestral P50 目標價與 P10-P90 信賴區間）、預期潛力、法人累計折線圖與逐日籌碼數據
