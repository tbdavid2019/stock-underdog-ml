# Proposal

## Why

目前前端界面全數集中於單一檔案 `api/templates/index.html`（超過 3,000 行），混雜 HTML 結構、Tailwind CSS 類別、全域 Vue 狀態、API 請求與 Chart.js 初始化邏輯，維護性與可讀性極低。同時，團隊交付了最新的 Figma 設計稿「外觀拉皮 v.1」，將整體調性提升至 Bloomberg/TradingView 風格的金融終端質感（曜石黑、琥珀金、高資訊密度、雙主題色彩規範）。

透過本變更，我們將前端架構進行模組化拆分（將模板、樣式設定、API 交互、圖表控制器解耦），並完整實作 Figma 外觀拉皮規格，達到極佳的代碼可維護性與頂級視覺工藝。

## What Changes

- **前端單一檔案架構模組化拆分**：
  - 將 `api/templates/index.html` 中的龐大內容拆解為 Jinja2 子模板（Header、Dashboard、Institutional、Broker、History、Footer/Drawer）。
  - 將 JavaScript 邏輯抽取至 `api/static/js/` 模組：
    - `api-client.js`：封裝後端 FastAPI 請求與錯誤處理。
    - `chart-manager.js`：管理 Chart.js 折線圖生命週期、深淺色動態更新與配色配置。
    - `app.js`：Vue 3 根應用、狀態管理、URL Query 參數同步與 Tab 切換。
  - 將 Tailwind 樣式設定與主題色票抽取至獨立樣式或配置。
- **實作「外觀拉皮 v.1」視覺規範 (Figma 283:19024)**：
  - **色彩系統升級**：支援深色 (Obsidian `#030302` / Amber `#F59E0B`) 與淺色 (`#EEEEEE` / `#FFFFFF`) 雙主題切換，提供符合金融數據可讀性的色階與狀態標籤（買賣超紅綠對比、數值等寬字型）。
  - **頂部導覽列 (Header)**：實現 `888 · Stock Quant` 品牌標題、膠囊藥丸 Tab（操盤看板、法人籌碼、個股歷史、分享）以及含文字狀態的深淺色切換按鈕。
  - **操盤看板 (Dashboard)**：微調大盤宏觀卡片、FedWatch/總經事件網格、最新量化訊號表（統一 Badge 樣式、股票展開 Drawer、歷史按鈕快捷跳轉）。
  - **法人籌碼與主力雷達 (Institutional & Broker)**：重構三大法人淨買賣超排行 TOP 30 表格、主力進出雷達 TOP 8 券商多折線走勢圖與 TOP 15 買賣超分點雙欄清單。
  - **個股歷史 (History)**：重繪個股量化預測紀錄、三大法人累計折線圖與每日籌碼明細表格。

## Capabilities

### New Capabilities
- `web-dashboard-ui`: 提供基於 Vue 3 與 Tailwind CSS 的金融終端使用者介面，涵蓋操盤看板、法人籌碼、券商分點與個股歷史時序視覺化，並具備模組化模板/腳本架構與深淺色雙主題切換能力。

### Modified Capabilities
<!-- 無既有 specs 需求改變，此為新增之 Web 前端介面能力 -->

## Impact

- **Affected Code**:
  - `api/templates/index.html`：重構為輕量的主佈局入口。
  - 新增 `api/templates/partials/`：包含各功能頁籤與對話框子模板。
  - 新增 `api/static/js/`：`api-client.js`, `chart-manager.js`, `app.js`。
  - `api/main.py`：確保靜態檔案路徑與 Jinja2 模板渲染正常支援 partials。
- **APIs**: 無破壞性變更，完全複用現有 FastAPI `/api/*` 端點。
- **Dependencies**: 無額外 npm 打包依賴，保持原生純淨之 ESM/CDN 載入機制（FastAPI + Jinja2 + Vue 3 + Tailwind CDN + Chart.js CDN），維持快速部署與零 build step 特性。
