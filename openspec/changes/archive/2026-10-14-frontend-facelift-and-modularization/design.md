# Design

## Context

目前前端代碼全數集中於 `api/templates/index.html`（3,011 行），維護困難且難以做多人協同切版。專案架構為 FastAPI 驅動，使用 Jinja2 渲染 HTML 並掛載 `/static` 目錄，前端採用 Vue 3 CDN + Tailwind CSS CDN + Chart.js CDN，具備無需 Node.js build step 的輕量運行特性。

設計團隊交付了最新 Figma「外觀拉皮 v.1」（Node `283:19024`），將系統升級為深色曜石黑與琥珀金搭配的 Bloomberg 金融終端視覺，並支援雙主題（深色/淺色）與緊湊的排版規格。

## Goals / Non-Goals

**Goals:**
- **單一檔案模組化**：將 3,000+ 行的 `index.html` 解耦為結構清晰的 Jinja2 子模板（`api/templates/partials/`）與原生 ES 模組（`api/static/js/`），大幅提升維護性。
- **100% 還原 Figma 外觀拉皮規格**：
  - 實作深色曜石黑（`#030302`）與淺色灰白（`#EEEEEE`）雙主題切換。
  - 導入琥珀金（`#F59E0B`）為核心視覺亮點（Active Tab、關鍵指標）。
  - 優化資訊排版密度：卡片圓角、間距收斂，等寬金融字體（`JetBrains Mono` + `tabular-nums`）。
  - 重構 Signal Table 標籤體系（多策略共振、玄鐵波段、低PE/PB、法人買超等彩色 Badges）。
  - 優化 Chart.js 折線圖視覺（霓虹多色趨勢線、深色細格線、主題切換即時重繪）。
- **零 Build Step 部署保障**：維持原生靜態與 CDN 機制，無需引入 npm/webpack/vite，容器啟動與本機測試保持極速。
- **既有功能零回退**：確保所有即時 API 呼叫、URL 參數同步、展開公司抽屜、法人篩選等功能正常運作。

**Non-Goals:**
- 不改動任何 FastAPI 後端 API 邏輯或 DuckDB 查詢。
- 不引入重型前端構建工具鏈（如 Vite、Node.js build pipeline）。

## Decisions

### 1. 模板拆分策略：Jinja2 `{% include %}` 子模板
- **決策**：在 `api/templates/partials/` 下建立模組化 HTML 片段：
  - `header.html`：品牌 Logo、膠囊導覽 Tab、深淺色切換鈕。
  - `tab_dashboard.html`：宏觀卡片、總經/FedWatch 網格、量化訊號表與公司抽屜。
  - `tab_institutional.html`：三大法人買賣超排行 TOP 30。
  - `tab_broker.html`：券商分點主力進出雷達圖與 TOP 15 買賣超分點表。
  - `tab_history.html`：個股歷史量化預測紀錄、法人累計折線圖與逐日明細。
  - `footer.html`：頁尾資訊與回到頂部按鈕。
- **優點**：`index.html` 將縮減至 80 行以內，只保留 HTML 骨架與 `<main>` 包含指令，每個視圖檔案獨立且易於維護。

### 2. 腳本架構：原生 ES Modules 職責分離
- **決策**：將前端 JavaScript 抽取至 `api/static/js/`，分為三個獨立模組：
  - `api-client.js`：純函式庫，專職呼叫後端 `/api/*` REST API 並統一處理錯誤與重試。
  - `chart-manager.js`：管理 Chart.js 實例生命週期、配色方案（深淺色網格、霓虹曲線）與 `updateTheme()` 重新著色方法。
  - `app.js`：Vue 3 根應用程式，負責狀態管理（`activeTab`, `isDark`, `filters`）、URL 同步與模板事件綁定。
- **替代方案考慮**：Vue SFC（.vue 檔案）。放棄原因為需要引入 Node.js 編譯打包環境，破壞目前輕量化的 Docker 與部署流程。原生 ES Module 在現代瀏覽器已完全原生支援。

### 3. 設計系統與 Tailwind 配置更新
- **決策**：於 `tailwind.config` 內收斂色彩與字體 Token：
  - `theme.colors.terminal`:
    - `dark-bg`: `#030302`
    - `dark-card`: `#0D0E11`
    - `dark-card-inner`: `#16181D`
    - `dark-border`: `#262626`
    - `amber`: `#F59E0B`
    - `amber-hover`: `#D97706`
    - `profit`: `#24D97A`（上漲綠/紅依規範配置）
    - `loss`: `#D33C3C`
  - 字體設置：數字預設套用 `font-feature-settings: 'tnum' 1`，確保表格數值上下對齊。

## Risks / Trade-offs

- **[Risk] Vue 數據綁定在拆分後遺失或作用域問題**
  → **Mitigation**：Jinja2 `{% include %}` 純粹是伺服器端字串拼接，產生的 DOM 仍然位於同一 Vue 根實例（`#app`）下，因此 Vue 的所有響應式狀態、計算屬性與方法完全不受影響。
- **[Risk] 深淺色切換時 Chart.js 圖表未同步變色**
  → **Mitigation**：在 `app.js` 的 `toggleTheme()` 方法中觸發 `chartManager.updateTheme(isDark)`，依據當前模式更新 `options.scales.x/y.grid.color` 與文字顏色並調用 `chart.update()`。
- **[Risk] CDN 快取導致使用者瀏覽器未即時更新 JS/CSS**
  → **Mitigation**：在 `index.html` 引入靜態模組時附加版本號參數（如 `/static/js/app.js?v=2.4.8`）。
