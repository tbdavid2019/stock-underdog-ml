# Proposal

## Why

整合 `impeccable`、`modern-web-guidance`、`design-taste-frontend` 與 `hallmark` 四項前端健檢結果，目前網站存在多處影響可用性、無障礙與維護性的問題：`tab_dashboard.html`（約 399 KB，模板總原始檔約 860 KB）因抽屜子分頁錯貼導致行動版與卡片版無法檢視「訊號共振 / 價位告警」且污染法說會行事曆彈窗；`繁中 / EN` 語言切換鈕僅翻譯單一盤勢函式；`tab_agent_hub.html` 展示之 Python 範例與 `app.js` 實際複製字串不一致；四個彈窗缺少原生焦點管理與背景捲動隔離；正式環境仍使用執行期 `cdn.tailwindcss.com` 且第三方 CDN 腳本缺少 SRI 雜湊、深淺色未同步 `color-scheme`。本次變更將分層修復結構缺陷、補齊現代 Web 平台無障礙與安全底座，並統一設計 Token 與圖示系統。

## What Changes

- **修復模板結構與功能完整性問題**：
  - 移除 [tab_dashboard.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/partials/tab_dashboard.html) 中錯貼於卡片檢視 `tender` 子分頁內（2 處）與法說會全月行事曆彈窗內（1 處）的 `alignment` / `alerts` 區塊，抽離為單一共用 Jinja 局部樣板掛載於桌面表格、桌面卡片與行動版卡片三種抽屜，確保六個子分頁完整運作且互不污染。
  - 補齊 [header.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/partials/header.html) 與 [app.js](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/static/js/app.js) 核心介面（導覽列、市場時鐘、總經卡片、6 維篩選器、表格表頭、彈窗標題與按鈕）之 `繁中 / EN` 雙語字典與 `<html lang>` 同步；統一 [tab_agent_hub.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/partials/tab_agent_hub.html) 展示區塊與 `app.js` `pythonSnippet` 來源，消除畫面上顯示 `{window.location.origin}` 佔位字面的問題。
- **現代 Web 無障礙、彈窗與表格互動升級**：
  - 將指令面板、告警中心、風險 X 光健檢與法說會行事曆四個彈窗遷移至常駐掛載的原生 `<dialog>` + `showModal()`，透過單一開關協調器同步 Vue 狀態、`Esc`／雙指標座標驗證之 backdrop light-dismiss、焦點捕捉與歸還（含觸發元素已移除之安全退回），並處理背景捲動鎖定與圖表尺寸重算。
  - 升級表格排序表頭為 `<th :aria-sort><button type="button">`、`.quant-table-wrap` 補齊 `tabindex="0"` 與具名 `role="region"`、Logo 控制項改為原生鍵盤可操作元素、表單 `<label>` 補齊唯一 `for`/`id` 關聯，並統一各分頁僅保留單一可視 `<h1>` 層級。
- **平台安全、快取一致性與預編譯 CSS 現代化**：
  - 為固定版本號的第三方 CDN `<script>`（Vue 3、Chart.js、Hammer.js、chartjs-plugin-zoom）補齊 `integrity="sha384-..."`、`crossorigin="anonymous"` 與 `referrerpolicy="no-referrer"`（動態 Google Fonts CSS 標記為例外並以 `referrerpolicy="no-referrer"` 與字型退回策略處理）。
  - 導入本機 Tailwind CLI 預編譯靜態 CSS（保留既有 `tailwind.config` 擴充並提交產物至 `api/static/css/`，無須在 Python Docker 容器安裝 Node.js）以取代 `cdn.tailwindcss.com`；同步統一 `api/main.py` 兩層靜態快取標頭與 `app.js` 傳遞 ES 模組依賴之版本指紋（import map）。
  - 同步 `<meta name="color-scheme">` 與實際 `color-scheme` 屬性、對齊 `site.webmanifest` 主題色，將 `prefers-reduced-motion: reduce` 改為精準停用空間位移／循環動畫，並於穩定的首屏外長區塊套用 `content-visibility: auto` 與 `contain-intrinsic-size`，將 `chart-manager.js` 複製排序升級為 ES2023 `.toSorted()` / `.toReversed()`。
- **設計系統 Token 與獨立公開頁一致性收斂**：
  - 統一全站圖示家族至單一方案，逐步以語意化 CSS 變數取代 `html:not(.dark)` `!important` 強制覆寫與 `transition-all`。
  - 維持公開個股頁 [ticker_seo.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/ticker_seo.html) 既有固定深色無 JS SSR 契約，補齊 `:focus-visible`、`overflow-x: clip`、`prefers-reduced-motion`、標題 `text-wrap: balance` 與共用預編譯樣式表。

## Capabilities

### New Capabilities
- （無新增獨立 capability，全部收斂於既有儀表板與公開個股頁規格強化）

### Modified Capabilities
- `web-dashboard-ui`: 強化儀表板雙主題 `color-scheme` 與 manifest 一致性、三版型抽屜子分頁完整性與隔離、原生 `<dialog>` 焦點與捲動隔離、鍵盤與表格無障礙、SRI 與預編譯樣式交付、統一圖示與精準減弱動態，以及限定範圍之雙語切換與一致程式碼範例。
- `public-ticker-seo`: 補齊 `/s/{ticker}` 公開個股頁在固定深色模式下的 `:focus-visible` 焦點環、`overflow-x: clip`、減弱動態支援、`text-wrap: balance` 與預編譯樣式表替換。

## Impact

- **受影響檔案**：
  - 樣板：`api/templates/index.html`、`api/templates/ticker_seo.html`、`api/templates/partials/header.html`、`api/templates/partials/footer.html`、`api/templates/partials/command_palette.html`、`api/templates/partials/alerts_modal.html`、`api/templates/partials/risk_xray_panel.html`、`api/templates/partials/tab_dashboard.html`、`api/templates/partials/tab_history.html`、`api/templates/partials/tab_agent_hub.html`、新增共用抽屜樣板 `api/templates/partials/_stock_drawer_tabs.html` 等。
  - 靜態資產與設定：`api/static/site.webmanifest`、`api/static/js/app.js`、`api/static/js/chart-manager.js`、新增 `tailwind.config.js`、`api/static/css/input.css`、`api/static/css/app.css`、`scripts/build_css.sh`。
  - 後端與測試：`api/main.py`（快取標頭統一）、`test/test_command_palette_and_tabs_ui.py`、`test/test_drawer_alignment_alerts_ui.py`、`test/test_risk_xray_ui.py`、`test/test_public_ticker_seo.py`、`test/test_api.py` 及新增回歸測試；同步更新 `docs/CHANGELOG.md` 與 `README.md`。
