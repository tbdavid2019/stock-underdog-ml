# Tasks

> 遵循 `AGENTS.md` 開發與驗證規範：每個切片皆採「先新增重現/驗證測試 → 實作修改 → 執行窄範圍驗證 → 更新 `docs/CHANGELOG.md`（及必要之 `README.md`）」順序，保持小步可回滾。

## 1. 切片一（P0）：抽屜結構去重、Agent Hub 範例與核心雙語字典

- [x] 1.1 先於 `test/test_drawer_alignment_alerts_ui.py` 新增重現測試（斷言法說會行事曆彈窗內不含 `drawerActiveSubTab`、且桌面表格／桌面卡片／行動版卡片三種檢視皆包含 `alignment` 與 `alerts` 面板），接著抽取共用局部樣板 `api/templates/partials/_stock_drawer_tabs.html`（含唯一 `for`/`id` 前綴）並清理 `api/templates/partials/tab_dashboard.html` 內 3 處錯貼區塊，執行 `pytest test/test_drawer_alignment_alerts_ui.py` 驗證全數通過。
- [x] 1.2 先於 `test/test_command_palette_and_tabs_ui.py` 新增斷言驗證 `tab_agent_hub.html` 不含硬編碼字面 `{window.location.origin}` 且 `app.js` 具備可切換核心介面之 `t()` 字典與 `<html lang>` 同步，接著統一 `tab_agent_hub.html` 綁定 `{{ pythonSnippet }}` 並補齊頂部導覽、市場時鐘、總經概覽、6 維篩選器、表格表頭與彈窗核心雙語字典，執行 `pytest test/test_command_palette_and_tabs_ui.py` 驗證通過。
- [x] 1.3 更新 `docs/CHANGELOG.md` 記錄抽屜結構修復、雙語切換補齊與 Agent Hub 範例統一，確認切片一測試全數通過。

## 2. 切片二（P1）：原生 `<dialog>` 遷移與鍵盤／表格無障礙

- [x] 2.1 先更新 `test/test_command_palette_and_tabs_ui.py` 與 `test/test_risk_xray_ui.py`（改為驗證 `<dialog`、`showModal`、`@cancel`、`@close`、雙指標 light-dismiss 與 `overscroll-behavior: contain`，保留所有功能覆蓋），再將 `command_palette.html`、`alerts_modal.html`、`risk_xray_panel.html` 與 `tab_dashboard.html` 行事曆彈窗改為常駐 `<dialog>` + `app.js` 統一開關協調器（含焦點捕捉/歸還退回、背景鎖定及嵌套圖表 resize），執行 `pytest` 並依狀態矩陣於瀏覽器實測（開啟→滾輪/觸控不捲背景→內容拖曳放開不誤關→`Esc` 關閉歸還焦點→重複開關）。
- [x] 2.2 先新增無障礙結構測試（驗證 `<th :aria-sort><button>`、`.quant-table-wrap` 具 `tabindex="0"` 與具名 `role="region"`、Logo 控制項可鍵盤操作、`<label for>` 對應唯一 `id`、各啟用分頁僅存在單一可視 `<h1>`），再實作 `header.html`、`index.html`、`alerts_modal.html` 與 `tab_dashboard.html` 等調整並驗證 `pytest` 全數通過。
- [x] 2.3 更新 `docs/CHANGELOG.md` 記錄原生彈窗與無障礙升級，確認切片二相關測試全部通過。

## 3. 切片三（P1）：CDN SRI、ES 模組快取鏈、深淺色同步與預編譯 CSS

- [x] 3.1 先新增測試驗證 `index.html` 外部 CDN `<script>` 鎖定具體版本並含 `integrity`、`crossorigin="anonymous"`、`referrerpolicy="no-referrer"`，以及 `api/main.py` 針對帶 `?v=` 與無版本 `/static/` 資產之快取標頭，再實作：鎖定 Vue/Chart.js/Hammer/Zoom 版本與 SRI、`index.html` 導入 `<script type="importmap">` 映射 `app.js` 三支子模組至 `?v=__APP_VERSION__`、統一 `api/main.py` middleware 與 `NoCacheStaticFiles` 快取策略，並為 `chart-manager.js` 加入帶相容保護之 `.toSorted()`/`.toReversed()`，執行 `pytest test/test_api.py` 與 `npx -y modern-web-guidance@latest legacy-scan api/templates api/static/js --json` 驗證 SRI 與 JS 規則清零。
- [x] 3.2 建立 `tailwind.config.js`（完整保留原 `darkMode`、字型、字級與 `terminal` 色票及動態 `safelist`）、`api/static/css/input.css` 與 `scripts/build_css.sh`，編譯產出 `api/static/css/app.css` 並移除 `index.html` 與 `ticker_seo.html` 之 `cdn.tailwindcss.com`；同步於 `README.md` 與 `docs/CHANGELOG.md` 記錄前端樣式重建指令，並於雙主題桌面/手機視口確認版面無視覺回歸後通過 `pytest test/test_api.py test/test_public_ticker_seo.py`。
- [x] 3.3 實作深淺色與降動態現代化：於 `index.html` 與 `app.js` 同步 `<meta name="color-scheme">` 與 `document.documentElement.style.colorScheme`、對齊 `site.webmanifest` `theme_color` 為 `#FFB010`，移除全域 `0.01ms !important` 改為精準 `prefers-reduced-motion`（僅停用循環/空間動效），並於首屏外穩定長區塊套用 `content-visibility: auto` 與 `contain-intrinsic-size`，執行 `legacy-scan` 確認 `dark-mode-toggle` 規則清零並更新 `docs/CHANGELOG.md`。

## 4. 切片四（P2）：圖示／Token 收斂與公開個股頁一致性

- [x] 4.1 統一全站圖示家族（將 `tab_dashboard.html`、`tab_history.html`、`tab_broker.html` 散落的手寫 inline SVG 與 `★/☆/✕` 收斂至一致圖示方案），逐步將 `html:not(.dark)` `!important` 規則收斂至語意 CSS Token 並把 `transition-all` 替換為精準屬性轉場，透過雙主題視口檢查確認色彩與對比度符合 WCAG AA/AAA。
- [x] 4.2 先於 `test/test_public_ticker_seo.py` 補齊 `:focus-visible`、`overflow-x: clip`、`prefers-reduced-motion`、`text-wrap: balance` 與無 `cdn.tailwindcss.com` 斷言，再更新 `api/templates/ticker_seo.html`（維持固定深色無 JS SSR 契約）並驗證 `pytest test/test_public_ticker_seo.py` 全數通過。
- [x] 4.3 整合回歸驗證：執行全套 UI 與 API 測試（`pytest test/test_api.py test/test_command_palette_and_tabs_ui.py test/test_drawer_alignment_alerts_ui.py test/test_risk_xray_ui.py test/test_public_ticker_seo.py`）及 `npx -y modern-web-guidance@latest legacy-scan api/templates api/static/js --json`，確認 `docs/CHANGELOG.md` 與 `README.md` 同步完成。

## Workflow follow-up

- 所有實作與測試完成並通過審核後，執行 `openspec-archive-change` 將 `web-ui-hardening-and-modernization` 歸檔並同步更新主規格。
