# Design

## Context

本專案前端採 FastAPI + Jinja2（SPA 使用 `[[ ]]` 邊界保護 Vue 3 `{{ }}`；`/s/{ticker}` SSR 頁使用標準 `{{ }}` 並強制 `autoescape=True`）與原生 ES Module 驅動，運行於無 Node.js 的 Python 3.12 Docker 容器，且透過 `./api:/app/api` volume 掛載支援 `git pull` 即時更新（見 [Dockerfile](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/Dockerfile) 與 [docker-compose.yml](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/docker-compose.yml)）。本次技術設計回應 OMP 與 Codex 計畫審查所指出的規格完整性、ES 模組快取鏈、`<dialog>` 生命週期、Tailwind 設定保真、抽屜共用機制及測試契約等關鍵設計邊界。

## Goals / Non-Goals

**Goals:**
- 以零容器 Node.js 執行依賴的方式，透過提交版控的預編譯 CSS 與版本化 import map 解決執行期 Tailwind CDN 與快取一致性問題。
- 以單一 Jinja 共用局部樣板消除三處重複抽屜與誤植程式碼，並為四個彈窗建立可重現測試的原生 `<dialog>` 生命週期。
- 嚴格保持既有 `web-dashboard-ui` / `public-ticker-seo` 契約與 SSR 無 JS 閱讀性。

**Non-Goals:**
- 不引入 Vite / Webpack / Node 建置容器或改寫為 `.vue` SFC。
- 不變更後端 DuckDB 量化運算與 REST/MCP 資料契約。
- 公開個股頁 `ticker_seo.html` 維持固定深色主題（`class="dark"`），不額外新增淺色切換控制項。

## Decisions

### 1. Tailwind 預編譯建置與保真策略（鎖定版本 + 版控產物）
- **決策**：建立根目錄 [tailwind.config.js](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/tailwind.config.js)（完整保留現有 [index.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/index.html#L72-L98) 的 `darkMode: 'class'`、`fontSize` (`xs: 13px`, `sm: 14px`)、`fontFamily` (`sans`, `mono`) 與 `colors.terminal.*`），搭配 `api/static/css/input.css` 與鎖定版本的 `tailwindcss@3.4.x` 透過 `scripts/build_css.sh` 產出壓縮後的 `api/static/css/app.css` 並提交版控。
- **動態類別保護**：於 `tailwind.config.js` 明確列出 `content: ["./api/templates/**/*.html", "./api/static/js/**/*.js"]`，並將由程式動態組裝且未完整寫出字面值的 utility 類別放入 `safelist`。`index.html` 與 `ticker_seo.html` 共用此 CSS（`ticker_seo.html` 維持 `<html lang="zh-TW" class="dark">`）。
- **替代方案**：在 Docker 容器安裝 Node.js 於啟動時建置 → 破壞 `git pull` 免重建容器流程且增加映像檔負擔，故捨棄。

### 2. 兩層靜態快取統一與 ES Module Import Map
- **決策**：同時修正 [api/main.py](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/main.py) 中 `add_static_cache_control` middleware 與 `NoCacheStaticFiles.file_response` 兩處標頭邏輯：當 `/static/` 請求帶有與目前 `get_asset_fingerprint()` 相符（或具 `?v=`）之版本參數時回傳 `Cache-Control: public, max-age=86400`；無版本參數時維持 `no-cache, no-store, must-revalidate`；HTML 頁面維持 `Cache-Control: no-cache`。
- **傳遞依賴版本化**：移除 [app.js](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/static/js/app.js#L6-L8) 硬編碼的 `?v=2.6.0`，改用裸路徑 `/static/js/api-client.js`、`/static/js/chart-manager.js`、`/static/js/risk-xray.js`，並於 `index.html` 透過 `<script type="importmap">` 將三支模組映射至 `/static/js/<name>.js?v=__APP_VERSION__`。同時將 `get_asset_fingerprint()` 掃描範圍確保涵蓋 `api/static/css/`（並於資產變更或長註解版號時更新雜湊/mtime）。
- **替代方案**：僅改 `NoCacheStaticFiles` 或僅在 HTML 入口加 `?v=` → 會被 middleware 覆寫或因 `app.js` 硬編碼 `?v=2.6.0` 導致子模組繼續使用舊快取，故捨棄。

### 3. 抽屜子分頁單一來源（Jinja 共用局部樣板）
- **決策**：採用 **Jinja 局部樣板** `api/templates/partials/_stock_drawer_tabs.html` 作為唯一來源（放棄 runtime JS 字串模板元件，以保持 HTML 編輯直覺與 Vue 模板可讀性），於「桌面表格列抽屜」、「桌面卡片抽屜」與「行動版卡片抽屜」透過 Jinja `{% with drawer_prefix='table'|'card'|'mobile' %}`（或等效 `set` 變數）引入。
- **隔離與唯一 ID**：每個抽屜表單之 `<label :for="...">` 與 `<select/input :id="...">` 以 `drawer_prefix + '-' + normalizeTicker(item.ticker)` 產生唯一 ID，避免多版型並存 DOM 時 ID 衝突；徹底刪除 [tab_dashboard.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/partials/tab_dashboard.html) 內 `tender` 迴圈及法說會行事曆 `day.calls` 內錯貼的 3 份副本。

### 4. 原生 `<dialog>` 生命週期、雙指標 Light-Dismiss 與焦點/捲動管理
- **決策**：移除四個彈窗外層的 `v-if="show*Modal"`（讓 `<dialog ref="...">` 常駐掛載於 DOM，內部重型清單可使用 `v-if="show*Modal"` 延遲渲染內容以節省效能），所有開關一律經由統一協調 helper `openNativeDialog(stateKey, refName)` / `closeNativeDialog(stateKey, refName)` 操作：
  1. **開啟**：記錄 `lastFocusedElement = document.activeElement` → 設 `this[stateKey] = true` → `this.$nextTick(() => { if (dialog && !dialog.open) dialog.showModal(); focusInitialControl(); resizeNestedCharts(); })`。
  2. **原生關閉同步**：監聽 `@cancel.prevent="closeNativeDialog(...)"` 與 `@close="syncDialogClosed(...)"` 確保按 `Esc`、程式呼叫 `close()` 或瀏覽器返回手勢時，Vue 布林狀態與 DOM `.open` 絕不同步失真。
  3. **安全 Light-Dismiss**：以 `@pointerdown` 記錄 `pointerId` 與點擊點是否位於 `dialog.getBoundingClientRect()` 之外，僅當相同 `pointerId` 於 `@pointerup` 亦落於外部（且未發生 `@pointercancel`）時才關閉，防止拖曳選取文字誤關彈窗。
  4. **背景捲動鎖定**：併用 `dialog:modal { overscroll-behavior: contain; }`、`dialog::backdrop { overflow: hidden; overscroll-behavior: contain; }` 以及 `html:has(dialog[open]) { overflow: hidden; scrollbar-gutter: stable; }`（針對不支援 `dialog::backdrop` overscroll 的瀏覽器提供可靠鎖定）。
  5. **焦點歸還退回**：關閉後若 `lastFocusedElement?.isConnected` 為真則 `focus()`，否則退回至 `main` 或頂部搜尋觸發鈕。

### 5. 深淺色 `color-scheme` 同步、CDN SRI 與減弱動態
- **深淺色同步**：於 [index.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/index.html) `<head>` 放置 `<meta name="color-scheme" content="light dark">`，內聯防閃爍腳本與 `app.js` `applyTheme()` 同步更新 `meta.content`（`'dark'` / `'light'`）與 `document.documentElement.style.colorScheme`；CSS 採用分離選擇器 `:root:has(> head > meta[name="color-scheme"][content="dark"]) { color-scheme: dark; }` 與 `...[content="light"] { color-scheme: light; }`，並將 [site.webmanifest](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/static/site.webmanifest) `theme_color` 統一為 `#FFB010`。
- **CDN SRI 邊界**：為 Vue（鎖定 `3.5.x` 具體版本）、Chart.js（鎖定 `4.4.x` 具體版本）、`hammerjs@2.0.8`、`chartjs-plugin-zoom@2.0.1` 計算並鎖定 `integrity="sha384-..."`、`crossorigin="anonymous"`、`referrerpolicy="no-referrer"`；Google Fonts CSS 屬動態 UA 回應無法固定 SRI，改標註為規格允許之例外，保留 `referrerpolicy="no-referrer"`、將 `Material Symbols` 由 `display=block` 改為 `display=swap` 或漸進式本機圖示字型/SVG 退回。
- **減弱動態與 `content-visibility`**：移除全域 `0.01ms !important`，改以 `@media (prefers-reduced-motion: reduce)` 關閉 `.animate-pulse`、`.animate-ping`、`.animate-spin`（或降速為極低頻）與位移/縮放 hover；`content-visibility: auto; contain-intrinsic-size: auto 420px;` 僅套用於首屏下方獨立長區塊（Podcast 單集區、Discovery 論文區、History 驗證成績區），**不套用於**含 sticky 表頭的主表格列。
- **ES2023 相容基線**：`chart-manager.js` `.toSorted()` / `.toReversed()` 目標基線為 Baseline 2023+（Chrome 110+ / Safari 16+ / Firefox 115+），並以安全補助（`Array.prototype.toSorted ? ... : [...arr].sort(...)`）確保不因舊內嵌 WebView 中斷。

### 6. 核心 UI 雙語字典與 Agent Hub 單一範例來源
- **雙語字典**：於 `app.js` 定義 `I18N_MESSAGES` 字典與 `t(key)` 輔助函式，覆蓋頂部導覽、市場時鐘標籤、總經概覽、6 維篩選器標題/按鈕、訊號表頭、彈窗標題/主要按鈕，切換時同步更新 `document.documentElement.lang`（`'zh-TW'` / `'en'`）並持久化於 `localStorage`；動態新聞與資料庫公司描述保留來源語系。
- **Agent Hub 範例**：將 [tab_agent_hub.html](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/templates/partials/tab_agent_hub.html#L167-L174) `<pre><code>` 改為直接綁定 `{{ pythonSnippet }}`，與 [app.js](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/api/static/js/app.js#L426) 複製按鈕共用單一資料來源。

## Risks / Trade-offs

- **[預編譯 CSS 漏編動態類別]** → 透過 `tailwind.config.js` `safelist` + CI/測試掃描模板與 JS 產出清單驗證，並在替換 CDN 前以瀏覽器雙主題截圖比對。
- **[既有 UI 測試斷言舊有 DOM 屬性]** → 如 `test_command_palette_and_tabs_ui.py` 檢查 `@keydown.window.esc`；同步更新測試改驗 `<dialog`、`@cancel`、`showModal` 與 `aria-label`，不減損任何測試覆蓋度。
- **[瀏覽器互動行為未經單純 HTML 字串測試覆蓋]** → `tasks.md` 針對 `<dialog>` 焦點、背景鎖定與抽屜 6 分頁切換補入瀏覽器/Playwright 驗證步驟（無 Playwright 時提供明確可重現驗收矩陣）。
