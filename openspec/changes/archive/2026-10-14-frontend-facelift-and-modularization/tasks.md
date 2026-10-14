# Tasks

## 1. 基礎模組化與架構重構 (Modular Scaffolding & Architecture Decomposition)

- [x] 1.1 建立 `api/static/js/` 與 `api/templates/partials/` 目錄結構，建立 JS 模組骨架 (`api-client.js`, `chart-manager.js`, `app.js`)，並驗證靜態資源可被 FastAPI 正確載入。
- [x] 1.2 抽取 JavaScript 邏輯至獨立模組，分離 API 呼叫、Chart.js 控制器與 Vue 3 狀態管理，並驗證模組導入無語法錯誤。
- [x] 1.3 拆分 `api/templates/index.html` 為 Jinja2 partials，將主樣板縮減至外殼骨架並以 `{% include %}` 組合，驗證本地啟動時前端頁面 200 OK 且 Vue 正常初始化。

## 2. 「外觀拉皮 v.1」色彩體系與全域導覽 (Design Tokens, Theme & Header)

- [x] 2.1 更新 Tailwind 配置與色票變數（曜石黑 `#030302`、暗黑卡片 `#0D0E11`、琥珀金 `#F59E0B`、淺色 `#EEEEEE`），驗證深淺色切換與 localStorage 偏好記憶。
- [x] 2.2 重構頂部導覽列（`partials/header.html`），實作 `888 · Stock Quant` 品牌標題、膠囊藥丸 Tab 導覽與「顯示: 深色/淺色模式」切換開關，驗證頁籤切換與 URL query 雙向同步。

## 3. 操盤看板與訊號清單重構 (Dashboard Tab & Signal Table Facelift)

- [x] 3.1 依照 Figma 規格重構大盤宏觀卡片（VIX、標普/費半均線、宏觀環境）與總經/FedWatch/財報/Polymarket 緊湊網格，驗證各項總經指標動態渲染。
- [x] 3.2 依照 Figma 規格重構最新量化訊號表（深黑表頭、等寬數字對齊、彩色策略/法人 Badge 系統、歷史快捷按鈕與公司簡介展開 Drawer），驗證策略篩選過濾與展開收合行為。

## 4. 法人籌碼、券商主力雷達與個股歷史 (Institutional, Broker & History Tabs)

- [x] 4.1 依照 Figma 規格重構三大法人淨買賣超排行 TOP 30 表格，驗證排序按鈕、市場別篩選與紅綠張數對齊。
- [x] 4.2 依照 Figma 規格重構「券商分點主力進出雷達」，實作 TOP 8 主力券商多折線走勢圖（霓虹色系、深淺色自適應網格）與 TOP 15 買賣超分點雙欄清單，驗證分點資料載入與 Chart.js 動態渲染。
- [x] 4.3 依照 Figma 規格重構「個股歷史」時序量化預測紀錄、三大法人累計折線圖與每日籌碼明細，驗證股票代號速查與圖表繪製。

## 5. 整合驗證與文件更新 (Integration Verification & Documentation)

- [x] 5.1 執行全域自動化驗證（遵循本機資源嚴格硬性限制：以輕量 JS 語法校驗 `node -c` 與模板解析斷言完成靜態路由與渲染驗證）。
- [x] 5.2 依照專案規範更新 `docs/CHANGELOG.md` 與 `README.md`，記錄前端模組化與「外觀拉皮 v.1」視覺升級細節。

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Verify the archived result.
