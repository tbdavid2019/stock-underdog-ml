# Tasks

## 1. 資料擷取與 DuckDB 儲存層

- [x] 1.1 建立 `data/industry_chain_fetcher.py` 抓取並解析 TPEx「產業價值鏈平台」上中下游分類與公司代號，並驗證單元測試通過
- [x] 1.2 建立 SEC EDGAR 10-K 大客戶與主要供應商解析器與本地備援快取，並驗證離線降級能力
- [x] 1.3 擴充 `data/duckdb_manager.py` 建立 `tw_industry_chains`、`tw_industry_nodes` 與 `tw_industry_relations` 資料表與查詢方法，並驗證讀寫測試

## 2. 後端 REST API 與量化指標融合

- [x] 2.1 建立 `api/routes/industry_chain.py` 並實作 `GET /api/v1/industry-chain/themes` 與 `GET /api/v1/industry-chain/graph` 端點
- [x] 2.2 在圖譜節點中動態關聯最新收盤價、漲跌幅、Timestral/TimesFM 目標價與法人連買天數，並驗證數據融合正確性
- [x] 2.3 註冊路由至 `api/main.py` 並撰寫 API 端點單元測試

## 3. 前端三欄式產業圖譜畫布與主題過濾

- [x] 3.1 於 `api/templates/partials/tab_industry_chain.html` 建立三欄階層（上游 ➔ 中游 ➔ 下游）卡片結構與主題膠囊按鈕
- [x] 3.2 於 `api/static/js/app.js` 與前端腳本實作動態 SVG 貝茲連線計算、滾輪縮放平移與點擊節點高亮聚焦
- [x] 3.3 於導覽列整合「產業圖譜」分頁與行動端適配，並確保符合 WCAG 易讀性標準

## 4. 驗證與整合上線

- [x] 4.1 執行全專案單元測試確保 100% 通過
- [x] 4.2 更新 `CHANGELOG.md`、`docs/CHANGELOG.md` 與 `README.md`
- [x] 4.3 進行 Codex CLI 與 OpenCode CLI 雙重代碼審查並修復建議項目
- [ ] 4.4 遠端伺服器部署與線上視覺驗證
