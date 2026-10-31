# Proposal

## Why

目前主流投資者在研究 AI 與半導體時，常依賴如 `stocks.mastersgo.cc` 等整理板，但現有工具的產業鏈主題與上下游關係幾乎皆為人工手寫的靜態資料，無法隨市場演進自動更新，亦缺乏量化時序目標價與籌碼驗證深度。
透過串接台灣櫃買中心官方「產業價值鏈資訊平台 (`ic.tpex.org.tw`)」與美國 SEC EDGAR 10-K 公開主要客戶/供應商揭露資料，我們能在零 API Key 門檻下，打造全自動動態更新的跨市場「AI 產業價值鏈與供應鏈知識圖譜」，並無縫結合現有 Timestral/TimesFM 雙時序大模型預測與法人籌碼，提供真正動態、兼具專業層級「上游 ➔ 中游 ➔ 下游」階層畫布的視覺化導航。

## What Changes

- **增設跨市場產業鏈資料擷取器 (`data/industry_chain_fetcher.py`)**：
  - 串接 TPEx 官方產業價值鏈平台公開 JSON API（涵蓋 AI 伺服器、半導體、光通訊 CPO、散熱液冷、先進封裝等 30+ 產業）。
  - 自動解析並正規化產業之「上游 (Upstream)」、「中游 (Midstream)」、「下游 (Downstream)」層級、主要業務角色與公司代號。
  - 支援 SEC EDGAR 10-K 關鍵客戶揭露（Major Customers >10%）與 Wikidata 開源知識圖譜關係三元組抽取。
- **DuckDB 產業鏈關係知識圖譜儲存 (`data/duckdb_manager.py`)**：
  - 新增 `tw_industry_chains`（產業鏈定義與主題分類）與 `tw_industry_nodes`（個股所屬產業鏈、層級、角色說明）。
  - 新增 `tw_industry_relations`（實體關係邊：供應鏈關係、資本開支傳導、大客戶依賴度、主題關聯）。
- **FastAPI REST API 產業鏈端點 (`api/routes/industry_chain.py`)**：
  - `GET /api/v1/industry-chain/themes`：取得所有可用研究主題（如 HBM/儲存、CPO/光互聯、電力/液冷、先進封裝、AI 晶片、機器人）及其標的計數。
  - `GET /api/v1/industry-chain/graph`：支援主題過濾 (`theme`) 與關係類型過濾，回傳上/中/下游三欄結構化節點，並自動注入即時 AI 預測目標價、T+1 開獎狀態與法人連續買超天數。
- **前端動態三欄產業鏈圖譜畫布 (`tab_industry_chain.html` 或整合於探索雷達)**：
  - 借鏡乾淨的「上游 (Upstream) ➔ 中游 (Midstream) ➔ 下游 (Downstream)」三欄垂直卡片版面。
  - 互動式 SVG 貝茲曲線動態繪製連線（支援點擊節點高亮其直接傳導鏈，其餘半透明淡化）。
  - 頂部支援主題膠囊按鈕（全部主題、HBM/儲存、CPO/光互聯、電力/液冷、先進封裝、機器人等）。
  - 節點卡片直接嵌入最新股價、當日漲跌幅、AI 雙模型目標價與籌碼訊號標籤。

## Capabilities

### New Capabilities
- `industry-supply-chain-graph`: 跨市場產業價值鏈動態圖譜模組，負責從 TPEx / SEC EDGAR 擷取上中下游公司角色與供應鏈關係邊，並在 DuckDB 中建立關聯圖譜。

### Modified Capabilities
- `fastapi-rest-service`: 新增產業鏈圖譜查詢 API (`/api/v1/industry-chain/themes` 與 `/api/v1/industry-chain/graph`)，支援即時量化與法人指標動態關聯。
- `web-dashboard-ui`: 新增「產業圖譜」互動分頁或視覺模組，提供三欄階層畫布、SVG 貝茲連線、滑鼠滾輪縮放平移與主題過濾。

## Impact

- **公開 API 依賴**：完全**不需要**申請任何付費或審核型 API Key！全部使用開放免 Key 來源（TPEx 官方 API、SEC EDGAR 官方免費開放介面、Wikidata）。
- **資料庫變更**：DuckDB 新增 3 張產業鏈相關資料表，採本機快取與增量同步架構，完全不影響現有 `predictions` 或 `tw_daily_bars`。
- **向下相容性**：完全向後相容現有 API 與看板結構。
