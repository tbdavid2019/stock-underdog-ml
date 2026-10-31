# Design

## Context

目前業界類似工具（如 `stocks.mastersgo.cc`）採用硬編碼前端常數呈現產業鏈圖譜，無法動態適應產業演進。本專案將利用台灣櫃買中心官方公開資料與美國 SEC EDGAR 10-K 公開客戶揭露，以完全免費、零 API Key 的方式在後端建立動態圖譜，並結合現有雙時序模型目標價與法人籌碼。詳細動機請參閱 `proposal.md`。

## Goals / Non-Goals

**Goals:**
- **零 Key、免審核之公開資料源串接**：串接 TPEx「產業價值鏈資訊平台」API 與 SEC 10-K 主要客戶文字揭露。
- **DuckDB 關聯圖譜架構**：在本地 DuckDB 中建立高效率節點與關係邊儲存，支援按主題快速檢索。
- **即時量化與籌碼指標動態注入**：在產出圖譜時，動態串接股票之即時收盤價、漲跌幅、Timestral/TimesFM 目標價與投信連買天數。
- **三欄階層畫布（上游/中游/下游）**：以前端 SVG 貝茲曲線與主題膠囊按鈕打造流暢好讀之產業傳導圖譜。

**Non-Goals:**
- 不引入重型 Neo4j 等獨立圖資料庫伺服器（使用輕量 DuckDB SQL 關聯即可滿足毫秒級拓撲查詢）。
- 不取代現有宏觀與個股歷史頁面，而是作為強大的互補導航工具。

## Decisions

### 1. 公開資料來源選擇 (TPEx + SEC EDGAR)
- **選擇**：
  - 台股：櫃買中心與證交所共同建置之「產業價值鏈資訊平台 (`ic.tpex.org.tw`)」，其將半導體、AI 伺服器、散熱、光通信等 30+ 產業明確劃分「上游 ➔ 中游 ➔ 下游」，為官方權威標準。
  - 美股：美國 SEC EDGAR 官方 10-K 年報中 "Item 1. Business - Major Customers" 之法定大客戶 (>10% 營收) 揭露。
- **替代方案評估**：
  - 人工手寫維護：靈活性差，無法擴展。
  - 付費金融終端（彭博/Refinitiv 供應鏈數據）：成本極高，一般個人與開源社群無法負擔。

### 2. DuckDB 圖譜儲存模型
- 在 DuckDB 建立三張表格：
  - `tw_industry_chains`：`chain_id`, `name`, `category`, `description`
  - `tw_industry_nodes`：`ticker`, `chain_id`, `layer` (上游/中游/下游), `role_name`, `source`
  - `tw_industry_relations`：`source_ticker`, `target_ticker`, `relation_type` (供應鏈/資本開支/大客戶/主題), `weight`, `evidence`

### 3. 三欄階層（上游 ➔ 中游 ➔ 下游）與 SVG 貝茲曲線
- **版面**：採用 CSS Flex/Grid 三欄卡片佈局，視覺結構遠優於傳統力導向蜘蛛網圖。
- **連線**：使用 SVG `<path>` 元素動態計算起點卡片右側錨點 `(x1, y1)` 與終點卡片左側錨點 `(x2, y2)`，生成三次貝茲曲線 `M x1 y1 C (x1+dx) y1, (x2-dx) y2, x2 y2`。

## Risks / Trade-offs

- **[Risk]** 上游 TPEx 網站改版或連線逾時。  
  → **Mitigation**：所有抓取資料均寫入本地 DuckDB 與快取目錄（14 天快取有效），遇網路失敗自動降級使用現存圖譜。
- **[Risk]** 大量連線造成前端 DOM 渲染負擔。  
  → **Mitigation**：預設僅渲染當前選定主題之關係線，點擊卡片時高亮相關傳導鏈，其餘淡化或隱藏弱關係。
