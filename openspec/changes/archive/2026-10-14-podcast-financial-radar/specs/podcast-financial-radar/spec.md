# Spec Delta

## Purpose

提供全球頂級財經與科技音訊逐字稿自動採集、LLM 結構化主客識別、繁體中文觀點精煉、股票 Ticker 映射與即時漲跌連動個股抽屜的完整雷達系統。

## ADDED Requirements

### Requirement: Top-Tier Channel Library and Feed Ingestion
系統 SHALL 維護全球與華語頂級財經科技 Podcast 頻道庫，並依據標準 Podcast RSS 2.0 與 YouTube 管道定期擷取最新發布單集之元資料。

#### Scenario: Scheduled ingestion of top podcast channels
- **WHEN** 系統執行定期頻道同步任務時
- **THEN** 系統遍歷預置頻道庫（包含 All-In, Lex Fridman, Bloomberg, Odd Lots, 小Lin說, 硅谷101 等），取得最新單集標題、發布時戳、音訊連結與時長

#### Scenario: Ingestion deduplication by GUID
- **WHEN** 擷取到已存在於資料庫之單集 GUID 時
- **THEN** 系統跳過重複抓取，僅於元資料有變更時更新既有單集記錄

### Requirement: Multi-Source Transcript Resolution
系統 SHALL 依序自 YouTube 逐字稿、快取逐字稿或語音辨識服務取得單集完整文字內容，供後續大語言模型進行深度提煉。

#### Scenario: YouTube transcript extraction
- **WHEN** 待處理單集具備對應之 YouTube 影片連結或 ID 時
- **THEN** 系統優先請求 YouTube 字幕軌並解析時間軸逐字稿文字

#### Scenario: Audio transcript fallback
- **WHEN** 單集缺乏官方字幕或 YouTube 字幕時
- **THEN** 系統自動降級呼叫 Whisper API 或 Gemini 音訊摘要介面進行語音解析

### Requirement: LLM Structured Insights and TLDR Extraction
系統 SHALL 透過大語言模型結構化輸出單集之主持人與來賓、150 至 200 字繁體中文精煉重點摘要（TL;DR）以及 3 至 5 個核心觀點標籤。

#### Scenario: Structured Host and Guest extraction
- **WHEN** LLM 分析單集標題與逐字稿時
- **THEN** 系統結構化產出明確標註之主持人（Host）與受訪特別來賓（Guest）名稱

#### Scenario: Traditional Chinese TLDR generation
- **WHEN** LLM 提煉單集內容時
- **THEN** 系統生成 150 至 200 字長度之繁體中文精準重點摘要，說明核心論點與商業意涵

#### Scenario: Core key takeaway tagging
- **WHEN** LLM 完成單集深度提煉時
- **THEN** 系統輸出 3 至 5 個代表性核心觀點標籤（如「模型蒸餾技術」、「邊緣AI散熱」、「降息週期影響」）

### Requirement: Financial Entity and Stock Ticker Linking
系統 SHALL 自動識別 Podcast 逐字稿與摘要中提及之上市公司或金融實體，精準映射至美股、台股或港股代碼，並關聯即時漲跌幅快照。

#### Scenario: Entity to ticker mapping
- **WHEN** 單集內容提及 Google、NVIDIA、美團或台積電等實體時
- **THEN** 系統自動映射至對應之標的代碼（如 GOOGL, NVDA, 3690.HK, 2330.TW）

#### Scenario: Real-time price change badge attachment
- **WHEN** 系統組裝單集情報資料時
- **THEN** 系統自動自量化行情庫計算標的最新價格與當日漲跌百分比，格式化為包含漲跌方向與顏色的 Badge 數據

### Requirement: DuckDB Podcast Storage and Indexing
系統 SHALL 在 DuckDB 中持久化維護 `podcast_channels` 與 `podcast_episodes` 兩張資料表，並為實體標籤、股票代碼及發布時間建立高效檢索索引。

#### Scenario: Storing extracted episode with entity metadata
- **WHEN** 單集完成 LLM 萃取與實體聯結後
- **THEN** 系統將單集完整結構、實體標籤陣列與股票代碼陣列原子化寫入 `podcast_episodes` 表

#### Scenario: Filter episodes by entity topic or ticker
- **WHEN** 應用程式透過實體標籤或股票代號查詢單集時
- **THEN** 系統能以次秒級延遲自 DuckDB 返回符合條件且依發布時間倒序排列之單集列表

### Requirement: RESTful Radar and Channel Discovery APIs
系統 SHALL 提供 FastAPI REST 端點，支援最新單集分頁與實體過濾、頻道庫清單、熱門話題排行聚合以及排程同步觸發。

#### Scenario: Querying episodes with topic filter
- **WHEN** 客戶端向 `/api/v1/podcast/episodes` 發送請求並指定 `topic=OpenAI` 時
- **THEN** 系統返回包含 OpenAI 實體標籤之單集分頁列表，每集包含 TL;DR、來賓資訊與股票代碼 Badge

#### Scenario: Querying trending topics aggregation
- **WHEN** 客戶端向 `/api/v1/podcast/topics` 發送請求時
- **THEN** 系統返回近期所有單集中出現頻次最高之熱門實體標籤排行榜（如 Claude, OpenAI, NVIDIA, Cursor, Google）

#### Scenario: Triggering podcast sync
- **WHEN** 管理者向 `/api/v1/podcast/sync` 發送 POST 請求時
- **THEN** 系統啟動後台任務同步所有訂閱頻道，抓取新單集並執行 LLM 萃取

### Requirement: Terminal Interface and Stock Drawer Integration
系統 SHALL 在前端金融終端提供熱門實體橫向滑動快篩列、單集情報卡片以及點擊股票 Badge 聯動個股時序與籌碼抽屜之互動功能。

#### Scenario: Filtering by horizontal entity slider
- **WHEN** 使用者在終端頂部橫向滑動列點擊特定實體標籤（如「NVIDIA」）時
- **THEN** 頁面即時快篩顯示所有探討該實體之單集情報卡片

#### Scenario: Clicking stock ticker badge opens stock drawer
- **WHEN** 使用者在單集卡片中點擊股票代碼 Badge（如 `NVDA ↓ 0.72%`）時
- **THEN** 終端介面無縫滑出個股詳情抽屜（Stock Drawer），載入該標的之歷史走勢、均線回調訊號與法人籌碼數據
