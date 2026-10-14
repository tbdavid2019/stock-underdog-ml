# Proposal

## Why

Podcast 與科技訪談影音（如 Lex Fridman、All-In、Bloomberg Businessweek、Odd Lots、Dwarkesh、The Diary of a CEO、The Economist、Y Combinator、AI Engineer、Lenny's Podcast、小Lin說、硅谷101 等）已成為獲取科技巨頭底層技術突破、創投先導動態與全球宏觀趨勢的第一線非結構化情報源。然而，投資人目前面臨資訊過載與斷層：單集動輒 1 至 3 小時，缺乏自動化結構提煉，更無法直接對齊即時金融行情與量化交易系統。

本變更旨在打造「Podcast 財經與科技智慧情報雷達（AI Financial Podcast Radar）」，建立頂級影音與音訊情報的自動化管線，結合大語言模型進行結構化解析（提取 Host/Guest、繁體中文精準 TL;DR、核心觀點標籤），並自動識別提及的股票代號與概念實體，關聯即時漲跌幅 Badge 並直接連動本系統的個股時序與籌碼抽屜，將非結構化音訊情報轉化為高資訊密度的量化催化決策依據。

## What Changes

- **全球與華語頂級頻道庫與訂閱系統（Channel Library & Subscription Pipeline）**：
  - 預置並支援擴充美股與科技領域頂尖頻道：Lex Fridman, All-In, Bloomberg Businessweek, Odd Lots, Dwarkesh, The Diary of a CEO, The Economist, Y Combinator, AI Engineer, Lenny's Podcast, 小Lin說, 硅谷101 等。
  - 支援頻道追蹤、單集總數統計、追蹤者人數與 YouTube / Podcast RSS 雙來源標識。
  - 對接標準 Podcast RSS 2.0 / iTunes 規範解析，並結合 YouTube Transcript API / Jina Reader 取得影音字幕；無字幕單集支援 Whisper API / Gemini 影音直接摘要備援。
- **AI 單集深度提煉與金融實體聯結（AI Extraction & Financial Entity Linking）**：
  - **結構化主客識別**：精準萃取主持人（Host）與特別來賓（Guest，如 Sam Altman, Jensen Huang, Dario Amodei）。
  - **繁體中文 TL;DR**：產出 150~200 字之高品質繁中精煉摘要，快速掌握核心主旨。
  - **觀點標籤**：提取 3~5 個核心要點標籤（Key Takeaways，如「AI Agent架構」、「散熱供應鏈瓶頸」、「聯準會降息路徑」）。
  - **金融實體與即時行情聯結（Killer Feature）**：自動識別內容中提及的股票代號（如 `NVDA`, `GOOGL`, `TSM`, `3690.HK`），自動映射系統行情資料庫並關聯即時漲跌幅 Badge（如 `GOOGL ↑ 0.16%`、`NVDA ↓ 0.72%`、`3690.HK ↑ 0.7%`）。點擊 Badge 即可直接連動開啟系統內部的個股時序與籌碼抽屜（Stock Drawer）！
- **時序快取資料庫（DuckDB Podcast Schema）**：
  - 新增 `podcast_channels` 資料表：記錄頻道 ID、名稱、發布者、說明、封面圖、RSS URL、YouTube Channel ID、追蹤數、分類標籤與更新時戳。
  - 新增 `podcast_episodes` 資料表：記錄單集 GUID、頻道 ID、標題、音訊 Enclosure URL、發布時間、時長、語言、主持人、來賓、繁中 TL;DR、觀點標籤、關聯股票 Ticker 陣列、即時價格快照 JSON。
  - 建立 `(channel_id, published_at)` 與全文標籤索引，實現毫秒級查詢。
- **後端 REST API 服務**：
  - `GET /api/v1/podcast/episodes`：支援分頁、頻道篩選、主題標籤 (topic) 篩選、股票 Ticker 篩選與語言過濾。
  - `GET /api/v1/podcast/channels`：返回所有收錄頻道清單、單集統計與訂閱狀態。
  - `GET /api/v1/podcast/topics`：返回頂部熱門實體與主題標籤聚合排行榜（Claude, OpenAI, NVIDIA, Google, Cursor, Meta, Gemini...）及其關聯單集頻次。
  - `POST /api/v1/podcast/sync`：供排程器或管理者觸發 RSS 抓取與 LLM 萃取增量同步。
- **金融終端介面整合（Financial Terminal Radar UI）**：
  - **頂部實體標籤橫向滑動列（Horizontal Entity Slider）**：即時聚合當前熱門科技與金融實體標籤，點擊一鍵過濾單集。
  - **情報卡片佈局（Episode Intelligence Cards）**：高資訊密度佈局，顯示節目來源、發布時間、時長、語言徽章、Host/Guest 標記、繁中 TL;DR、觀點標籤。
  - **可互動股票代號 Badge**：即時漲跌色標，點擊無縫開啟個股籌碼抽屜，實現「聽見趨勢 → 立即驗證量化籌碼」的完整閉環。

## Capabilities

### New Capabilities
- `podcast-financial-radar`: 涵蓋全球與華語頂尖財經科技 Podcast/YouTube 逐字稿採集、LLM 結構化觀點與實體萃取、股票 Ticker 映射與漲跌幅 Badge 關聯、DuckDB 時序儲存、FastAPI 查詢端點與金融終端連動雷達之完整能力。

### Modified Capabilities
<!-- 無既有 specs 需求改變，此為全新之 Podcast 智慧情報雷達能力 -->

## Impact

- **Affected Code**:
  - `data/podcast_fetcher.py`（新模組：負責 RSS Feed、YouTube Transcript API 及音訊逐字稿採集）
  - `data/podcast_extractor.py`（新模組：利用 LLM 結構化輸出 Host/Guest、繁中 TL;DR、觀點標籤與 Ticker 映射）
  - `data/duckdb_manager.py`（擴充：新增 `podcast_channels` 與 `podcast_episodes` 表結構與查詢/upsert 介面）
  - `api/routes/podcast.py`（新模組：提供 episodes, channels, topics, sync 等 REST 端點）
  - `api/schemas.py`（擴充：新增 PodcastChannelItem, PodcastEpisodeItem, PodcastTopicItem 等 Pydantic 結構）
  - `api/main.py`（註冊 `podcast` 路由）
  - `api/static/js/api-client.js`（新增 Podcast 相關 API 呼叫）
  - `api/static/js/app.js`（新增 Podcast 雷達頁籤狀態管理、實體過濾器與股票抽屜聯動邏輯）
  - `api/templates/partials/podcast_radar.html` 或 `api/templates/index.html`（嵌入雷達頁籤與卡片視圖）
  - `run_daily.sh`（排程：每日定期觸發增量 Podcast 頻道同步與情報萃取）
- **APIs**:
  - 新增 `GET /api/v1/podcast/episodes`
  - 新增 `GET /api/v1/podcast/channels`
  - 新增 `GET /api/v1/podcast/topics`
  - 新增 `POST /api/v1/podcast/sync`
- **Dependencies**:
  - `feedparser`（輕量 RSS 2.0 / Atom 解析，純 Python 無編譯依賴）
  - 複用既有 `duckdb`, `requests`, `fastapi`, `pydantic` 與現行 `LLMConfig` 3-tier fallback 架構（Gemini-Flash / DeepSeek / OpenAI-Mini）。
