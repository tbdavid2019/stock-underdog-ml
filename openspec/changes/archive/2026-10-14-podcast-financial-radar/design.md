# Design

## Context

金融與科技領域的深度對話（如 All-In Podcast、Lex Fridman、Dwarkesh Patel、Bloomberg Odd Lots 等）蘊含前瞻性的產業變革線索與企業戰略意圖。目前市場公開產品展示了結合話題標籤與摘要的價值，但尚未將音訊逐字稿實體直接對齊量化交易系統的「時序行情」與「籌碼共振」。

本專案現有架構具備：
1. `core/config.py` 中的 `LLMConfig` 具備 3-Tier Fallback（Gemini 2.5 Flash、DeepSeek-V3、OpenAI-Mini）。
2. `data/duckdb_manager.py` 具備高效時序存儲、去重 upsert 與毫秒級過濾檢索能力。
3. `api/main.py` 與 `api/static/js/app.js` 具備金融終端風格、個股詳情抽屜（Stock Drawer）、籌碼分析與均線預測聯動。

本設計旨在以最小架構開銷，將外部 Podcast 音訊與 YouTube 逐字稿提煉為量化決策情報，並與現有個股抽屜深度閉環。更多動機說明參見 [proposal.md](proposal.md)。

## Goals / Non-Goals

**Goals:**
- **標準化多源採集**：對接 Podcast RSS 2.0 / iTunes Feed 與 YouTube Transcript API，具備 GUID 增量去重能力。
- **結構化 LLM 萃取**：利用 3-Tier LLM 精準提取 Host/Guest、150~200 字繁體中文 TL;DR、觀點標籤與上市公司金融實體。
- **實體與行情雙向映射**：將提到的公司（如 NVIDIA, Google, 美團, 台積電）精準映射至市場代碼（`NVDA`, `GOOGL`, `3690.HK`, `2330.TW`），並即時關聯當日漲跌幅 Badge。
- **終端連動抽屜（Killer Flow）**：前端提供熱門實體橫向滑動快篩，點擊股票 Badge 直接滑出個股詳情抽屜（Stock Drawer），整合技術買點與外資投信籌碼。
- **資源友好與高容錯**：採集與萃取採用非同步背景作業，本地 DuckDB 零拷貝查詢，避免高負載本機運算。

**Non-Goals:**
- 不在本機伺服器自建昂貴的 Whisper 模型推論叢集；逐字稿優先取用 YouTube Transcript / Jina Reader，無字幕時透過遠端雲端 API（Whisper API / Gemini Multimodal）完成。
- 不做音訊檔案的本地永久儲存，音訊播放直接使用官方 CDN / Enclosure URL。

## Decisions

### 1. 資料管道與逐字稿取得分層（Ingestion & Transcript Resolver）

- **頻道庫管理**：
  預置種子頻道庫（`config/podcast_seed_channels.json`），覆蓋美股與科技領域頂級頻道：
  - *美股科技核心*：All-In Podcast, Lex Fridman Podcast, Dwarkesh Podcast, The Logan Bartlett Show, Y Combinator, AI Engineer, Lenny's Podcast
  - *全球金融總經*：Bloomberg Businessweek, Bloomberg Odd Lots, The Economist, The Diary of a CEO
  - *華語科技財經*：小Lin說, 硅谷101
- **逐字稿解析階梯（Transcript Fallback Ladder）**：
  1. *第 1 級（首選）*：YouTube Transcript API（藉由單集關聯之 YouTube 影片獲取官方或自動生成字幕，解析速度快、成本近乎為零）。
  2. *第 2 級（備援）*：Jina Reader / create360.ai 代理抓取 YouTube / 官方網站逐字稿網頁。
  3. *第 3 級（兜底）*：若僅有純音訊 RSS 且無字幕，呼叫 Gemini Multimodal Audio（直接傳送音訊 URI 進行摘要）或 OpenAI Whisper API。
- **替代方案評估**：
  - *本機執行 faster-whisper*：違背本機資源緊繃之硬性限制，且處理 2 小時音訊會造成 CPU/GPU 滿載。

### 2. LLM 結構化輸出與實體對齊機制（PodcastExtractor）

- **提示詞架構（Prompt Architecture）**：
  使用 Pydantic 結構強制約束模型回傳 JSON：
  ```python
  class EpisodeExtractedInsight(BaseModel):
      host: str
      guest: Optional[str]
      tldr_zh_tw: str  # 150-200 字繁中重點摘要
      key_takeaways: List[str]  # 3-5 個核心要點
      mentioned_entities: List[str]  # 如 ["Claude", "OpenAI", "NVIDIA", "Google"]
      mentioned_tickers: List[str]  # 如 ["NVDA", "GOOGL", "2330.TW"]
      relevance_score: float  # 0.0 - 1.0 (與科技/金融投資關聯度)
  ```
- **代碼映射驗證過濾器（Ticker Verification Filter）**：
  - 為防止 LLM 幻覺（例如將普通英文單詞 "CAN", "IT", "SO" 當作股票代碼），所有提取出的代碼必須通過：
    1. 實體-代碼字典驗證（如 `SectorConfig.TICKER_SECTOR_MAP`）。
    2. `ticker_resolver.py` 之模糊校驗，確認為有效之美股、台股或港股標的。
- **3-Tier Fallback 執行**：
  呼叫 `core/config.py` 中的 `LLMConfig`：Gemini-2.5-Flash（超長上下文處理逐字稿最佳）→ DeepSeek-V3 → GPT-4o-mini。

### 3. DuckDB 資料表模型設計

於 `data/duckdb_manager.py` 初始化兩張新表：

```sql
-- 頻道資訊表
CREATE TABLE IF NOT EXISTS podcast_channels (
    channel_id VARCHAR PRIMARY KEY,      -- 頻道代號 (slug, 如 all-in, odd-lots)
    title VARCHAR NOT NULL,             -- 頻道名稱
    author VARCHAR,                     -- 創作者/出版機構
    description VARCHAR,                -- 頻道簡介
    image_url VARCHAR,                  -- 頻道封面 URL
    rss_url VARCHAR,                    -- 標準 Podcast RSS URL
    youtube_channel_url VARCHAR,        -- YouTube 頻道或播放清單 URL
    follower_count BIGINT DEFAULT 0,    -- 訂閱/追蹤人數
    episode_count INTEGER DEFAULT 0,    -- 已收錄單集數
    category VARCHAR,                   -- 分類 (tech, finance, macro)
    language VARCHAR,                   -- en / zh
    is_subscribed BOOLEAN DEFAULT TRUE, -- 是否啟用排程同步
    created_at VARCHAR,
    updated_at VARCHAR
);

-- 單集情報表
CREATE TABLE IF NOT EXISTS podcast_episodes (
    id VARCHAR PRIMARY KEY,             -- 單集唯一識別碼 (GUID 或 Hash)
    channel_id VARCHAR NOT NULL,        -- 外鍵關聯 podcast_channels
    title VARCHAR NOT NULL,             -- 單集標題
    published_at TIMESTAMP NOT NULL,    -- 發布時間 (標準 UTC)
    audio_url VARCHAR,                  -- 音訊 Enclosure URL
    youtube_url VARCHAR,                -- 對應之 YouTube 觀看連結
    duration_seconds INTEGER,           -- 音訊時長 (秒)
    language VARCHAR,                   -- en / zh
    host VARCHAR,                       -- 主持人
    guest VARCHAR,                      -- 特別來賓
    tldr_zh_tw VARCHAR,                 -- AI 繁體中文摘要 (150-200 字)
    key_takeaways VARCHAR,              -- JSON 字串陣列
    topic_tags VARCHAR,                 -- JSON 字串陣列 (如 ["Claude", "NVIDIA"])
    tickers VARCHAR,                    -- JSON 字串陣列 (如 ["NVDA", "GOOGL"])
    ticker_prices_json VARCHAR,         -- JSON 物件快照 (含代碼、價格、漲跌幅)
    status VARCHAR DEFAULT 'extracted', -- pending / extracted / error
    created_at VARCHAR,
    updated_at VARCHAR
);

-- 索引優化
CREATE INDEX IF NOT EXISTS idx_podcast_episodes_pub ON podcast_episodes (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_podcast_episodes_channel ON podcast_episodes (channel_id);
```

### 4. RESTful API 規格

在 `api/routes/podcast.py` 實作端點：
1. `GET /api/v1/podcast/episodes`:
   - 參數：`topic` (str, 篩選特定實體/話題), `ticker` (str, 篩選特定股票代碼), `channel_id` (str), `lang` (str), `limit` (預設 20), `offset` (預設 0)。
   - 行情增強（Price Enrichment）：響應時，動態對比 DuckDB 最新的收盤或報價快照，組裝即時漲跌幅 Badge 結構（`{ ticker: "NVDA", price: 125.4, change_pct: -0.72, direction: "down" }`）。
2. `GET /api/v1/podcast/channels`:
   - 返回已收錄的頻道清單、最新發布日期、單集統計與追蹤者人數。
3. `GET /api/v1/podcast/topics`:
   - 統計近期單集之 `topic_tags`，依出現頻次降序輸出熱門 Top 20 實體清單（如 Claude, OpenAI, NVIDIA, Google, Cursor, Meta, Gemini, Apple, AMD 等），供前端橫向滑動列使用。
4. `POST /api/v1/podcast/sync`:
   - 接受非同步背景任務觸發，支援單一頻道或全頻道同步。

### 5. 前端終端雷達介面與連動抽屜整合

- **頁籤架構**：
  在頂部導覽列（`Navbar` / `SubHeader`）新增「🎙️ 智慧情報雷達（Podcast）」一級頁籤。
- **實體標籤橫向滑動列（Horizontal Entity Slider）**：
  - 展示「🔥 熱門實體」，點擊切換 `selectedTopic`，即時更新單集情報清單。
- **單集情報卡片設計（Terminal Aesthetic）**：
  - 頻道 Logo、名稱與來源圖示（YouTube 標誌 / Podcast RSS 標誌）。
  - 發布時程（如 `3 天前`）、時長徽章（如 `1小時 42分`）、語言標籤（`EN` / `ZH`）。
  - 主持人與來賓徽章（`🎙️ Host: Lex Fridman` | `👤 Guest: Dario Amodei`）。
  - **AI 繁中精華觀點（TL;DR）**：高易讀性排版。
  - **核心觀點標籤（Key Takeaways）**：深色邊框膠囊。
  - **實時股票代碼 Badge（量化連動核心）**：
    - 渲染為綠漲紅跌（或美股綠漲紅跌配置）的互動晶片：`NVDA ↓ 0.72%`、`GOOGL ↑ 0.16%`。
    - **點擊行為**：呼叫 `app.js` 的 `toggleRowDrawer(ticker)`，直接於右側/底端滑出個股詳情抽屜，零跳轉無縫檢視技術指標、玄鐵均線買點與外資/投信籌碼！

## Risks / Trade-offs

- **[Risk 1: 逐字稿過長超出單次 LLM 上下文或費用過高]**  
  → **Mitigation**: 使用 Gemini 2.5 Flash 原生支援 100 萬 token 上下文，成本極低（百萬 token 約 $0.15）；長文本先進行段落關鍵詞粗篩或截取前 15,000 字精要部分。
- **[Risk 2: YouTube 字幕下載阻擋或 IP 限流]**  
  → **Mitigation**: 採用多層降級機制（YouTube Transcript API → Jina Reader proxy → 遠端 Gemini 音訊直讀），並設有本地檔案快取目錄 `cache/podcast/`。
- **[Risk 3: 股票代碼識別出現假陽性（例如將常用字 IT/AI/CAN 誤判為 Ticker）]**  
  → **Mitigation**: 引入嚴格的雙重實體驗證，僅當 LLM 同時識別出公司名稱實體且符合 `ticker_resolver` 與預置白名單時才標記為 Ticker Badge。
- **[Risk 4: 本機運算資源受限]**  
  → **Mitigation**: 不在本機執行任何音訊轉檔或本地神經網絡運算；所有耗時操作均為遠端 I/O 請求；單元驗證遵循輕量模式。

## Migration Plan

1. 於 `data/duckdb_manager.py` 增量執行 `CREATE TABLE IF NOT EXISTS podcast_channels` 與 `podcast_episodes`，完全向後相容。
2. 匯入預設種子頻道庫數據，保證初始化即可查看預製情報。
3. 部署 FastAPI 路由與靜態 HTML/JS 擴充，不影響既有操盤看板功能。
