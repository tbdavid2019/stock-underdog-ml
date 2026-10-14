# Tasks

## 1. 資料庫儲存與種子頻道庫建立 (DuckDB Schema & Seed Channels)

- [x] 1.1 在 `data/duckdb_manager.py` 中新增 `podcast_channels` 與 `podcast_episodes` 表結構（包含頻道元數據、單集主客標記、繁中 TL;DR、觀點標籤、股票 Ticker 陣列與即時行情快照），並驗證資料庫初始化時表結構與索引正確建立。
- [x] 1.2 在 `DuckDBManager` 實作 `save_podcast_channel`、`save_podcast_episodes_batch`、`get_podcast_episodes` 與 `get_podcast_topics_aggregation` 檢索方法，支援唯一鍵 upsert 去重與多維篩選，並驗證儲存與檢索語法正確。
- [x] 1.3 建立 `config/podcast_seed_channels.json` 預置美股科技與全球總經頂級頻道（Lex Fridman, All-In, Bloomberg, Odd Lots, Dwarkesh, 小Lin說, 硅谷101 等），並驗證 JSON 格式與必填欄位完整性。

## 2. 多源逐字稿採集與 LLM 結構化萃取 (Ingestion & AI Extraction Pipeline)

- [x] 2.1 實作 `data/podcast_fetcher.py`：解析標準 Podcast RSS 2.0 / iTunes 規範與 YouTube 播放清單，萃取單集音訊連結、發布時間、時長與簡介，並驗證 Feed 解析模組能正確提取單集列表。
- [x] 2.2 實作逐字稿解析階梯：優先整合 YouTube Transcript API，降級支援 Jina Reader proxy 與 Gemini 音訊摘要，並驗證字軌解析與長文本分段邏輯。
- [x] 2.3 實作 `data/podcast_extractor.py`：利用現有 `LLMConfig` 3-Tier Fallback 提示詞架構，結構化輸出 Host/Guest、150~200 字繁中精煉 TL;DR、核心觀點標籤與實體，並驗證 LLM 結構化輸出解析。
- [x] 2.4 實作金融實體與股票 Ticker 映射校驗器：將提取之科技/金融實體精準映射至美股、台股與港股標的代碼（NVDA, GOOGL, 2330.TW, 3690.HK），過濾無效單詞，並驗證代碼映射白名單正確性。

## 3. 後端 REST API 與行情增強服務 (FastAPI REST Service)

- [x] 3.1 於 `api/schemas.py` 定義 Podcast 資料模型（`PodcastChannelItem`, `PodcastEpisodeItem`, `PodcastTopicItem`, `PodcastSyncResponse` 等），並驗證 Pydantic 結構與 JSON 序列化。
- [x] 3.2 於 `api/routes/podcast.py` 實作 `GET /api/v1/podcast/episodes` 端點，支援 `topic`、`ticker`、`channel_id`、`lang` 篩選與動態行情漲跌幅 Badge 增強，並驗證 API 路由返回結構。
- [x] 3.3 於 `api/routes/podcast.py` 實作 `GET /api/v1/podcast/channels` 與 `GET /api/v1/podcast/topics` 端點，提供收錄頻道統計與熱門話題頻次排行榜，並驗證聚合查詢正確性。
- [x] 3.4 於 `api/routes/podcast.py` 實作 `POST /api/v1/podcast/sync` 非同步背景同步端點，並在 `api/main.py` 註冊路由與驗證 Swagger 文件正確顯示。

## 4. 前端終端雷達介面與個股抽屜連動 (Frontend Radar UI & Drawer Integration)

- [x] 4.1 在導覽列（`Navbar` / `SubHeader`）新增「🎙️ Podcast 雷達」一級頁籤與切換狀態管理，並驗證頁籤切換與 URL 路由同步。
- [x] 4.2 實作頂部熱門實體橫向滑動快篩列（Horizontal Entity Slider），動態加載 Top 20 熱門實體並支援點擊一秒快篩，驗證實體按鈕互動與選取狀態。
- [x] 4.3 實作單集情報卡片（Episode Intelligence Cards），渲染節目來源、發布時間、時長、語言徽章、Host/Guest 標記、繁中 TL;DR、觀點標籤與音訊外鏈，並驗證金融終端高資訊密度佈局。
- [x] 4.4 實作股票代號 Badge 即時漲跌色標，點擊直接呼叫 `toggleRowDrawer(ticker)` 滑出個股時序與籌碼抽屜，驗證點擊 Badge 觸發抽屜之連動閉環。
- [x] 4.5 在 `api/static/js/api-client.js` 與 `api/static/js/app.js` 封裝 Podcast API 請求與快取管理，並驗證無資料時的優雅空狀態顯示。

## 5. 端到端流程整合與文檔同步 (End-to-End Flow & Documentation)

- [x] 5.1 整合排程腳本 `run_daily.sh`，加入每日定時增量同步 Podcast 頻道與萃取任務，驗證排程呼叫參數與日誌記錄。
- [x] 5.2 按照專案規範更新 `docs/CHANGELOG.md` 與 `README.md`，說明 Podcast 智慧情報雷達功能架構、API 端點與操作指南。
