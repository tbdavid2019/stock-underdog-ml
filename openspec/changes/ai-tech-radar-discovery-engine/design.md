# Design: AI 產業鏈前沿挖掘機 (AI Tech Radar & Discovery Engine)

## Context

參照 `proposal.md`。當前系統已具備完善的 DuckDB 列式資料庫、FastAPI REST 服務、雙 Foundation Model（TimesFM + Timestral）推論引擎以及 100 頂級 Podcast TL;DR 情報流。然而，現有選股偏向行情啟動後的右側技術與籌碼追蹤，缺乏學術前沿（arXiv）到硬體瓶頸、再到台美供應鏈核心受惠標的的「左側前瞻雷達」。

在經過 Codex 與 OpenCode 雙審架構複核後，本設計對 DuckDB 併發寫入鎖、快取一致性、代號白名單、評分公式確定性及手機端導覽列佈局進行完整強化。

## Goals / Non-Goals

**Goals:**
- **前沿學術與產業訊號擷取**：自動化收集 arXiv（`cs.AI`, `cs.RO`, `cs.LG`, `cs.CV`）與科技硬體發布，經由輕量 LLM 結構化萃取技術突破點與供應鏈瓶頸。
- **代號正規化與白名單硬防護**：嚴格過濾與驗證台美股代號，杜絕下市、錯碼或 LLM 幻覺標的。
- **確定性 7 維多因子評分**：固化權重公式，產出數值在 `[0, 100]` 的總分，並依照嚴格閾值分流至四級推薦標籤。
- **資料庫與快取雙層原子發布**：DuckDB 交易成功 commit 後，再以暫存檔搭配 `os.replace` 原子替換靜態 JSON 快取，消除分裂腦風險。
- **DuckDB 讀寫連線分離與防鎖機制**：FastAPI 端點強制以 `read_only=True` 存取；Pipeline 寫入端具備連線重試；排程增加 `flock` 檔案鎖。
- **彭博終端調性之 4/2/1 斷點儀表板**：無縫融入現有深淺雙色主題，手機端底部導覽列支援橫向滾動適配，桌面 4 欄、平板 2 欄、手機單欄堆疊。

**Non-Goals:**
- 不進行高頻 WebSocket 逐筆報價推播（採開盤前批次計算 + 日內輪詢更新）。
- 不涉及自動下單交易接口（定位為高信賴度量化決策與探索雷達）。

## Decisions

### 1. DuckDB 連線隔離與併發寫入防撞鎖
- **決策**：
  1. FastAPI 的資料讀取一律採用 `duckdb.connect("stocks.duckdb", read_only=True)`，確保永遠不持有寫入鎖，不阻塞背景批次作業。
  2. 資料管線寫入集中由 [`data/duckdb_manager.py`](file:///Users/david/Documents/git/tbdavid2019/stock-underdog-ml/data/duckdb_manager.py) 管理，寫入時啟用短暫交易（Transaction），並加入指數退避重試（最多 3 次，間隔 1s, 2s, 4s）。
  3. 每日排程腳本 `run_daily.sh` 透過 `flock -n /tmp/stock_discovery.lock` 防止重疊執行。
- **理由**：徹底解決 DuckDB 單一寫入進程（Single-writer process lock）引發的 `Resource temporarily unavailable` 崩潰問題。

### 2. 雙層持久化之原子替換 (Two-Phase Atomic Commit)
- **決策**：
  1. Pipeline 計算完畢後，第一階段開啟 DuckDB Transaction 批次寫入 `discovery_candidates`、`discovery_papers`、`discovery_signals` 與 `discovery_history`。
  2. DuckDB Transaction 成功 commit 後，第二階段將對應之 JSON 資料寫入暫存檔（`cache/discovery/*.tmp`），隨後使用 `os.replace()` 原子性替換為正式快取檔（`cache/discovery/*.json`）。
  3. 每次發布皆帶有統一的 `run_id`（UUID）與 `as_of`（ISO-8601），確保資料庫與快取嚴格一致。
- **理由**：杜絕寫入中斷造成的「DB 更新但快取落後」或「快取更新但 DB 失敗」之分裂狀態。

### 3. 代號正規化與白名單硬檢驗 (Ticker Canonicalization)
- **決策**：
  1. 建立 `core/ticker_canonicalizer.py`，將台股代號統一規格化為純數字代碼（例如 `2330.TW` $\rightarrow$ `2330`），美股代號統一轉大寫（`nvda` $\rightarrow$ `NVDA`）。
  2. 所有自 LLM 或論文映射出的標的，必須與 DuckDB 中既有的股票 Universe 清單做白名單比對；未命中的標的直接過濾並記錄於 `reject_signals`。
- **理由**：徹底防範 LLM 幻覺產生非上市公司代碼，保證下游行情關聯 100% 可用。

### 4. 確定性 7 維多因子評分公式 (Deterministic Scoring Formula)
- **決策**：
  各因子歸一化為 `[0, 100]` 區間，總分計算公式定義如下：
  $$\text{Total Score} = \text{clamp}\Big(0.20 \times S_{\text{src}} + 0.20 \times S_{\text{purity}} + 0.20 \times S_{\text{paper}} + 0.10 \times S_{\text{novelty}} + 0.10 \times S_{\text{sent}} + 0.10 \times S_{\text{setup}} - 0.10 \times S_{\text{risk}},\, 0,\, 100\Big)$$
  - $S_{\text{src}}$: 消息來源權重分（arXiv 頂會頂級分類 100，產業官網公告 80，二級媒體 50）
  - $S_{\text{purity}}$: 題材純度分（核心業務營收關聯度）
  - $S_{\text{paper}}$: 前沿技術訊號分（解決關鍵硬體架構瓶頸權重）
  - $S_{\text{novelty}}$: 新穎度分（首次出現在主題雷達給予加分）
  - $S_{\text{sent}}$: 語意情緒評分（正面發展 100，中立 50）
  - $S_{\text{setup}}$: 量價技術位置分（當前價格處於回調支撐區給予高分）
  - $S_{\text{risk}}$: 執行與商業化風險扣分（離商用過遠或估值過高）
  
  **分流閾值標準**：
  - `propose_add`：未在現有池中，且 $\text{Total Score} \ge 75$ 且 $S_{\text{purity}} \ge 60$。
  - `observe`：未在現有池中，且 $60 \le \text{Total Score} < 75$。
  - `already_in_pool`：標的已在現有池中，且有新增前沿訊號支持。
  - `reject`：$\text{Total Score} < 50$ 或 $S_{\text{purity}} < 40$。

### 5. 供應鏈映射設定檔外部化
- **決策**：將前沿瓶頸與台美供應鏈關聯抽出至 `config/discovery_supply_chain.json`，由 `core/supply_chain_map.py` 讀取載入。
- **理由**：研究員日後更新或擴充主題（如矽光子、全固態電池）時，只需維護 JSON 設定，不需修改 Python 業務邏輯代碼。

### 6. 前端導覽列容納度與響應式斷點
- **決策**：
  1. 手機端底部導覽列（`header.html`）改為支援橫向平滑滾動（`overflow-x: auto; -webkit-overflow-scrolling: touch`）或彈性自適應，確保 5 個 Tab 在 375px 小螢幕下排版平整不破版。
  2. 4 欄式矩陣斷點規則：
     - 桌面寬螢幕（$\ge 1280\text{px}$）：完整 4 欄並排。
     - 平板（$768\text{px} \sim 1279\text{px}$）：雙欄網格（2 欄 $\times$ 2 行）。
     - 手機螢幕（$< 768\text{px}$）：單欄縱向卡片堆疊，各區塊支援折疊與快速錨點切換。
  3. 歷史趨勢迷你圖（Sparkline）採用原生 Inline SVG，無第三方圖表依賴，並嚴格遵循 `prefers-reduced-motion`。
  4. 卡片底部標示明確免責聲明「前沿訊號不等同商業訂單，需財報/籌碼驗證」。

## Risks / Trade-offs

- **[Risk 1: arXiv API 存取頻率限制 429 或網路逾時]**  
  $\rightarrow$ *Mitigation*：爬蟲內建 3 秒請求間隔，遇 429 採用指數退避重試（最多 3 次）。若完全不可用，則載入前次有效快照，API 與前端回應標記 `stale: true` 及實際資料日期。
- **[Risk 2: 論文累積造成 DuckDB 磁碟過度膨脹]**  
  $\rightarrow$ *Mitigation*：制定資料保留 TTL 策略：`discovery_papers` 保留 180 天、`discovery_signals` 保留 90 天、`discovery_history` 永久保留；每日管線執行末尾執行過期資料清理。
- **[Risk 3: 前端初次載入過大 Payload 導致卡頓]**  
  $\rightarrow$ *Mitigation*：API 端點支援分頁（預設回傳 Top 20 候選與 Top 10 論文），研究卡（Dossier）採客戶端 Lazy-render 展開，確保初始傳輸 Payload < 50KB。
