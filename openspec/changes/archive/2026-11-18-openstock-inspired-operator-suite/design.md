# Design

## Context

動機與範圍請見 `proposal.md`（Why & What Changes）。目前系統採用單體 Python FastAPI（`api/main.py`）搭配嵌入式列式資料庫 DuckDB（`data/duckdb_manager.py`），在生產部署中常駐 `stock-ml-api` 與排程 `stock-ml-cron`（`run_daily.sh`）為兩個共享掛載 `./data/storage` 的獨立 OS 進程；前端則採用免 Node.js 編譯步驟之 Jinja2 Partials（`[[ ]]` 定界符保護 Vue 語法）與原生 ES Modules（`api/static/js/app.js`、`api-client.js`、`chart-manager.js`），視覺體系鎖定曜石黑（`#030302` / `#0D0E11`）與暖米白（`#FFF0CC` / `#FFF9E8`）搭配琥珀金（`#FFB010`）與深藍（`#17324D`）雙主題，並遵循 WCAG AA/AAA 對比度規範。

## Goals / Non-Goals

**Goals:**
- 透過跨進程檔案鎖（`fcntl.flock`）與嚴格唯讀隔離，確保 `run_daily.sh` 與 FastAPI 同時存取 DuckDB 時零死鎖、零靜默吞錯。
- 採用 **Fail-Closed** 寫入防護與雙軌自選架構，保護公開站點（`stockdata.david888.com`）免於未授權竄改、XSS 竊鑰與推播洗版，同時讓匿名訪客透過瀏覽器 `localStorage` 享受零阻力自選篩選。
- 維持前端 0ms 首屏互動與免 Build Step 特性，所有新增 UI 嚴格遵守 Impeccable Operate 模式工藝標準（`tabular-nums`、`>= 44px` 觸控目標、禁止卡片套卡片與紫藍漸層反模式）。
- 100% 複用既有內部權威數據（`MarketCalendar`、`PodcastService`、`DiscoveryPipeline`、`SupplyChainMap`、`_RESOLVER_CACHE`），零新增付費第三方 API 依賴。

**Non-Goals:**
- **不引入** TradingView 外部黑盒 Iframe Widgets（保留原生 Chart.js T+1 平移預測線、P10/P90 區間與三大法人水庫圖）。
- **不引入** Better Auth / MongoDB 多租戶註冊登入牆，亦**不引入** Next.js 或 Inngest 雲端佇列。
- **不提供** 毫秒級逐筆搓合高頻報價串流；到價告警巡檢定位為盤前排程與授權按需批次檢查。

## Decisions

### 0. DuckDB 跨進程寫入仲裁（`fcntl.flock`）與唯讀隔離治理
- **決策**：
  - 仿照 `data/covariate_store.py` 抽取共用跨進程檔案鎖模組 `core/duckdb_lock.py`，鎖定與 `.duckdb` 同掛載卷之 `<db_path>.write.lock`（嚴禁鎖在容器私有 `/tmp`）。所有 DuckDB 寫入與 `_init_schema()` 建表必須先取得排他 `flock`（預設逾時 15 秒，逾時拋出 `DuckDBBusyError` 並由 API 映射為 `HTTP 503 Service Unavailable`）。
  - 修正 `DuckDBManager._get_connection(read_only=True)`：**徹底移除「`read_only=True` 失敗時反向回退開 `read_only=False`」之危險邏輯**；唯讀查詢遇跨進程寫鎖衝突時僅做短暫指數退避 jitter 重試，仍忙碌則回退記憶體快取或明確回傳 `503`。
- **替代方案考量**：僅依賴既有 `threading.RLock()` 無法跨 `stock-ml-cron` 與 `stock-ml-api` 進程協調，必在尖峰排程引發 `IOException` 鎖衝突。

### 1. DuckDB 自選股與量化告警 Schema + Fail-Closed 雙軌權限與 WebMCP 防護
- **決策**：
  - 在 `DuckDBManager._init_schema()`（受 `flock` 保護）新增：
    - `watchlist_items(profile_id VARCHAR DEFAULT 'default', ticker VARCHAR, canonical_ticker VARCHAR, market VARCHAR, note VARCHAR, sort_order INTEGER, created_at TIMESTAMP, PRIMARY KEY (profile_id, canonical_ticker))`（強制驗證 `len(note) <= 500`、單一命名空間上限 100 檔）。
    - `quant_price_alerts(alert_id VARCHAR PRIMARY KEY, profile_id VARCHAR DEFAULT 'default', canonical_ticker VARCHAR, rule_type VARCHAR, threshold_value DOUBLE, reference_model VARCHAR DEFAULT 'auto', enabled BOOLEAN, cooldown_minutes INTEGER DEFAULT 720, last_triggered_at TIMESTAMP, last_triggered_price DOUBLE, created_at TIMESTAMP)`。
  - **Fail-Closed 寫入守衛（`core/netlimiter.py`）**：統一解析真實客戶端 IP（`CF-Connecting-IP` ➔ `X-Real-IP` ➔ `client.host`），修復既有 `api/routes/decision.py` IP 解析不一致與 `GET /api/v1/decision/health` 無限流/URL 暴露問題（加上 60 秒快取、限流，且錯誤僅輸出 host 與例外型別）。當 `DEPLOYMENT_MODE == 'production'` 時，所有寫入與巡檢端點（`POST/DELETE/PATCH /api/v1/watchlist`、`/api/v1/alerts`、`POST /api/v1/alerts/check`）**強制要求**有效 `WATCHLIST_API_KEY`（未設定變數或未帶 Header 一律回傳 `401 Unauthorized`，絕不 Fail-Open）；僅在 `DEPLOYMENT_MODE != 'production'` 且明確設定 `ALLOW_ANONYMOUS_LOCAL_WRITES=true` 時允許本機開發免金鑰直寫。
  - **公開訪客雙軌體驗**：未持有金鑰之公開訪客點擊 `★` 時自動無縫讀寫瀏覽器私有 `localStorage`（`sq_local_watchlist_v1`），互不汙染；持有 `WATCHLIST_API_KEY` 之管理員則同步讀寫 DuckDB 主清單實現跨瀏覽器同步。URL `?watchlist_share=` 嚴格限制正則 `^[A-Za-z0-9.\^]{1,12}$` 與最多 100 檔去重，前端純文字綁定（嚴禁 `v-html`）。
  - **FastMCP / WebMCP 邊界**：由於 `/mcp/sse` 不經 FastAPI `Depends`，公開 MCP 預設僅註冊唯讀查詢工具（`get_watchlist`、`get_quant_alerts`、`get_signal_alignment`）；任何寫入型服務方法皆在業務層強制驗證 `api_key`。

### 2. 免編譯前端 `⌘K` 全域指令面板、最近個股列與資產指紋升級
- **決策**：
  - 新增 `api/templates/partials/command_palette.html`，在 `app.js` 註冊 `⌘K / Ctrl+K / Esc`；輸入 `< 2` 字元（非純數字代號）僅過濾本地已載入快照，`>= 2` 字元防抖 180ms 查詢後端 `/api/v1/predictions/resolve/{query}`（搭配記憶體快取與限流）。
  - 導覽列下方新增「最近瀏覽 / 釘選個股工作列」（`localStorage` 上限 10 檔）。
  - 升級 `api/main.py:get_asset_fingerprint()`：掃描 `api/static/` 與 `api/templates/` 下所有 `.js/.css/.html` 最大 `mtime` 結合 `APP_VERSION`，確保修改任一靜態模組或樣板皆能立即破除 CDN 與瀏覽器快取。

### 3. 跨市場即時交易時段（含日股午休與美股半日市）與交易日感知新鮮度
- **決策**：
  - 在 `core/market_calendar.py` 新增 `get_live_sessions_summary(now_utc)`，支援五態枚舉：`pre_market | regular | lunch_break | after_hours | closed`。精確納入東京證交所（TSE）`11:30–12:30 JST` 午間休市（`lunch_break`）、美股夏/冬令（`America/New_York`）與感恩節次日/平安夜 13:00 提前收盤規則，回傳 `next_transition_epoch`。
  - 資料 `stale` 判定改採 **交易日感知（Trading-Day-Aware Freshness）**：對照該市場最近一個已收盤且過批次結算時間之交易日判定，杜絕農曆春節或長週末誤報全站 `stale`。

### 4. 確定性數值門檻之「多源訊號一致度（Signal Alignment）」合成器
- **決策**：`data/signal_alignment.py`（路由統一掛載於 `api/routes/predictions.py`）以唯讀連線聚合四維度並標準化為 `+1`（看多）、`0`（中性）、`-1`（看空）或 `None`（不適用/缺漏，**嚴禁轉為 0**）：
  1. **時序模型維度**：有效啟用模型平均隔日潛力 `>= +0.5%` 為 `+1`、`<= -0.5%` 為 `-1`，其餘 `0`。
  2. **台股籌碼維度**：僅 `TW` 適用（`US`/`JP` 標記 `applicable=False, score=None`）；近 5 日外資+投信合計淨買超 `> 0` 且投信非連賣為 `+1`，合計淨賣超 `< 0` 為 `-1`。
  3. **Podcast 情報維度**：近 30 天提及該標的之單集多空姿態淨值（逾 30 天標記 `stale=True` 排除分母）。
  4. **AI 前沿雷達維度**：最新有效快照列為 `propose_add`/`already_in_pool` 且 `total_score >= 65` 為 `+1`，`reject` 為 `-1`。
  - 有效來源 `< 2` 回傳 `insufficient_sources`；模型維度與籌碼/情報維度出現 `+1` 對 `-1` 反向時判定為 `divergence_warning`；全非負且至少兩項 `+1` 為 `bullish_alignment`；全非正且至少兩項 `-1` 為 `bearish_alignment`；其餘為 `mixed`。

### 5. 量化價位告警動態/靜態語意與防洗版原子冷卻
- **決策**：
  - **靜態規則**（`above_price`、`below_price`）：必填正數 `threshold_value`。
  - **動態規則**（`below_p10`、`above_p50`、`pullback_ma60`、`pullback_ma120`）：每次巡檢讀取標的最新有效交易日預測動態計算門檻。`reference_model` 支援 `timestral | moirai | timesfm | auto`；因 `predictions` 表中具分位數欄位者為 `timestral_p10` 與 `moirai_p10`，`below_p10` 在 `auto` 模式下取有效模型 P10 之**最小值（最保守防守位）**，若無任何有效 P10 則建立時拒絕或標記 `waiting_for_quantile`；`above_p50` 取對應模型之 `predicted_price`（`auto` 取有效模型等權均值）。
  - **原子冷卻防洗版**：`POST /api/v1/alerts/check` 具備進程內單實例互斥鎖與跨進程 `flock`；巡檢命中時**先在 DuckDB 交易內成功寫入 `last_triggered_at`**，確認 commit 成功後才呼叫 `notifier.py` 發送推播——若 DuckDB 鎖失敗則中止推播，確保絕不因寫庫失敗而無限重發洗版。告警檢查直接串接於 `run_daily.sh` 主管線完成後執行。

### 6. 獨立 `autoescape=True` SSR 公開個股頁（`/s/{ticker}`）與動態 Sitemap
- **決策**：
  - 不共用 `index.html` 的 `autoescape=False` 渲染器！在 `api/main.py` 建立專用 `_render_ssr_template(template_name, context)`，採用獨立 `jinja2.Environment(..., autoescape=True)`（標準 `{{ }}` 定界符），徹底阻絕公司名稱或外部欄位之 Stored XSS。
  - 入口代號強制通過白名單正則 `^[A-Za-z0-9.\-]{1,12}$` 驗證；JSON-LD 區塊透過專用 helper 將 `<`、`>`、`&` 轉義為 `\u003c`、`\u003e`、`\u0026`，防止 `</script>` 閉合注入。
  - `/sitemap.xml` 設定 `Cache-Control: public, max-age=3600`，`<lastmod>` 取實際最新預測交易日而非當下系統時間。

### 7. 共用上游韌性守衛（`core/upstream_guard.py`）與機密遮罩環境健檢（`scripts/check_env.py`）
- **決策**：
  - `core/upstream_guard.py` 提供單進程執行緒安全之 `KeyPoolGuard`（逗號多 Key 輪詢 + `BoundedSemaphore` 單進程併發限制 + 60 秒 `RecentFailureCooldownError` 熔斷；多 Worker 拓撲下各進程獨立計數並在設計文件明示）。
  - `scripts/check_env.py` 驗證資料庫、LLM 備援、Clef、`WATCHLIST_API_KEY`（提醒 `production` 模式下未設定將預設鎖定寫入）與已棄用變數，敏感欄位僅輸出 `[SET]` / `[MISSING]`，整合進 `scripts/install_new_machine.sh`。

## Risks / Trade-offs

- **[跨容器 DuckDB 檔案鎖逾時]** → 排程大批次寫入期間，API 寫入請求若逾 15 秒未取得 `flock` 將優雅回傳 `503` 提示稍後重試，唯讀查詢走短快取不受阻塞。
- **[單進程 Semaphore 在多 Worker 下倍增]** → 目前 Docker 預設單一 Uvicorn 程序運作；文件明確記錄若未來擴展 `workers > 1` 應將 `MAX_IN_FLIGHT_PER_KEY` 按 Worker 數折算。

## Migration Plan

嚴格遵循「後端先行、向下相容、逐切片驗證提交」：
1. **Phase 1（P2 安全與併發底座）**：先上 `core/duckdb_lock.py`、`core/upstream_guard.py`、`core/netlimiter.py`（含修復 `/decision/health`）與 `scripts/check_env.py`。
2. **Phase 2A ➔ 2B（P0 操盤三件套）**：先部署 2A 後端（`watchlist_items`、`/api/v1/watchlist`、`/api/v1/market/calendar/sessions`），確認 API 穩定後再上 2B 前端（`⌘K`、最近個股列、`★ 我的自選` 雙軌儲存、開休市含日股午休時鐘與全目錄資產指紋）。若前端回滾，2A 後端維持無害向後相容。
3. **Phase 3（P1 決策閉環）**：部署 `data/signal_alignment.py`、`quant_price_alerts`、原子冷卻 `alert_evaluator.py` 與抽屜 UI。
4. **Phase 4（P1 公開 SEO 頁）**：部署獨立 `autoescape=True` 之 `/s/{ticker}`、`/sitemap.xml` 與 `/robots.txt`。
