# Tasks

## 1. 5-Tier Resilient Decision Engine Service

- [x] 1.1 建立 `data/decision_service.py` 實作 5 級階梯式韌性決策鏈（Tier 1 Clef 主力、Tier 2 Clef 備援、Tier 3 JEV、Tier 4 OpenAI、Tier 5 本地啟發式規則），並在 `core/config.py` 擴充端點設定。
- [x] 1.2 實作 100% 確定性本地啟發式規則兜底引擎（涵蓋個股量化規則與 Podcast 頻道權重/來賓/實體情緒視窗分析），撰寫測試驗證在「所有遠端端點全部斷線」時仍能 100% 成功返回合法決策物件。

## 2. DuckDB Schema 遷移與 Podcast 情報服務整合

- [x] 2.1 於 `data/duckdb_manager.py` 擴充 `podcast_episodes` 表結構，支援 `importance_score`、`importance_level`、`ticker_sentiments` 與 `decision_slot` 欄位與無損 migration。
- [x] 2.2 於 `data/podcast_service.py` 整合 `DecisionService`，提供單集智慧決策評估、實體多空標記與批量預載增強功能。

## 3. API 路由與健康監控擴充

- [x] 3.1 於 `api/routes/podcast.py` 新增單集決策端點 `POST /api/v1/podcast/episodes/{episode_id}/evaluate` 與批量評估端點 `POST /api/v1/podcast/batch-evaluate`。
- [x] 3.2 於 `api/routes/decision.py` 升級 `/health` 端點以即時回報 5 個 Tier 的存活狀態，並確保 `/evaluate` 支援完整降級標籤。

## 4. 前端介面與 Impeccable 視覺工藝升級

- [x] 4.1 於 `api/templates/partials/tab_podcast.html` 單集卡片實作 AI 重要性膠囊徽章（⭐ 重磅、⚡ 重要、📰 關注）與股票代號多空姿態箭頭（↑ 看多、↓ 看空、↔ 持平）。
- [x] 4.2 於 `api/static/js/app.js` 綁定重要性篩選邏輯，並支援一鍵快篩高催化劑重磅單集。

## 5. 文件同步與遠端部署驗證

- [x] 5.1 更新 `docs/CHANGELOG.md` 與 `README.md`，說明 5 級決策降級架構與 Podcast 智慧評級能力（嚴禁出現任何上游敏感名稱）。
- [x] 5.2 提交變更至 Git，推播至遠端 `10.9.0.99` 伺服器並重啟驗證線上環境正常運作。
