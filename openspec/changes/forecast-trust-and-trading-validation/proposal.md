# Proposal

## Why

目前平台具備快篩、深度研究與雙模型預測，但跨股票、跨模型的可靠性尚缺乏共同驗證依據。建立可追溯開獎、可信度排名與模擬交易閉環，讓使用者能回答「最近誰最準、照做能否覆蓋成本、何時降低模型權重」。

## What Changes

- 保留 TimesFM 2.5 與 Timestral 209M，新增 Amazon Chronos-2 作為第三個影子候選：保存預測與成績，通過驗收後才參與選股；TimesFM 3.0 自架權重限研究評估，正式導入另立授權與技術方案。
- 以不可覆寫的模型版本、資料截止時間、目標交易日與行情來源建立統一預測／開獎紀錄；單股 PK 改採共同後端結果。
- 建立股票可信度榜與模型戰績榜，分市場、期間、預測步長與版本比較；揭露樣本、覆蓋率、基準與不確定性。
- 分階段交付按時間向前的成本回測、模擬交易帳本與投資組合風控。
- 建立模型退化狀態、降權／停用與恢復門檻；保留原始預測供持續觀測。
- 先交付統一開獎、Chronos-2 影子推論與兩種排名，再交付交易驗證與風控，最後啟用經驗證的自動模型降權。

## Capabilities

### New Capabilities

- `forecast-verification`: 版本化預測、交易日對齊、可追溯與冪等的開獎。
- `forecast-performance`: 股票／模型榜、共同樣本比較、基準與可信度。
- `historical-trading-simulation`: 可重現、時間隔離、包含成本的策略回測。
- `paper-trading`: 模擬訂單、成交、現金、持倉與每日淨值帳本。
- `portfolio-risk`: 單股、產業、相關性與整體曝險檢查。
- `model-health`: 退化偵測、分層狀態、有效權重與恢復紀錄。

### Modified Capabilities

- `stock-strategy-engine`: 新增 Chronos-2 第三模型與版本化影子執行合約。
- `composite-evaluation`: 候選模型晉升與健康權重覆蓋，保留既有雙模型標籤語意。
- `fastapi-rest-service`: 增加績效、模型健康、回測與模擬帳本查詢。
- `web-dashboard-ui`: 新增可信度與模型戰績視圖、模擬績效及健康提示，統一單股 PK 資料。

## Impact

- 實作接點：`models/`、`strategies/`、`core/config.py`、`pipeline/orchestrator.py`、`evaluators/composite_evaluator.py`、`data/duckdb_manager.py`、`backtest/`、`api/routes/`、`api/static/js/` 與 dashboard partials。
- 新增版本化 DuckDB tables 與可選 Supabase 同步 schema；寫入前嚴格 JSON 驗證，遠端變更前唯讀盤點與備份。
- 新增固定版本的 `chronos-forecasting` 與 checkpoint revision；核對既有 `timesfm>=3.0.0` 套件與 2.5 checkpoint 的相容性，套件號與模型號分開記錄。
- 既有歷史紀錄以可證明的資料回填，資料血緣不足者顯示限制；歷史重播與實際事前預測分開。
- 本次產物為規劃文件。實作各切片先寫重現／驗收測試，再更新 `docs/CHANGELOG.md`；使用與資料流程同步更新 `README.md`。
