# Design

## Context

動機見 proposal.md。已查核的現況：

- `core/config.py` 預設為 TimesFM 2.5、Timestral 209M，均採 128 根輸入與 5 步輸出；`requirements.txt` 的 timesfm 套件號並非 checkpoint 版本，Timestral 依賴目前指向可變 Git HEAD。
- `strategies/base.py` 已有 `StrategyResult.metadata`，可擴充版本與執行身分；registry 可擴充策略，但 `pipeline/orchestrator.py`、綜合報告、持久化與前端仍有明確雙模型分支。
- `data/duckdb_manager.py` 的 `predictions` 是展示用寬表，`get_ticker_history` 會按日期／模型去重，現有 schema 缺完整 forecast identity；正式評估須保存原始執行。
- `api/static/js/app.js` 以相鄰有紀錄日期推算隔日實價，且 2 筆／50% 即可獲單股領先資格；`backtest/backtest.py` 是 Supabase 開獎腳本，並非成交與持倉引擎。
- `backtest/resolver.py` 依產生時間判斷台／美目標日，其他標的套用美國時間；`core/market_calendar.py` 已支援台美日，需補實際 session 收盤、提早收盤及資料完整性契約。
- 現有 composite 主規格的合成敘述含技術阻力，實作目前採雙模型等權；本案新增版本化合成模式，legacy 模式保留既有實際行為並揭露此差異，不藉本案悄悄改寫既有規格。

## Goals / Non-Goals

**Goals:** 共用單一開獎與績效資料來源，模型升級可比較／回滾，分市場逐階驗證新增模型與自動降權；交付順序有可獨立驗收的切片。

**Non-Goals:** 真實券商串接、自動實盤下單、融資放空、跨幣別合併報酬、完整總經因果推論。本次亦不將 TimesFM 3.0 自架權重部署為正式服務，不預設 HF 通用榜單等同股票交易能力。

## Decisions

### 1. 模型目標改為 TimesFM 3.0、Timestral 209M、MOIRAI-2

使用者指定TimesFM 3.0直接取代2.5，第三模型採Salesforce MOIRAI-2（`Salesforce/moirai-2.0-R-small`），本案移除Chronos-2整合。新執行清單只含三個目標模型；TimesFM 2.5僅保留歷史版本與成績，不並行推論、不作3.0失敗fallback。這是使用者指定的版本遷移，並非宣稱新版或較新模型必然更準；績效資格仍由同場樣本與成本驗證決定。

TimesFM 3.0 adapter改用官方`timesfm3.TimesFM3Evaluator`／`ModelConfig`，`predict_batch`回傳每序列forecast及9個0.1至0.9分位數；P10/P50/P90索引為0/4/8，取代目前2.5的1/5/9。更新`models/timesfm_model.py`、`core/config.py`及測試，完全移除2.5載入類別與compile假設。未通過新版載入與輸出契約測試前不聲稱已完成升級。

新增`models/moirai_model.py`、`strategies/moirai.py`與`MOIRAI_*`設定，採Uni2TS的`Moirai2Module`、`Moirai2Forecast`與官方predictor流程；權重config包含9個0.1至0.9分位數，透過quantile key取值，禁止複製舊模型索引或以固定百分比合成區間。先對三模型採相同完整收盤序列、128根context、T+1與T+5，同場保存所有有效預測。共變量實驗另用設定版本，先核對各adapter實際支援的形狀及缺值規則。推論與行情抓取分開、限制batch及CPU threads，序列載入並記錄峰值記憶體，超資源預算時顯示缺覆蓋原因。

執行角色與用途資格分開：Timestral延續目前角色；MOIRAI-2初期為research shadow候選。TimesFM3使用新的forecast版本重新累積績效，研究環境可直接替代原TimesFM位置，不先繼承2.5勝率。用`MODEL_ROLES`、`DEPLOYMENT_MODE`與可稽核的usage eligibility控制：TimesFM3下載／自架權重限非商業、非正式研究；MOIRAI-2權重為CC BY-NC 4.0，Apache 2.0的Uni2TS程式授權不等於權重可商用。正式服務以已取得適用權利／授權管道的adapter供應，資格不足標disabled並以其他合格模型降級，禁止啟動2.5代替。Google Cloud路徑另以明示部署設定整合，不默默產生雲端費用或視為已完成授權。

所有adapter鎖定套件、Git commit及checkpoint revision，使用分鍵模型快取與原生分位數來源標記。Timestral既有固定±5% fallback標為非原生或失敗，排除區間績效與依賴區間的選股。MOIRAI-2並非最新發布保證或金融專用模型，本案採用依據為使用者指定，實際表現須測試。

來源（2026-11-03查核）：

- https://huggingface.co/Salesforce/moirai-2.0-R-small
- https://huggingface.co/Salesforce/moirai-2.0-R-small/raw/main/config.json
- https://github.com/SalesforceAIResearch/uni2ts
- https://huggingface.co/mistralai/Timestral-209M
- https://github.com/google-research/timesfm
- https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE

### 2. 新增標準長表，保留展示寬表

預計新增：

| 資料 | 關鍵內容 |
|---|---|
| model_runs | run ID、模型／套件／權重／程式／設定版本、role、資料快照、成功與失敗狀態 |
| forecast_records | forecast ID、run ID、canonical ticker、市場、輸入最後完整 bar、data_cutoff、issued_at、目標 session、步長、點預測及原生分位數、價格基準、evidence_type |
| forecast_actuals | forecast ID、驗證版本、實價／來源／行情版本、actual_available_at、verified_at、pending／invalid／settled 與原因 |
| performance_snapshots | as_of、篩選、規則版本、來源驗證版本、樣本／排除統計與榜單 |
| model_health_events | 市場／版本／步長、狀態前後、有效時間、證據快照、政策版本與倍率 |
| simulation_runs | 回測／帳戶設定、版本化成本、股票池、行情與訊號快照、限制說明 |
| paper_orders / paper_fills / account_events / equity_snapshots | 冪等訂單與成交、帳戶事件、每日對帳及估值 |

唯一鍵為批次 run ID + 標的 + 模型 revision + 設定 + 步長，pipeline 重試沿用 run ID；同日多指數共用同次推論結果，避免重複計分。不同時間新推論保持獨立。每日榜單每標的／模型／步長採事先定義 session cutoff 前最後一次合格輸出，禁止事後挑最好預測。明確保留市場、日期、幣別、價格調整方式，禁止依字串日期與下一筆結果猜測。

DuckDB 為新績效資料的單一讀取來源，背景 writer 使用現有寫入鎖及交易，API 唯讀；Supabase 同步可選且在提交本機後經 outbox 重試。遠端寫入經 `json_safety.py` 嚴格序列化驗證；原有 predictions 寬表續供相容讀取。相較直接加大量模型專屬欄位，長表可支援第四模型、版本比較與多步長。

### 3. 開獎以最後完整輸入 bar 與目標 session 為準

從完整日 K 的 session date 前進 horizon 個交易 session，盤前使用前一完整 session；盤中未完成 bar 排除。`issued_at` 必須早於目標 session 收盤；若資料截止晚於目標定價或預測發布過晚，列 invalid。session calendar 涵蓋 TW／US／JP、時區、DST、假日、US 提早收盤；日股採對應日期有效收盤時間。僅驗證該目標日，停牌／缺價保持 pending 並揭露 ageing。排程可重試，經確認的價格供應商資料才入帳。

預測評估採同一調整基準的 current／predicted／actual；若除權拆股造成歷史調整基準變動，建立新驗證版本並重算受影響快照。成交帳本採未還原價加公司事件，防止配息雙重計入。歷史 replay 的新模型可能已見過過去資料，公開標記預訓練污染風險，不列入 prospective 證據。舊資料缺血緣標 legacy_unverified，允許展示但不進正式榜；不可捏造當時版本。

### 4. 股票榜與模型榜共用確定性績效引擎

母體採 `data/universe_providers.py` 與 `market_universe` 的 TW／US／JP 官方普通股清冊，新增按 snapshot_date 保存的 universe_memberships 與 eligibility snapshot，保留新增、下市及停牌成員。官方來源成功完整更新才產生新母體，失敗保留上一版標 stale。當日預測池是母體的明示子集合；保存每個母體成員的 not_scheduled／insufficient_history／suspended／inference_failed 等原因，分別報告母體覆蓋、合格池覆蓋與共同樣本覆蓋，不把只成功預測的股票當分母。歷史清冊不足的期間標 current_universe_replay；HK／CN／EU／UK 等既有證券查詢來源暫不宣稱可排名，UI明示評估市場僅TW／US／JP。

新增 `evaluators/forecast_performance.py`，前端取消重算評分。預設 T+1，20／60／120 交易日可切換；預設 60、最低 30 有效樣本，因此 20 日視窗預設顯示觀察榜，管理者可版本化調整門檻。每市場 calendar 決定視窗，as_of 為帶時區時間。只納入 `actual_available_at <= as_of` 的資料，回溯報告保存所用行情版本。

方向預設 sign(return)，持平門檻為 0，並顯示三類規則。報告命中率、Wilson 95% 下界、MAPE、MAE、native P10–P90 覆蓋與 pinball loss（可用時）、樣本與每日覆蓋。持平價格基準 improvement = 1 - Σabs(model-actual)/Σabs(current-actual)；分母零時 unavailable。方向基準採校準區間多數方向，禁止從測試區間選方向。股票榜按 Wilson 下界、基準改善、樣本、代號排序；未改善基準者保留結果並標示。

模型榜依共有 ticker/session 交集計主排名，各股票先等權彙整，避免多股票日造成權重偏差；列出每模型全集與交集、股票數與 session 數。Wilson 在模型榜僅為描述統計，不能把相關股票當獨立樣本宣稱顯著；晉升比較使用以 session 分塊的 bootstrap 差異區間。市場、步長、evidence_type、版本分組，不把 JP／TW／US 合併成一個冠軍。個股最佳模型在每個決策 as_of 重新只看既有開獎。

### 5. 先做日頻多頭模擬，共用回測與紙上交易引擎

新增 `backtest/simulation.py` 與 `backtest/paper.py`。訊號在可用後下一 session 開盤進場，預設持有 5 sessions 後收盤出場，開盤至同日收盤為 1 session；反向訊號於下一開盤出場，不加槓桿、不放空。T+1 是預測評分，5 sessions 是獨立交易政策，報告分開。止損與止盈先作收盤觸發、下一開盤執行，遇跳空按可成交價；避免日 K 猜測盤中先後。費用／稅／滑價按市場與商品建立生效日版本，實作時核對官方費率，報告同時保留零成本對照；缺費率設定拒絕啟動正式成本回測。

walk-forward 預設校準 120 sessions、測試 20，purge 至少 horizon sessions，股票池與基本面須有當時快照，缺少時限制為價格模型重播並標明現今股票池偏誤。每期只用校準資料決定排序與權重，測試 OOS 獨立彙整。Sharpe 採日淨值、依市場 sessions 年化、設定無風險率；少於 20 報酬樣本或零變異顯示 unavailable。預測準確度與交易獲利明確分列。

paper 與 backtest 共用 fill、cost、risk、accounting 邏輯。訂單／成交／費用／持倉於單一 DB transaction 更新，外部行情 IO 在交易外；故障注入驗證中斷回復。使用本機 CLI 建立／執行／重設帳戶，公開 API 僅查詢允許公開的模擬帳戶。各幣別獨立帳戶，避免隱藏匯率收益。

開盤容量另外採決策前20個sessions的開盤競價量中位數1%，並與1% ADV20取較小值；不足10個歷史開盤量樣本時拒單，日K無法提供開盤量時列 capacity_unavailable，不產生可執行成交主張。成交上限允許部分成交，餘量當session取消，訂單狀態增加 partially_filled，費用以實際成交量計。滑價至少為設定bps，另加按訂單量／保守容量的非負衝擊項，保存公式與版本；當日全天量及事後競價量只能作對照，不作當時容量。成本回測先用具備該資料的可驗證範圍；其餘明確回報無成交與資料缺口。

### 6. 初始風控與模型退化政策版本化

部分成交且剩餘取消的最終order status固定為 partially_filled，另列 filled_quantity、cancelled_quantity、remaining_quantity=0 與 remainder_reason；只有零成交且全部取消才為 cancelled。所有重試按fill ID冪等，成交、費用與取消事件同筆交易提交。帳戶不得以最終status取代成交事件計算現金。

日頻模擬政策初始值：單股淨值 10%、產業 30%、相關群組 40%、單筆交易參與量 1% ADV20；20% 帳戶回撤暫停加倉。相關群組採過去 60 sessions、至少 40 成對報酬、相關係數 >= 0.8 的連通群組，樣本不足暫停新增風險。總曝險取宏觀建議與帳戶上限較低值；未知分類採資料不足拒單。初始值是可調實驗政策，保存每次使用版本，不宣稱為最佳配置。減倉允許，模擬強制減倉先列建議、由明示再平衡政策在下一 session 執行。

模型政策先 `observe_only`：有效樣本 >=30 且涵蓋 >=20 sessions 後才判品質；最近20 sessions 命中率較前60 sessions 下降 >=10 個百分點，或基準改善率 <=0，記 watch。連續3個新 settled sessions 觸發則 degraded，倍率0.5；連續5個觸發或session block bootstrap命中率上界<0.5則 suspended，倍率0。healthy／watch 倍率1；insufficient_data 的新模型僅 shadow。資料服務 unavailable 另以 availability guard 暫停參與，不寫成品質退化。

健康引擎每市場／模型版本／步長每日先股票等權聚合，再以session等權形成視窗；session去重並使用固定評估cutoff。單session至少達預先設定合格池覆蓋80%才推進連續計數，開獎不足僅標資料品質。最近20與其前60為互不重疊區間，各需完整合格session才使用下降判準；不足時只用已滿足樣本條件的基準改善判準。品質停用採session block bootstrap命中率上界<0.5；固定seed、block長度5 sessions、2000次重抽樣並記政策版本。品質視窗由固定calendar sessions定義，覆蓋不足不往前補日；未完整的視窗停止相應判準。回填／修正產生獨立研究重算時間線；既有已執行權重事件與模擬成交保留當時證據，不回寫過去決策。冪等事件鍵確保同session不多算。

恢復採停用後至少30個新驗證標的日且涵蓋20 sessions、基準改善>0、命中率>=55%，連續5個新 sessions 成立，先 degraded，再經下一批20 sessions 確認 healthy。shadow 推論在 suspended 期間續跑。自動品質覆蓋待回測與紙上驗證後以設定啟用，事件記錄可還原。

MOIRAI-2 晉升預設至少60個 prospective sessions、共同覆蓋>=80%，成本回測淨收益不劣於現有合成且最大回撤不增加，以 session block bootstrap 的命中率差異95%區間下界>=0，並改善價格基準；同時要求用途資格成立，由管理設定明確晉升，未達門檻持續觀察。晉升後模型子組預設各1/3，退化倍率後正規化；技術與籌碼權重不重新分配給模型。新雙模型模式指TimesFM3與Timestral，初始各0.5並套健康／用途資格。2.5舊合成僅歷史查詢，不進新執行。三模型共識另標；不足雙模型資格時顯示實際參與者。

### 7. 分期查詢與 UI

Phase A：開獎、完整跨市場協變量管線、TimesFM3替換與MOIRAI-2研究shadow、performance／health 唯讀 API、股票與模型榜、單股多模型 PK。健康先顯示觀測狀態。

Phase B：成本回測、paper、組合風控、模擬帳戶與淨值 API／UI。

Phase C：模型晉升、健康倍率與自動降權、恢復及回滾演練。TimesFM3版本替換在Phase A完成；Phase C評估的是選股資格，不是延後保留2.5。

新增端點見 delta specs。短期快取 key 包含 as_of、全部篩選、評分與資料版本；修正開獎時失效，讀端不觸發模型。行情或計算失敗保留上一快照且標 stale。API 禁止傳入任意檔案路徑／SQL。前端沿用既有導覽與主題；實作介面時先載入 Impeccable 指引做手機與桌面驗證。既有 history 欄位保留，新增 verification 欄位供圖表與 PK 使用。

### 8. 本次必要交付的分市場協變量管線

以下資料採必做範圍，包含來源接入、歷史回填、每日更新、儲存與模型實際輸入；僅交付實驗設定不足以完成。波動率及微調可以保留研究角色，協變量管線與三市場完整特徵組須於Phase A交付。

| 市場 | 必須接入的資料 |
|---|---|
| 台股 | 個股OHLCV、台灣加權、對應台灣產業指數、外資／投信每日淨買賣超、美元兌台幣、當時可得的費半／VIX |
| 美股 | 個股OHLCV、SPY、對應美國產業指數、VIX、費半、美債2年／10年殖利率及10年減2年利差 |
| 日股 | 個股OHLCV、日經225、TOPIX、對應日本產業指數、美元兌日圓、當時可得的費半／VIX |

匯率統一為每1美元兌多少台幣／日圓；殖利率為年化百分比，利差為百分點。指數保存price／total-return定義及產業碼；各市場建立ticker→產業→指數映射及生效日期。現有7類自建板塊不等同官方產業指數；ETF或自建代理須有獨立series ID及proxy標記，原指數不可得時列未完成缺口，不能靜默視為已交付。

新增`data/covariates.py`與版本化series registry，重用`data/fetcher.py`日K、`data/macro.py`下載與`tw_institutional_daily`。補USD/TWD、美債期限利率、TOPIX及各市場產業指數adapter；利率以官方或具來源血緣的利率序列為準，不用債券ETF價格替代。首個實作切片交付逐series來源清冊與真實樣本驗證，記來源代碼、權限、歷史範圍與發布時間規則；這些新增來源目前尚待實作，禁止將規劃描述為現有能力。

新增`covariate_observations`、`covariate_sync_runs`、`feature_profiles`、`feature_snapshots`。長表保存series ID、原始／正規化值、單位、觀測日、published_at、available_at、ingested_at、source、revision及quality／stale；輸入快照保存forecast ID、profile hash、特徵矩陣／mask與來源revision。風控最新摘要不足以替代歷史原始序列。

回填預設5年、可指定起迄；上市不足5年保存全可得期間，其他缺口列報告與限制，禁止補造。每日依來源發布與市場session增量同步，可斷點續傳、冪等重跑；驗證schema、日期唯一性、有限值、單位與異常跳動後commit。來源失敗保留有效版本並揭露freshness，逐series分開更新。

feature builder依決策時間as-of join；指數／匯率用報酬與滯後變動、殖利率用水準及變動、量以校準期規則正規化、法人採已公布淨買賣與累積值。profile保存定義與lookback，初版為128個完整目標市場sessions。跨市場休市可沿用最近可得觀測，最多5個來源市場交易sessions；超限或必要特徵缺失輸出`features_unavailable`與清單，另跑純價格基準須明示price_only，禁止冒充完整協變量版。歷史公布時間不詳時採保守可得時間規則並標estimated及研究限制。

三模型adapter均須接收分市場past-only協變量，能力檢查、shape、mask及實際使用欄位寫入run；未來市場觀測值禁止輸入。先交付可運作完整組合，再在相同ticker/session比較price_only、market_features、full_features三版；效果不佳可由政策降權，資料管線仍為必要交付。profile／資料版本納入榜單、快取與健康scope。

### 9. 方向、報酬與波動率研究分開驗證

負樣本外R²表示平方誤差劣於指定基準，不等於方向命中率低於50%；標準R²以測試均值為基準，對零報酬／歷史均值基準的OOS skill須分別命名與列公式，禁止混用。方向榜與價格誤差榜保留，另外記錄return target type及returns MSE；資料表身分與績效key加入target_type、label_definition_version，禁止混合價格與波動任務。

波動研究先用日K可得的未來h-session實現方差代理`mean(r²)`（close-to-close，不減局部均值；r為log return，h=1時為單日平方報酬），標為daily proxy；不宣稱有5分鐘高頻realized volatility。HAR在log variance上用滯後1/5/22日特徵、expanding校準，epsilon與log轉回的偏差校正僅由校準期決定。比較持續性、RV20及HAR，報告QLIKE（方差尺度）、log variance R²、RMSE與block bootstrap差異；同樣價格分位區間寬度不是已校準波動率，若推論未提供對應target則標unavailable。多步長標籤到期後才入校準與排名。

價量、指數、殖利率、基本面或新聞情緒的協變量須保存published_at／available_at，只允許決策當時可得值，校準與標準化均在訓練視窗；總經及財報修訂值保留vintage，微調採獨立版本及OOS。這些是可驗證實驗，不預先承諾勝過HAR或產生alpha；波動研究結果暫不直接驅動部位，需另通過風控驗收。

研究依據與限制：原作者crypto實驗中的0.64～0.67是BTC／ETH／SOL明日log volatility對過去512日均值的R²；同場HAR為0.77～0.89。該數字不是普遍股市成績或TimesFM3／MOIRAI-2的驗收保證。

- https://xmarva.github.io/blog/2026/ai-crypto-trading/
- https://arxiv.org/abs/2606.27100

## Risks / Trade-offs

- [樣本冷啟動需累積交易日] → 即刻啟動 shadow，歷史 replay 供研究，正式榜顯示累積進度。
- [通用模型無法穩定超越股票持平基準] → 基準比較與成本回測列為驗收，允許維持 shadow。
- [三模型資源超過現有部署機] → 限制批量、序列載入與獨立影子排程，記錄耗時／峰值記憶體，超預算保留部分覆蓋並顯示原因。
- [資料回填的版本、股票池與時間血緣不足] → 分類 legacy／replay，限制 point-in-time 主張，正式榜採 prospective。
- [同產業／跨日樣本相依與多重選模] → 相同樣本、按 session 分塊、OOS 與晉升門檻，揭露估計不確定性。
- [新版規格與舊合成敘述存在差異] → 新模式記錄 policy version，legacy 標籤保持來源，實作測試固定既有可觀察行為。
- [自動政策震盪與資料延遲] → observe_only、連續新 session 計數、恢復遲滯與可回滾事件；品質和可用性分離。

## Migration Plan

1. 實作先用暫存資料庫寫重現測試，包含漏跑、日股、DST、早收、拆股、固定分位數 fallback 與重複批次；讀取遠端 schema、排程與部署機資源盤點，備份後才遷移。
2. 可加不破壞式新 tables/indexes、鎖版 adapter 與 feature flags；保留舊 schema／歷史read path，以 dry-run 顯示 backfill 分類／排除統計。schema rollback 採切回舊讀路徑並保留新增資料，避免破壞歷史證據。
3. Phase A 啟動每日收盤確認後開獎→績效→健康觀測，再跑新 shadow；依各市場 session 排程，pending 可重試。前端先切共同後端 PK，再啟用榜單。
4. Phase B 離線 OOS 與每日 paper 驗證現金對帳、成本與風控，寫入功能僅本機管理命令。Phase C 先記錄觀測事件，再驗證啟用倍率與恢復。
5. 每切片跑相應測試並更新 `docs/CHANGELOG.md`／`README.md`，小提交保留回滾點。回滾關閉新模型／新讀路徑／健康覆蓋旗標，2.5維持退役並以其他合格模型或觀望降級，保留 forecast 與 account events；已停用模型不得被回滾默默恢復為 active。
