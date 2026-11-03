# Tasks

每一切片先寫會失敗的重現／驗收測試，再實作、測試並小提交。Phase A（1–6）包含全部協變量交付；Phase B（7–9）依賴已驗證訊號；Phase C（10）依賴共同樣本與成本驗證。長期樣本累積不以歷史重播替代，候選資格不足時維持 shadow。

## 1. Phase A — 預測身分與儲存

- [x] 1.1 在暫存 DuckDB 寫同批重試、多指數重複、版本更新、非法 JSON 的失敗測試；驗證目前缺少不可變預測與版本身分的問題可重現。
- [x] 1.2 實作 model_runs、forecast_records、forecast_actuals 與績效／健康快照 migration，保留舊 predictions；驗證唯一鍵、原始不可覆寫、重試冪等及 migration 重跑測試。
- [x] 1.3 將 run ID、最後完整 bar、checkpoint revision、設定 hash 與每檔失敗狀態由策略串至持久化；驗證 `test_orchestrator` 與雙模型相容性測試。
- [x] 1.4 實作 legacy backfill dry-run 及可選 Supabase outbox／嚴格 JSON 寫入；用 fixture 驗證分類、排除統計、遠端失敗重試與 secret 不出現在輸出。
- [x] 1.5 更新 README 的資料血緣／舊資料限制及 docs/CHANGELOG.md；交付唯讀盤點、備份與 migration rollback 指令，於暫存資料庫驗證。

## 2. Phase A — 統一開獎

- [x] 2.1 先擴充 `test_backtest_date_resolution.py`／`test_market_calendar.py`，以固定時鐘重現漏跑、盤中 bar、台美日假日、DST、提早收盤與延遲資料情境。
- [x] 2.2 基於最後完整 bar、session calendar 與步長實作目標日期及發布時間檢查；驗證 T+1／T+5、JP 時區與過晚發布標 invalid。
- [x] 2.3 實作正式目標日行情開獎與 pending 原因，修正時新增驗證版本並使快照失效；驗證缺價、停牌、收盤未完成、拆股與重試測試。
- [x] 2.4 提供本機開獎／回填 CLI 與各市場收盤後排程設定；用 fixture 時鐘驗證 pending 可重試、readonly 查詢不寫入。
- [ ] 2.5 更新 README、backtest/README.md 與 docs/CHANGELOG.md，示範漏跑仍以目標日開獎；以相同 fixture 核對文件輸出。

## 3. Phase A — 全部跨市場協變量資料與輸入（必要交付）

- [ ] 3.1 先建立逐series來源清冊與真實樣本驗證，涵蓋美元兌台幣、美元兌日圓、2年／10年美債、TOPIX、日經、加權、SPY、VIX、費半及台美日產業指數；驗收來源代碼、單位、權限、歷史起迄與公布時間，缺少任一項列未完成。
- [x] 3.2 先寫失敗測試：匯率反向、殖利率尺度、產業代碼錯置、重複回填、來源空資料、時差及未公布資料；以fixture證明現有管線缺口可重現。
- [x] 3.3 實作covariate_observations／sync_runs／profiles／feature_snapshots長表與遷移；驗證原始值、revision、時間血緣、唯一鍵、冪等與回滾。
- [x] 3.4 實作USD/TWD及USD/JPY來源adapter、歷史回填與每日更新；以真實少量樣本及fixture核對每1美元報價、休市／缺值與來源切換版本。
- [x] 3.5 實作2年／10年美債殖利率及10Y-2Y利差；以真實少量樣本核對年化百分比／百分點、來源發布延遲、假日與修訂版本，禁止以債券ETF代替利率。
- [x] 3.6 補TOPIX與台美日產業指數adapter及股票產業映射，接既有加權／日經／SPY／VIX／費半；核對各市場代表股票真實來源及price／total-return類型，代理序列與原指數分開驗收。
- [ ] 3.7 將個股OHLCV與台股外資／投信歷史接入統一資料管線，提供預設5年回填與增量CLI／排程；驗證續傳、來源失敗保留有效資料、缺口報告及重跑零重複。
- [x] 3.8 實作TW／US／JP完整feature profiles、128-session as-of join與新鮮度／mask；以台股盤前美股未公布、台日休市、超期匯率及法人發布延遲fixture驗證無前視、無缺值補零。
- [x] 3.9 串接三模型past-only協變量及輸入快照，驗證完整特徵確實進入推論、必要資料缺失明示features_unavailable、price_only不冒充full_features；保存各市場端到端證據。
- [ ] 3.10 更新README資料清冊、來源／歷史範圍、回填命令、更新時間與限制及docs/CHANGELOG.md；驗收每項必要資料可查歷史與最新值，三市場profile各跑一個可重現案例。

## 4. Phase A — TimesFM 3.0 替換與 MOIRAI-2 整合

- [x] 4.1 先寫失敗測試：TimesFM預設3.0、禁止2.5載入與fallback、3.0九分位數0/4/8對應、MOIRAI-2原生quantile key、非有限輸出與單檔失敗；確認舊程式無法通過新版合約。
- [x] 4.2 更新TimesFM wrapper至官方timesfm3載入／predict_batch介面，鎖定套件commit與google/timesfm-3.0-pytorch revision；驗證mock契約與少量真實checkpoint smoke，保留2.5歷史查詢。
- [x] 4.3 新增鎖版Uni2TS與Salesforce/moirai-2.0-R-small wrapper／strategy、MOIRAI_*及MODEL_ROLES；驗證registry、config、分鍵快取、T+1/T+5與原生分位數。
- [x] 4.4 實作DEPLOYMENT_MODE與usage eligibility，區分績效與用途；驗證研究模式可載入、正式配置資格不足顯示disabled、TimesFM3失敗仍不載入2.5。
- [x] 4.5 鎖定Timestral相容套件／權重並標記固定區間fallback來源；驗證非原生區間排除、mock與真實checkpoint smoke，套件號及checkpoint分別記錄。
- [x] 4.6 串接orchestrator／報告／sink泛用模型結果與實際合成來源；驗證三模型研究輸出、MOIRAI shadow零選股貢獻、部分disabled仍完成其他合格模型及通知。
- [x] 4.7 以代表股票池量測三模型耗時、峰值記憶體及CPU/device相容性，交付資源預算與逐模型排程；驗證超預算可降batch／停研究候選並揭露覆蓋原因。
- [x] 4.8 更新README模型表、3.0替換、2.5退役、MOIRAI-2研究設定與兩種權重用途限制，並更新docs/CHANGELOG.md；核對文件命令與實際模型身分一致。

## 5. Phase A — 股票與模型可信度引擎

- [x] 5.1 先寫 fixture 測試：3/3 對 60/100、負R²但方向命中率高於50%、持平類別、零基準誤差、未來開獎、缺樣本、重複指數及交集不一致；驗證排序／排除行為。
- [ ] 5.2 保存官方母體與設定股票池歷史快照、下市成員及每個母體成員缺預測原因；以1800檔／1000預測fixture驗證三種覆蓋分母，歷史清冊不足與不支援市場均明示限制。
- [x] 5.3 實作共用績效引擎、Wilson／MAPE／MAE／return MSE／明示基準的OOS R²／區間分數，提供20／60／120交易日及 as_of；對手算資料逐項核對並驗證方法版本。
- [x] 5.4 實作股票榜與模型共同樣本榜、等股票權重、全集與交集覆蓋；以 fixture 驗證跨市場／版本／target_type／步長／prospective 與 replay 完全分離。
- [ ] 5.5 實作當時最佳模型與 health observe_only 快照，回溯僅使用當時可取得驗證；驗證未来資料注入不改變過去排名。
- [ ] 5.6 建立獨立波動研究標籤、daily close-to-close variance proxy與持續性／RV20／log-HAR基準；先用fixture測試未來標籤洩漏，再核對QLIKE／log variance R²與模型target隔離，缺對應預測時顯示unavailable。
- [ ] 5.7 以已交付的完整資料管線建立price_only／market_features／full_features與微調的獨立比較設定，保存published_at／available_at及資料vintage；以尚未發布財報、後修訂總經與未來情緒fixture驗證拒用，交付三市場實際OOS比較報告。
- [ ] 5.8 更新 README 指標、30樣本預設及20日觀察榜限制，更新 docs/CHANGELOG.md；以示例榜單核對文字與引擎一致。

## 6. Phase A — API 與使用者可見排名

- [x] 6.1 先寫 performance／health／history API 測試，涵蓋非法篩選、空排名、readonly、快取失效與 stale；驗證既有端點相容性。
- [x] 6.2 新增 performance／health endpoints 與 history verification 欄位，快取 key 納入全部篩選及資料版本；通過 API 契約與版本失效測試。
- [x] 6.3 依 Impeccable 指引實作股票可信榜、模型戰績榜與單股多模型 PK，取代前端相鄰紀錄開獎；以固定資料驗證榜單與單股數值一致及模型可切換。
- [x] 6.4 驗證手機／桌面、深淺主題、載入／pending／legacy／失敗／shadow 狀態與股票連結；交付截圖及使用情境檢查結果。
- [ ] 6.5 更新 README API／操作說明及 docs/CHANGELOG.md；以 Phase A fixture 跑端到端：輸入→三模型→開獎→兩種榜單→單股PK，核對相同 forecast ID。

## 7. Phase B — 成本與時間隔離回測

- [x] 7.1 先寫未來贏家洩漏、發布後成交、重疊 horizon purge、漲跌停、容量、跳空與成本侵蝕測試；用手算成交確認預期。
- [x] 7.2 建立按市場／商品／生效日的成本與數量設定，實作共用 fill／cost 邏輯；交付官方費率來源及生效日期，測試缺設定拒跑與最低費用。
- [x] 7.3 接入具版本的歷史開盤量資料與保守容量／衝擊政策，支援部分成交及餘量取消；用未來全天量突增、不足10個開盤樣本、部分成交成本fixture驗證無前視與正確拒單。
- [ ] 7.4 實作版本化訊號／行情／股票池快照與 walk-forward 120/20、purge，缺 PIT 資料標研究限制；驗證改動未來資料不影響既有決策與成交。
- [ ] 7.5 實作 simulation run、淨值與回撤／Sharpe／換手率／基準報告；驗證相同快照重播逐筆一致，少樣本統計 unavailable。
- [ ] 7.6 提供管理 CLI 與唯讀 `/backtests/{run_id}`，交付固定案例報告；API 測試及成本回測整合案例通過。
- [ ] 7.7 更新 backtest/README.md、README 與 docs/CHANGELOG.md，示範毛／淨報酬及預訓練污染、現今股票池限制；核對示例可重現。

## 8. Phase B — 組合風控核心

- [x] 8.1 先寫同產業集中、相關群組、現金不足、宏觀縮減、回撤停買及必要資料過期測試；驗證多訊號仍受限。
- [x] 8.2 實作版本化 pre-trade 風控、縮量／拒單原因及 post-trade 超限建議；測試價格變動超限、未知分類、相關資料不足與降低曝險。
- [ ] 8.3 將風控接入共用模擬引擎並保存決策 as_of／限額；用同一快照驗證 backtest 與 paper 預交易結果一致。
- [ ] 8.4 更新 README 初始實驗政策、風控限制與 docs/CHANGELOG.md；交付可重現集中度拒單範例。

## 9. Phase B — 模擬帳本與績效介面

- [ ] 9.1 先寫成交中斷重試、現金對帳、配息／拆股、缺價格及幣別分離測試；用故障注入驗證交易原子性需求。
- [ ] 9.2 實作訂單／成交／帳戶事件及淨值 tables、管理 CLI，接共用成交／成本／風控；通過冪等、公司事件及逐日對帳測試。
- [ ] 9.3 提供公開 allowlist 帳戶與 orders／equity／risk 唯讀 API；驗證未知／未公開帳戶404，GET 無副作用及公開端無 mutation。
- [ ] 9.4 實作回測／模擬績效、成本、交易明細與風險視圖；以手機／桌面驗證空帳戶、stale、超限與多幣別呈現。
- [ ] 9.5 更新 README 管理命令、模擬標記、部署排程與 docs/CHANGELOG.md；交付包含中斷重試的多日 fixture 帳本報告並核對淨值。

## 10. Phase C — 模型健康、晉升與回滾

- [ ] 10.1 先寫品質與可用性分離、連續新session計數、停用後恢復、版本冷啟動與所有模型零權重測試；重跑同日資料不能增加觸發次數。
- [ ] 10.2 實作先股票再session等權的健康統計、80%覆蓋守衛及session block bootstrap；用100檔相關股、缺開獎、資料回填、重疊步長fixture驗證session只計一次、分組隔離及固定seed結果。
- [ ] 10.3 實作 model_health_events 與版本化 observe_only／enforced 政策、降權及恢復；驗證狀態轉換與歷史 effective_at 可重現。
- [ ] 10.4 實作 prospective 晉升報告、共同覆蓋與 session block bootstrap、管理設定晉升；用合成／固定歷史資料驗證通過與拒絕分支，真實樣本不足時保持 shadow。
- [ ] 10.5 實作 active 模型權重合成與共振資格，顯示實際模型來源；驗證 shadow／suspended 零貢獻、2.5僅歷史／3.0現行標籤、三模型標籤與無模型觀望。
- [ ] 10.6 API／UI 呈現狀態原因、有效權重與恢復進度；以fixture核對同一健康事件貫穿排名、合成、模擬與前端。
- [ ] 10.7 更新 README 啟用／降權／恢復／回滾操作及 docs/CHANGELOG.md；完成 shadow→晉升→退化→恢復與停用新旗標的整合演練，保存報告。

## Workflow follow-up

- 真實 prospective 樣本持續累積；長期晉升條件未滿可交付 shadow 功能及累積進度，保持其真實候選狀態。
- 實作完成後按專案流程驗收與 archive；本規劃階段維持所有實作 checkbox 未完成。
