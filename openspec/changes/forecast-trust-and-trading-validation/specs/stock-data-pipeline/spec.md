## ADDED Requirements

### Requirement: Required cross market covariate coverage
系統 SHALL 在本次變更交付美元兌台幣、美元兌日圓、美債2年及10年殖利率與10年減2年利差、TOPIX、日經225、台灣加權、SPY、VIX、費半、台美日產業指數、個股價量及台股外資／投信資料。每項須有可運作來源與具版本的定義、單位、歷史覆蓋及品質狀態，僅列候選來源不構成完成。

#### Scenario: Acceptance of the complete data inventory
- **WHEN** 驗收本次協變量資料交付
- **THEN** 每項均可查詢歷史範圍與最新有效值，美元兌台幣、美債殖利率、TOPIX及各市場產業指數亦列為必要驗收項目

### Requirement: Historical backfill and daily persistence
系統 SHALL 為必要協變量提供可設定期間的歷史回填、每日增量更新與持久化，保存來源、觀測日、公布／可得時間、抓取時間及修訂版本；重跑須冪等，來源失敗保留最後有效資料並標示stale與失敗原因。

#### Scenario: Restarting an interrupted import
- **WHEN** 歷史回填中斷後重跑或同日更新執行兩次
- **THEN** 紀錄保持唯一且可續傳，價格或利率修訂另存版本，失敗資料不覆蓋有效值

### Requirement: Market specific prediction feature profiles
系統 SHALL 依標的市場套用協變量組合。台股包含加權、產業、價量、法人及美元兌台幣；美股包含SPY、產業、價量、VIX與美債殖利率及利差；日股包含日經、TOPIX、產業、價量及美元兌日圓。國際費半／VIX等跨市場資料須依版本化設定及當時可得性使用。

#### Scenario: Predictions for Taiwan, United States and Japan
- **WHEN** 管線分別預測三個市場的股票
- **THEN** 各模型收到對應市場特徵、來源及可用性遮罩，執行紀錄可證明其實際輸入；資料僅用於事後風控不足以通過此驗收

### Requirement: Point in time feature alignment
系統 SHALL 以決策時間對齊協變量，只使用當時已公布及可取得的完整觀測；休市、時差、發布延遲及修訂須保留狀態，前值沿用須受明示新鮮度上限限制，禁止未來值補洞或將缺值當零。

#### Scenario: Taiwan forecast before United States session closes
- **WHEN** 台股預測的決策時點早於某筆美股或美債資料可得時間
- **THEN** 採上一筆當時可得且仍合新鮮度的資料，否則標缺失並輸出降級原因

### Requirement: Explicit financial series semantics
系統 SHALL 保存匯率報價方向與單位、殖利率年期及百分比尺度、指數價格／報酬類型及產業映射版本；ETF或自建板塊代理須明示與原指數的差異，不得默認等同。缺乏所需產業映射須揭露覆蓋缺口。

#### Scenario: Provider changes symbols or units
- **WHEN** 來源切換、匯率報價反轉或殖利率尺度變更
- **THEN** 驗證正規化後數值與原定義一致，設定及資料版本更新且原始值仍可追溯
