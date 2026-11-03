## Purpose

讓使用者以相同且可重現的歷史證據判斷哪些股票容易預測、哪些模型近期較可靠，並顯示樣本限制、簡單基準及資料覆蓋，避免將預測漲幅誤認為已驗證的準確度。

## ADDED Requirements

### Requirement: Explicit universe and coverage denominator
系統 SHALL 以版本化官方證券清冊定義台美日各市場普通股母體，另保存當時設定股票池與預測合格條件；覆蓋率分母須包含母體內未產生預測的股票，揭露未納入、停牌、新上市與失敗原因。歷史成員須保留下市紀錄，缺歷史清冊時須標示限制。

#### Scenario: Pipeline predicts a subset of listed stocks
- **WHEN** 母體有1800檔而管線僅對1000檔產生預測
- **THEN** 顯示母體1800、預測1000、正式排名合格數及其餘800檔狀態，不將1000檔視為全市場覆蓋；其他市場明示尚未支援評估

### Requirement: Stock reliability ranking
系統 SHALL 提供依市場、模型版本、步長、as_of 與 20/60/120 個交易日視窗篩選的股票可信度榜，輸出方向命中率、95% Wilson 下界、MAPE、相對持平價格基準改善率、有效樣本與覆蓋率。

#### Scenario: Small sample appears perfect
- **WHEN** 股票僅有 3 次預測且全部命中，最低樣本設定為 30
- **THEN** 顯示樣本累積中並排除正式排名；合格股票依 Wilson 下界、基準改善率、樣本數與代號穩定排序

### Requirement: Comparable model leaderboard
系統 SHALL 提供模型戰績榜，主排名限定相同市場、步長、價格基準與各比較模型共有的標的交易日樣本；每模型完整樣本結果另列，並揭露股票數、日期數、樣本數與宏觀狀態分組。

#### Scenario: Candidate only predicts easy stocks
- **WHEN** 候選模型僅覆蓋部分股票而既有模型涵蓋全股票池
- **THEN** 主排名僅採交集並揭露覆蓋差異，交集不足顯示證據不足

### Requirement: Reproducible metrics and baseline
系統 SHALL 以 as_of 前已完成且當時可取得的開獎計算績效，方向採上漲／持平／下跌三類且揭露分類門檻；模型與持平價格、歷史多數方向基準須使用相同樣本；replay 與 prospective 須分開呈現。

#### Scenario: Flat and missing values
- **WHEN** 實際報酬為零、預測缺失或價格無效
- **THEN** 持平按統一三類規則計算，缺失與無效值排除並計入覆蓋資訊，所有消費端顯示一致數值

#### Scenario: Best model selection without future leakage
- **WHEN** 使用者查詢某股票當時最佳模型或歷史日期榜單
- **THEN** 選擇僅使用該時點前已開獎樣本，輸出評分版本、排序依據與無合格模型狀態

### Requirement: Distinct financial prediction objectives
系統 SHALL 分開呈現方向命中率、價格／報酬誤差與波動率研究結果，R²須標示目標、分母基準與樣本；負R²僅表示平方誤差劣於指定基準。波動率預測須有獨立標籤與持平、歷史波動及HAR基準，價格分位數寬度不得直接當作已驗證波動率。

#### Scenario: Negative return R squared with positive direction accuracy
- **WHEN** 模型報酬R²為負但方向命中率高於50%
- **THEN** 分列兩項結果、各自樣本與不確定性，不以R²宣稱方向低於隨機

#### Scenario: Covariates and fine tuning experiment
- **WHEN** 比較純歷史價格、協變量或微調版本
- **THEN** 分列設定版本與樣本外結果，協變量採當時發布時間，禁止將未公布財報、修訂後總經或未來情緒值用於歷史預測
