## Purpose

讓使用者以相同且可重現的歷史證據判斷哪些股票容易預測、哪些模型近期較可靠，並顯示樣本限制、簡單基準及資料覆蓋，避免將預測漲幅誤認為已驗證的準確度。

## ADDED Requirements

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
