# forecast-verification Specification

## Purpose
提供跨股票與跨模型一致的預測開獎依據，保存事前資料截止、模型版本、目標交易日及行情來源，使排行榜、回測與單股成績具備可追溯且一致的解釋。

## Requirements

### Requirement: Immutable forecast identity and provenance
系統 SHALL 保存每筆預測的唯一識別、標的、市場、模型與權重版本、程式與設定版本、產生時間、輸入資料截止、價格基準、目標類型、標籤定義版本、步長及目標交易日；同一邏輯執行重試須冪等，不同執行須可追溯。

#### Scenario: Retrying and upgrading forecasts
- **WHEN** 相同批次重送，或模型改版後再次預測相同標的
- **THEN** 重送維持同筆紀錄，改版產生獨立版本，原始預測保持可查詢

### Requirement: Market session aligned settlement
系統 SHALL 根據最後完整輸入 K 棒、台美日市場行事曆、當地時區及步長解析目標交易日，僅用該目標日已確認收盤且價格基準一致的行情開獎；缺資料須保留待驗證原因。

#### Scenario: Missing prediction run between sessions
- **WHEN** T 日預測 T+1，而下一次管線紀錄為 T+3
- **THEN** 開獎採 T+1 正式行情；該行情缺漏時保持待驗證，T+3 價格不得代用

#### Scenario: Holiday, partial bar and market close
- **WHEN** 輸入含盤中 K 棒，或目標跨假日、提早收盤與美國夏令時間
- **THEN** 使用完整 K 棒與對應市場 session 計算，正式收盤確認前維持 pending

### Requirement: Versioned verification and historical evidence
系統 SHALL 分離原始預測與開獎版本，行情修正須保留來源與修正歷程；歷史資料以 prospective、replay、legacy_unverified 分類，血緣不足的舊資料須排除於正式可信榜。

#### Scenario: Backfill without known checkpoint
- **WHEN** 舊資料具備價格但缺少可證明的權重版本或輸入截止時間
- **THEN** 顯示 legacy_unverified，揭露排除筆數並保留歷史查詢

#### Scenario: Revised actual price
- **WHEN** 行情供應商更正拆股或收盤資料
- **THEN** 產生新版驗證並使相關績效快照失效，歷史版本仍可稽核
