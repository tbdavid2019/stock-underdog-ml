## ADDED Requirements

### Requirement: Third foundation model shadow forecasts
系統 SHALL 支援 Amazon Chronos-2 為可配置第三模型，初次以 shadow 模式執行並保存所有有效預測；輸出 T+1 點預測、獨立步長軌跡、原生 P10/P50/P90、版本與推論狀態。shadow 結果僅供比較，晉升後才參與選股。

#### Scenario: Initial candidate deployment
- **WHEN** 管理者啟用 Chronos-2 shadow
- **THEN** 模型與 TimesFM 2.5、Timestral 209M 同場累積成績，既有選股不受候選分數影響

#### Scenario: Candidate failure
- **WHEN** 第三模型載入或單檔推論失敗、輸出非有限數或分位數無效
- **THEN** 保存失敗原因與覆蓋資訊，其他模型繼續完成；缺失分位數不得以固定百分比偽裝為原生分位數

### Requirement: Reproducible model upgrades
系統 SHALL 分別記錄套件版本與 checkpoint revision，模型更新須保留舊版本成績並重啟新版本樣本累積；TimesFM 3.0 自架權重只允許隔離研究評估，正式供應維持可適用授權的模型與服務。

#### Scenario: Package update with unchanged checkpoint
- **WHEN** 套件升級但 checkpoint 仍為 TimesFM 2.5
- **THEN** 版本資訊分別顯示且通過相容性驗證後才部署，介面仍顯示 TimesFM 2.5
