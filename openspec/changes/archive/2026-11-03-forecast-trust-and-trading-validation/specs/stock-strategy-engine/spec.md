## ADDED Requirements

### Requirement: Third foundation model shadow forecasts
系統 SHALL 支援 Salesforce MOIRAI-2（Salesforce/moirai-2.0-R-small）為第三模型，初次以研究 shadow 模式執行並保存所有有效預測；輸出 T+1 點預測、獨立步長軌跡、原生 P10/P50/P90、版本與推論狀態。shadow 結果僅供比較，績效與用途資格成立後才參與選股。

#### Scenario: Initial candidate deployment
- **WHEN** 管理者於適用研究環境啟用 MOIRAI-2 shadow
- **THEN** 模型與 TimesFM 3.0、Timestral 209M 同場累積成績，選股結果明列實際合格參與模型

#### Scenario: Candidate failure
- **WHEN** 第三模型載入或單檔推論失敗、輸出非有限數或分位數無效
- **THEN** 保存失敗原因與覆蓋資訊，其他模型繼續完成；缺失分位數不得以固定百分比偽裝為原生分位數

### Requirement: Reproducible model upgrades
系統 SHALL 分別記錄套件版本與 checkpoint revision，模型更新須保留舊版本成績並重啟新版本樣本累積；TimesFM 預設改為 3.0，2.5 僅供歷史查詢，執行路徑及失敗 fallback 均排除 2.5。

#### Scenario: TimesFM replacement and failure
- **WHEN** 新版管線執行或 TimesFM 3.0 載入失敗
- **THEN** 只執行設定的3.0或回報其失敗，維持2.5歷史紀錄原版本；不得靜默載入2.5

### Requirement: Model usage eligibility
系統 SHALL 分開揭露模型的推論、績效與用途資格。TimesFM 3.0 自架權重僅用於非商業／非正式研究，MOIRAI-2 權重依 CC BY-NC 4.0 判定用途；正式供應僅啟用適用的授權與管道，資格不足須輸出 disabled 原因並保留其他合格模型。

#### Scenario: Research model enabled in a production configuration
- **WHEN** 正式服務設定要求自架研究權重而其用途資格不足
- **THEN** 該模型列 disabled 並輸出原因，保留其他合格模型，TimesFM 2.5 不作替代

### Requirement: Observable covariate model inputs
系統 SHALL 將必要市場協變量接入三模型推論，保存feature profile版本、實際使用欄位、資料截止、來源revision及輸入遮罩。必要特徵缺漏須明示features_unavailable；純價格基準使用獨立設定身分，禁止靜默忽略協變量。

#### Scenario: Missing exchange rate in a full feature run
- **WHEN** 台股完整特徵預測缺少可用美元兌台幣資料
- **THEN** 該執行回報必要特徵缺漏與原因；另行純價格預測須明確標示price_only，不能列為完整特徵版
