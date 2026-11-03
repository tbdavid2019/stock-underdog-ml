## ADDED Requirements

### Requirement: Explicit candidate promotion and health overlays
系統 SHALL 僅將已通過共同樣本及成本回測門檻、經管理設定晉升的模型納入合成，以基礎權重乘健康倍率後正規化；shadow 或停用模型貢獻為零。預測漲幅排序與可信度排名須分開標示。

#### Scenario: Candidate promoted after evaluation
- **WHEN** Chronos-2 滿足晉升政策且管理者啟用 active
- **THEN** 報告列出三模型版本、有效權重與合成來源，可逐筆重現合成價格

### Requirement: Truthful consensus labels
系統 SHALL 維持既有雙模型標籤僅指 TimesFM 與 Timestral 均為 active、可用、健康政策允許且看漲；三模型共識須獨立標示。停用或單模型輸出須顯示實際參與者，模型目標須限定相同步長。

#### Scenario: Original model suspended
- **WHEN** TimesFM 停用而 Timestral 與已晉升 Chronos-2 看漲
- **THEN** 顯示實際兩個參與模型，既有雙模型與四重共振資格不成立
