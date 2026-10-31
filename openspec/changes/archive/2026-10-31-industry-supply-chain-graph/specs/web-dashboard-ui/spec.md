# Spec Delta

## ADDED Requirements

### Requirement: Interactive Industry Value Chain Graph Canvas
系統 MUST 於前端儀表板提供「產業圖譜」視覺化畫布，採用三欄階層架構（上游 ➔ 中游 ➔ 下游）呈現產業鏈結構，並繪製動態傳導連線。

#### Scenario: Visualizing three-column stream layout
- **WHEN** 使用者進入產業圖譜視圖
- **THEN** 畫面依上游、中游、下游將各領域公司卡片垂直排列，並顯示當前漲跌幅與 AI 預測目標價

#### Scenario: Theme and edge filtering
- **WHEN** 使用者點選特定主題膠囊（例如「CPO/光互聯」）或點擊任一股票卡片
- **THEN** 畫布動態高亮相關傳導關係線並將不相關節點半透明淡化

#### Scenario: Zoom and pan navigation on graph canvas
- **WHEN** 使用者在產業鏈畫布上使用滾輪縮放或拖曳畫布
- **THEN** 畫布平滑縮放與平移，並支援雙擊快速還原初始視野
