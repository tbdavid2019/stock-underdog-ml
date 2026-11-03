# 規劃審查紀錄

日期：2026-11-03。本紀錄僅描述規劃審查，模型更新與功能實作尚待 apply。

## 本次資料範圍增補

依使用者要求，USD/TWD、美債2年／10年及利差、TOPIX、台美日產業指數與其餘既有市場資料均納入Phase A必要交付。新增stock-data-pipeline delta與10項資料／模型接入任務，包含歷史回填、每日更新、單位與時間對齊、實際模型輸入及三市場驗收。此增補尚未經獨立審查，原審查結論不延伸套用。

## 本次模型修訂狀態

使用者後續指定TimesFM3取代2.5並加入MOIRAI-2，Chronos-2導入已移除。proposal、design、delta specs及tasks已同步，另補目標類型、波動研究與協變量發布時間驗證。下述Luna審查僅對應修訂前方案；其通用母體、容量、統計與帳本修正仍保留，不代表本次模型組合已獲獨立複核。OpenCode依使用者指示維持停止。

## Codex Luna（修訂前方案）

- Reviewer：獨立 subagent，`gpt-6-luna`，唯讀審查提案、設計、任務、10份 delta specs 與相關現有程式。
- 初審指出兩項阻擋：全市場排名母體與覆蓋分母未定義、開盤容量缺乏可成交依據；另建議釐清健康狀態的session聚合方式。
- 已納入修正：TW／US／JP官方普通股母體與歷史成員快照、未預測原因與三種覆蓋分母；使用事前歷史開盤量的保守容量、資料不足拒單、部分成交與滑價政策；固定session視窗與覆蓋守衛、session block bootstrap、狀態冪等與歷史研究重算隔離。
- 複核追加的兩處清晰度建議亦已修正：健康政策移除Wilson停用條件；部分成交後餘量取消的終態固定為partially_filled，明列各數量與重跑冪等。
- 最終複核結論：目前未見阻擋，規劃可進入實作。實作仍須依tasks驗證固定seed、session重跑、回填隔離及部分成交帳本案例。

## OpenCode

- 依使用者指示啟動獨立OpenCode評估，使用plan agent，省略`--model`沿用預設模型。
- 啟動時CLI版本1.19.4；session metadata確認實際模型為`opencode/deepseek-v4.1-flash-free`。期間使用者更新CLI，原審查程序未取得最終findings。
- 使用者確認原程序當機並明確要求停止等待，故本次OpenCode評估狀態為未完成，未列為通過或雙審核完成。
- 本次不切換模型、不重新啟動評估；僅保留已知執行證據。

## 規劃驗證

- `openspec validate forecast-trust-and-trading-validation --strict`：通過。
- proposal、design、specs、tasks四項規劃產物齊備；11份delta規格與64項未完成實作任務。
- 規劃採三階段：A 統一開獎／TimesFM3替換與MOIRAI-2研究整合／股票與模型榜；B 成本回測／模擬帳本／組合風控；C 模型晉升／退化降權／恢復。
