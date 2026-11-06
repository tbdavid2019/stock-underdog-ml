# Turnover-Aware Allocation Specification

## Purpose

提供純 Python / NumPy 確定性 ADMM 多頭部位權重優化模組，在無槓桿總曝險上限、單一標的上限（預設對齊 `RiskPolicy`）與產業群組上限約束下，將買賣非對稱 L1 換手率懲罰納入凸優化目標函數，抑制頻繁換倉造成之滑價與台股賣方證交稅耗損。

## ADDED Requirements

### Requirement: Asymmetric Turnover-Penalized ADMM Weight Optimization
系統必須（SHALL）提供無重型外部商業求解器依賴之純 NumPy ADMM 多頭權重優化函式，支援對稱或買賣非對稱 L1 換手懲罰（`gamma_buy * ||(w - w_prev)+||_1 + gamma_sell * ||(w_prev - w)+||_1`），搭配最終可行域裁剪嚴格滿足 `w_i >= 0`、`sum(w) <= max_total_weight`、`w_i <= max_single_weight`（預設 `0.10`）及產業群組上限 `sum_{i in g} w_i <= max_sector_weight`。

#### Scenario: Asymmetric sell penalty suppresses high-tax sell turnover
- **WHEN** 傳入既有持倉權重 `w_prev`、微幅變動之預期報酬與反映台股證交稅之正值 `gamma_sell > gamma_buy > 0`
- **THEN** 求解後之目標權重 `w_target` 保持在可行域內，且相較零懲罰顯著降低單邊換手率 `0.5 * ||w_target - w_prev||_1`

#### Scenario: Risk-neutral zero risk aversion and caps remain numerically stable
- **WHEN** 呼叫端傳入 `risk_aversion = 0.0` 或高預期報酬標的觸及單股上限與產業上限
- **THEN** 優化器不會發生除以零例外，且輸出之 `w_target` 經最終裁剪後在誤差 `1e-8` 內嚴格不超過單股、產業與總曝險上限，並回傳明確 `solver_status`

### Requirement: Shrinkage Covariance and Degenerate Input Fallback
系統必須（SHALL）對歷史報酬共變異矩陣套用對角線收縮正則化以確保半正定性；當有效重疊歷史交易日少於最低門檻（預設 30 日）或輸入含非有限數值時，必須（MUST）拒絕硬解並回傳結構化降級狀態且強制 `allow_nan=False`。

#### Scenario: Collinear or short return series handled safely without NaN
- **WHEN** 傳入兩檔完全共線標的或重疊日數不足 30 日之報酬序列
- **THEN** 優化器不拋出矩陣奇異例外、不產生 `NaN`/`Infinity`，並回傳明確 `solver_status` 與有限數值權重結果
