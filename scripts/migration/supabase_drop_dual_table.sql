-- ========================================
-- 刪除 dual_strategy_predictions 表格
-- ========================================
-- 這張表格已被整合到 predictions 表格中
-- 執行此 SQL 以清理
-- ========================================

DROP TABLE IF EXISTS dual_strategy_predictions CASCADE;

-- 確認刪除成功
-- 執行後應該會看到: "Table 'dual_strategy_predictions' does not exist"
