-- 清空舊的回測資料（保留預測記錄，只清空 actual_price）
-- 在 Supabase SQL Editor 執行

UPDATE predictions 
SET 
    actual_price = NULL,
    actual_date = NULL,
    accuracy = NULL,
    absolute_error = NULL,
    percentage_error = NULL
WHERE actual_price IS NOT NULL;

-- 查看清空結果
SELECT 
    COUNT(*) as total_predictions,
    COUNT(actual_price) as verified_predictions,
    COUNT(*) - COUNT(actual_price) as pending_verification
FROM predictions;
