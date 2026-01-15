-- ========================================
-- 擴展 predictions 表格 - 新增雙軌策略欄位
-- ========================================
-- 在原本的 predictions 表格新增以下欄位：
-- 1. 策略類型 (strategy_type)
-- 2. MA 均線指標 (ma5, ma10, ma60, ma120, ma250)
-- 3. 回調類型 (pullback_type)
-- 4. 基本面數據 (pe, pb, forward_pe)
-- ========================================

-- 新增策略類型欄位
ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS strategy_type text;

-- 新增 MA 均線欄位
ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS ma5 numeric,
ADD COLUMN IF NOT EXISTS ma10 numeric,
ADD COLUMN IF NOT EXISTS ma60 numeric,
ADD COLUMN IF NOT EXISTS ma120 numeric,
ADD COLUMN IF NOT EXISTS ma250 numeric;

-- 新增回調類型欄位
ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS pullback_type text;

-- 新增基本面數據欄位
ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS pe numeric,
ADD COLUMN IF NOT EXISTS pb numeric,
ADD COLUMN IF NOT EXISTS forward_pe numeric;

-- 新增索引
CREATE INDEX IF NOT EXISTS idx_predictions_strategy_type 
    ON predictions(strategy_type);

CREATE INDEX IF NOT EXISTS idx_predictions_index_strategy 
    ON predictions(index_name, strategy_type);

-- 新增註解
COMMENT ON COLUMN predictions.strategy_type IS '策略類型: 玄鐵重劍, LSTM預測, 雙重符合, 或 NULL (舊版資料)';
COMMENT ON COLUMN predictions.ma5 IS '5日移動平均線';
COMMENT ON COLUMN predictions.ma10 IS '10日移動平均線';
COMMENT ON COLUMN predictions.ma60 IS '60日移動平均線 (大勢判斷)';
COMMENT ON COLUMN predictions.ma120 IS '120日移動平均線';
COMMENT ON COLUMN predictions.ma250 IS '250日移動平均線 (價值中樞)';
COMMENT ON COLUMN predictions.pullback_type IS '回調類型: MA60回調, MA120回調';
COMMENT ON COLUMN predictions.pe IS '本益比 (P/E Ratio - Trailing)';
COMMENT ON COLUMN predictions.pb IS '股價淨值比 (P/B Ratio)';
COMMENT ON COLUMN predictions.forward_pe IS '預估本益比 (Forward P/E)';

-- ========================================
-- 查詢範例
-- ========================================

-- 1. 查詢雙重符合（同時通過 LSTM 和玄鐵策略）
-- SELECT ticker, current_price, predicted_price, potential, 
--        ma60, pullback_type, pe, pb, timestamp
-- FROM predictions
-- WHERE strategy_type = '雙重符合'
-- ORDER BY timestamp DESC;

-- 2. 查詢玄鐵重劍策略結果
-- SELECT ticker, current_price, ma5, ma60, ma120, 
--        pullback_type, pe, pb, timestamp
-- FROM predictions
-- WHERE strategy_type = '玄鐵重劍'
-- ORDER BY timestamp DESC;

-- 3. 查詢 LSTM 預測結果（含基本面）
-- SELECT ticker, current_price, predicted_price, potential,
--        pe, pb, forward_pe, timestamp
-- FROM predictions
-- WHERE strategy_type = 'LSTM預測'
--   AND potential > 3
-- ORDER BY potential DESC;

-- 4. 混合查詢（新舊資料兼容）
-- SELECT ticker, model_name, strategy_type, current_price, 
--        predicted_price, potential, timestamp
-- FROM predictions
-- WHERE timestamp > NOW() - interval '7 days'
-- ORDER BY timestamp DESC;
