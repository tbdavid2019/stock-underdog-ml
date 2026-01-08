-- SAFE Migration: Only adds new columns, doesn't modify existing data
-- Run this in Supabase SQL Editor

-- Step 1: Add new columns for backtesting (safe - only adds columns)
ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS actual_price numeric,
ADD COLUMN IF NOT EXISTS actual_date timestamp with time zone,
ADD COLUMN IF NOT EXISTS accuracy numeric,
ADD COLUMN IF NOT EXISTS absolute_error numeric,
ADD COLUMN IF NOT EXISTS percentage_error numeric;

-- Step 2: Create indexes for faster queries (safe - only improves performance)
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_timestamp 
ON predictions(ticker, timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_model_timestamp 
ON predictions(model_name, timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_backtest 
ON predictions(ticker, timestamp) 
WHERE actual_price IS NOT NULL;

-- Step 3: Create views for analysis (safe - read-only)
CREATE OR REPLACE VIEW model_performance AS
SELECT 
    model_name,
    index_name,
    COUNT(*) as total_predictions,
    COUNT(actual_price) as verified_predictions,
    AVG(accuracy) as avg_accuracy,
    AVG(ABS(percentage_error)) as avg_abs_error,
    STDDEV(percentage_error) as error_std_dev,
    COUNT(CASE WHEN (predicted_price > current_price AND actual_price > current_price) 
               OR (predicted_price < current_price AND actual_price < current_price) 
          THEN 1 END) * 100.0 / NULLIF(COUNT(actual_price), 0) as direction_accuracy
FROM predictions
WHERE actual_price IS NOT NULL
GROUP BY model_name, index_name;

-- Step 4: Create view for predictions needing verification
CREATE OR REPLACE VIEW predictions_to_verify AS
SELECT 
    id,
    ticker,
    model_name,
    timestamp,
    current_price,
    predicted_price,
    potential,
    period,
    CASE 
        WHEN period = '1mo' THEN timestamp + interval '1 month'
        WHEN period = '3mo' THEN timestamp + interval '3 months'
        WHEN period = '6mo' THEN timestamp + interval '6 months'
        WHEN period = '1y' THEN timestamp + interval '1 year'
        ELSE timestamp + interval '1 month'
    END as target_date
FROM predictions
WHERE actual_price IS NULL
  AND timestamp < NOW() - interval '1 day'
ORDER BY timestamp DESC;

-- Add helpful comments
COMMENT ON COLUMN predictions.actual_price IS 'Actual stock price observed after prediction period';
COMMENT ON COLUMN predictions.accuracy IS 'Prediction accuracy: 1.0 = perfect, 0.0 = no better than baseline';
COMMENT ON VIEW model_performance IS 'Aggregated model performance metrics for backtesting';
COMMENT ON VIEW predictions_to_verify IS 'Predictions that need actual price verification';
