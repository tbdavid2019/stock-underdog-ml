-- Add missing columns to predictions table on Supabase
-- Run this in Supabase SQL Editor to fix "Could not find column" errors

-- 1. Add ev_ebitda column (Enterprise Value / EBITDA)
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ev_ebitda numeric;

-- 2. Add other dual strategy columns if they are missing
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS strategy_type text;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ma5 numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ma10 numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ma60 numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ma120 numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ma250 numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS pullback_type text;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS pe numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS pb numeric;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS forward_pe numeric;

-- 3. Add comment for documentation
COMMENT ON COLUMN predictions.ev_ebitda IS 'Enterprise Value to EBITDA ratio';

-- 4. Create index for strategy_type if not exists
CREATE INDEX IF NOT EXISTS idx_predictions_strategy_type ON predictions(strategy_type);
