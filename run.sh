#!/bin/bash
# Stock Prediction Application Runner
# Runs backtest first, then prediction
# Total time: ~15-25 minutes

cd /home/ec2-user/stock-underdog-ml
source myenv/bin/activate

echo "=========================================="
echo "Stock Prediction System"
echo "Start Time: $(date)"
echo "=========================================="

# Step 1: Backtest (verify past predictions)
echo ""
echo "[1/2] Running backtest..."
python backtest/backtest.py
BACKTEST_EXIT=$?

if [ $BACKTEST_EXIT -eq 0 ]; then
    echo "✅ Backtest completed"
else
    echo "⚠️ Backtest failed (exit code: $BACKTEST_EXIT)"
fi

# Step 2: Generate new predictions
echo ""
echo "[2/2] Generating predictions..."
python main.py
PREDICT_EXIT=$?

if [ $PREDICT_EXIT -eq 0 ]; then
    echo "✅ Predictions completed"
else
    echo "❌ Predictions failed (exit code: $PREDICT_EXIT)"
fi

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

# Exit with prediction status (more critical)
exit $PREDICT_EXIT
