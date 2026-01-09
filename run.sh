#!/bin/bash
# Stock Prediction Application Runner
# Runs backtest first, then prediction
# Total time: ~15-25 minutes

cd /home/ec2-user/stock-underdog-ml
source myenv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/run_${TIMESTAMP}.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Stock Prediction System"
log "Start Time: $(date)"
log "Log File: $LOG_FILE"
log "=========================================="

# Step 1: Backtest (verify past predictions)
log ""
log "[1/2] Running backtest..."
python backtest/backtest.py 2>&1 | tee -a "$LOG_FILE"
BACKTEST_EXIT=${PIPESTATUS[0]}

if [ $BACKTEST_EXIT -eq 0 ]; then
    log "[OK] Backtest completed"
else
    log "[WARN] Backtest failed (exit code: $BACKTEST_EXIT)"
fi

# Step 2: Generate new predictions
log ""
log "[2/2] Generating predictions..."
python main.py 2>&1 | tee -a "$LOG_FILE"
PREDICT_EXIT=${PIPESTATUS[0]}

if [ $PREDICT_EXIT -eq 0 ]; then
    log "[OK] Predictions completed"
else
    log "[ERROR] Predictions failed (exit code: $PREDICT_EXIT)"
fi

log ""
log "=========================================="
log "End Time: $(date)"
log "=========================================="

# Exit with prediction status (more critical)
exit $PREDICT_EXIT
