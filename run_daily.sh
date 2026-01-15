#!/bin/bash
# 每日自動執行雙軌策略分析
# 建議執行時間：台股收盤後 14:00-15:00 或美股收盤後早上 6:00-7:00

# 切換到專案目錄
cd /home/human/stock-underdog-ml

# 載入 conda
source /home/human/miniconda3/etc/profile.d/conda.sh

# 啟動 stockml 環境
conda activate stockml

# 創建 logs 目錄
mkdir -p logs

# 時間戳記
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/daily_${TIMESTAMP}.log"

# 記錄函數
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "🚀 股票雙軌策略系統"
log "開始時間: $(date)"
log "環境: stockml"
log "日誌檔案: $LOG_FILE"
log "=========================================="

# Step 1: 先執行回測（驗證昨日預測）
log ""
log "[1/3] 執行回測驗證..."
python backtest/backtest.py 2>&1 | tee -a "$LOG_FILE"
BACKTEST_EXIT=${PIPESTATUS[0]}

if [ $BACKTEST_EXIT -eq 0 ]; then
    log "✅ [OK] 回測完成"
else
    log "⚠️ [WARN] 回測失敗 (exit code: $BACKTEST_EXIT)"
fi

# Step 2: 執行雙軌策略分析
log ""
log "[2/3] 執行雙軌策略分析 (LSTM + 玄鐵重劍)..."
python main.py 2>&1 | tee -a "$LOG_FILE"
PREDICT_EXIT=${PIPESTATUS[0]}

if [ $PREDICT_EXIT -eq 0 ]; then
    log "✅ [OK] 預測完成"
else
    log "❌ [ERROR] 預測失敗 (exit code: $PREDICT_EXIT)"
    exit 1
fi

# Step 3: 清理舊日誌 (保留最近 30 天)
log ""
log "[3/3] 清理舊日誌..."
find logs/ -name "daily_*.log" -mtime +30 -delete
find logs/ -name "run_*.log" -mtime +30 -delete
log "✅ [OK] 舊日誌已清理"

log ""
log "=========================================="
log "✅ 所有任務完成"
log "結束時間: $(date)"
log "=========================================="

# 顯示結果摘要
tail -100 "$LOG_FILE" | grep -E "(符合條件|預測完成|雙重符合|✅ 雙軌策略分析完成)"
