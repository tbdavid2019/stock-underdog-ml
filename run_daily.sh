#!/bin/bash
# 每日自動執行雙軌策略分析
# 建議執行時間：台股收盤後 14:00-15:00 或美股收盤後早上 6:00-7:00

# 切換到專案目錄
# 自動取得腳本所在目錄，並切換過去
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 載入 conda (如果存在且有需要)
if [ -f "/home/human/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/human/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 啟動環境與日誌設定
mkdir -p logs

# 時間戳記
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/daily_${TIMESTAMP}.log"

# 記錄函數
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 啟動環境與 Python 執行路徑
# 1. 優先尋找本地 venv (SETUP.md 建議)
# 2. 其次尋找指定的 conda 環境
# 3. 最後使用系統預設 python
if [ -f "myenv/bin/python" ]; then
    PYTHON_EXEC="myenv/bin/python"
    ENV_NAME="myenv (venv)"
elif [ -d "/home/human/miniconda3/envs/stockml" ]; then
    PYTHON_EXEC="/home/human/miniconda3/envs/stockml/bin/python"
    ENV_NAME="stockml (conda)"
elif [ -d "$HOME/miniconda3/envs/stockml" ]; then
    PYTHON_EXEC="$HOME/miniconda3/envs/stockml/bin/python"
    ENV_NAME="stockml (conda)"
else
    PYTHON_EXEC="python"
    ENV_NAME="system default"
fi

log "=========================================="
log "🚀 股票雙軌策略系統"
log "開始時間: $(date)"
log "執行環境: $ENV_NAME"
log "Python 路徑: $PYTHON_EXEC"
log "日誌檔案: $LOG_FILE"
log "=========================================="

# 檢查 Python 是否可用
$PYTHON_EXEC --version 2>&1 | tee -a "$LOG_FILE"
if [ $? -ne 0 ]; then
    log "❌ [ERROR] 找不到可用的 Python 執行檔: $PYTHON_EXEC"
    exit 1
fi

# Step 1: 先執行回測（驗證昨日預測）
log ""
log "[1/3] 執行回測驗證..."
$PYTHON_EXEC backtest/backtest.py 2>&1 | tee -a "$LOG_FILE"
BACKTEST_EXIT=${PIPESTATUS[0]}

if [ $BACKTEST_EXIT -eq 0 ]; then
    log "✅ [OK] 回測完成"
else
    log "⚠️ [WARN] 回測失敗 (exit code: $BACKTEST_EXIT)"
fi

# Step 2: 執行雙軌策略分析
log ""
log "[2/3] 執行雙軌策略分析 (LSTM + 玄鐵重劍)..."
$PYTHON_EXEC main.py 2>&1 | tee -a "$LOG_FILE"
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
