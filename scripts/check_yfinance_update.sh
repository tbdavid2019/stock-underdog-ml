#!/bin/bash
# ==============================================================================
# scripts/check_yfinance_update.sh - yfinance 定時巡檢與 CI/CD 自動升級 Shell 入口
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# 確保 Python 環境
if [ -d "${PROJECT_ROOT}/venv" ]; then
    PYTHON="${PROJECT_ROOT}/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 啟動 yfinance 定時自動巡檢..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_yfinance_update.py" "$@"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 巡檢結束"
