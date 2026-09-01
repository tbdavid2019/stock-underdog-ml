#!/bin/bash
set -e

# ==============================================================================
# Docker Container Entrypoint Script
# Handles runtime initialization, directory prep, and mode dispatch.
# ==============================================================================

# 確保必要運行目錄存在
mkdir -p /app/logs /app/cache /app/data/storage /app/data/cache /app/models

# 模式派發
case "$1" in
  main|daily|"")
    echo "🚀 [Docker] 啟動股票多維量化分析主程序 (main.py)..."
    exec python main.py
    ;;
  sync)
    echo "📥 [Docker] 執行 Supabase ➔ DuckDB 全量數據同步 (export_supabase_to_duckdb.py)..."
    exec python scripts/export_supabase_to_duckdb.py
    ;;
  backtest)
    echo "📈 [Docker] 啟動量化回測系統 (backtest.py)..."
    exec python backtest/backtest.py "${@:2}"
    ;;
  api|server)
    echo "🌐 [Docker] 啟動 FastAPI 高效能量化 REST 服務 (0.0.0.0:8000)..."
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
    ;;
  test)
    echo "🧪 [Docker] 執行全套單元測試 (unittest)..."
    exec python -m unittest discover -s test -p 'test_*.py'
    ;;
  cron|scheduler)
    echo "⏰ [Docker] 啟動容器內建定時排程器 (Cron Daemon - Asia/Taipei)..."
    if [ -f /app/docker/crontab ]; then
      crontab /app/docker/crontab
      echo "📋 已載入排程設定 (/app/docker/crontab):"
      cat /app/docker/crontab
    fi
    # 建立日誌管道並在前台啟動 cron
    touch /app/logs/cron_daily.log
    cron -f
    ;;
  *)
    # 執行傳入的自訂命令 (例如 bash, python -c ...)
    exec "$@"
    ;;
esac
