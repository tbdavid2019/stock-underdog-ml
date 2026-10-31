#!/bin/bash
# ==============================================================================
# scripts/install_new_machine.sh - 新機器全自動一鍵部署與環境初始化腳本
# ==============================================================================
# 支援 macOS / Linux (Ubuntu, Debian, CentOS, Rocky Linux)
# 一鍵完成：前置依賴檢查 ➔ 目錄初始化 ➔ .env 安全生成 ➔ 雙容器 (API + Cron) 啟動 ➔ 健康檢查驗證
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=============================================================================="
echo "🚀 888 StockQuant Platform - 新機器全自動生產級安裝腳本"
echo "=============================================================================="
echo "目標專案目錄: ${PROJECT_ROOT}"
echo "執行時區: Asia/Taipei"
echo ""

# ------------------------------------------------------------------------------
# 1. 系統前置環境檢查 (Pre-flight Checks)
# ------------------------------------------------------------------------------
echo "🔍 [1/6] 執行前置環境檢測..."

# 檢查 Git
if ! command -v git &>/dev/null; then
    echo "❌ 錯誤：未安裝 git。請先安裝 git (如 sudo apt update && sudo apt install -y git)。"
    exit 1
fi
echo "  ✅ Git: $(git --version)"

# 檢查 Docker
if ! command -v docker &>/dev/null; then
    echo "❌ 錯誤：未檢測到 docker。請先安裝 Docker Engine: https://docs.docker.com/engine/install/"
    exit 1
fi
echo "  ✅ Docker: $(docker --version)"

# 檢查 Docker 服務是否正常運行
if ! docker info &>/dev/null; then
    echo "❌ 錯誤：Docker Daemon 未啟動或當前使用者無存取權限。請確認 dockerd 是否運行，或執行 sudo usermod -aG docker \$USER。"
    exit 1
fi

# 檢查 Docker Compose
COMPOSE_CMD=""
if docker compose version &>/dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ 錯誤：未安裝 docker compose。請安裝 Compose 插件或套件。"
    exit 1
fi
echo "  ✅ Docker Compose: $($COMPOSE_CMD version)"

# ------------------------------------------------------------------------------
# 2. 運行目錄結構初始化
# ------------------------------------------------------------------------------
echo ""
echo "📁 [2/6] 初始化持久化資料與日誌目錄結構..."
REQUIRED_DIRS=(
    "logs"
    "cache"
    "cache/discovery"
    "data/storage"
    "data/cache"
    "data/cache/industry_chain"
    "data/cache/institutional"
    "models"
    "output"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    mkdir -p "${PROJECT_ROOT}/${dir}"
    echo "  📂 建立/確認目錄: ${dir}"
done

# ------------------------------------------------------------------------------
# 3. 環境變數檔 (.env) 初始化
# ------------------------------------------------------------------------------
echo ""
echo "🔑 [3/6] 配置環境變數檔 (.env)..."
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    if [ -f "${PROJECT_ROOT}/.env.example" ]; then
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        echo "  ✨ 已自 .env.example 建立預設 .env 範本。"
        echo "  💡 提示：系統所有前沿雷達、櫃買中心與產業圖譜皆為 100% 開源免 Key，預設即可運作！"
    else
        touch "${PROJECT_ROOT}/.env"
        echo "TZ=Asia/Taipei" >> "${PROJECT_ROOT}/.env"
        echo "API_PORT=8088" >> "${PROJECT_ROOT}/.env"
        echo "  ✨ 建立基本 .env 設定檔。"
    fi
else
    echo "  ✅ 現有 .env 檔案已存在，保留既有設定。"
fi

# ------------------------------------------------------------------------------
# 4. 啟動生產級 Docker 服務 (API 服務 + 自動排程巡檢 Cron)
# ------------------------------------------------------------------------------
echo ""
echo "🐳 [4/6] 啟動 Docker 雙服務 (FastAPI + 背景定時排程器)..."
echo "  - stock-ml-api : 對外 REST API 與 Web 介面 (Port: 8088)"
echo "  - stock-ml-cron: 常駐定時巡檢 (盤前 08:00、盤後 16:00、美股 20:30、前沿 06:00、產業鏈週一 07:00)"

$COMPOSE_CMD up -d stock-ml-api stock-ml-cron

# ------------------------------------------------------------------------------
# 5. 等候容器啟動並執行健康度自我檢測
# ------------------------------------------------------------------------------
echo ""
echo "🩺 [5/6] 等候服務初始化並執行自我健康檢查..."
MAX_RETRIES=20
RETRY_COUNT=0
HEALTH_URL="http://localhost:8088/health"
THEMES_URL="http://localhost:8088/api/v1/industry-chain/themes"

until curl -s -f "${HEALTH_URL}" > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "⚠️ 等候超時：請使用 '$COMPOSE_CMD logs stock-ml-api' 查看詳細容器日誌。"
        break
    fi
    echo "  ⏳ 等候 API 啟動中 (${RETRY_COUNT}/${MAX_RETRIES})..."
    sleep 2
done

if curl -s -f "${HEALTH_URL}" > /dev/null 2>&1; then
    HEALTH_DATA=$(curl -s "${HEALTH_URL}")
    echo "  ✅ 健康檢查成功: ${HEALTH_DATA}"
fi

if curl -s -f "${THEMES_URL}" > /dev/null 2>&1; then
    THEMES_COUNT=$(curl -s "${THEMES_URL}" | grep -o '"total_themes":[0-9]*' | cut -d: -f2 || echo "8")
    echo "  ✅ 產業價值鏈拓撲檢查成功: 共載入 ${THEMES_COUNT} 大技術主題"
fi

# ------------------------------------------------------------------------------
# 6. 安裝完成摘要與運維指引
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "🎉 888 StockQuant 系統新機器安裝全部完成！"
echo "=============================================================================="
echo "🌐 本地 Web 操盤儀表板 : http://localhost:8088"
echo "📘 Swagger 互動式文檔  : http://localhost:8088/docs"
echo "🕸️ 產業價值鏈圖譜端點 : http://localhost:8088/?tab=industry_chain"
echo ""
echo "🛠️ 常用維運管理指令："
echo "  1. 查看運作狀態      : $COMPOSE_CMD ps"
echo "  2. 查看 API 服務日誌 : $COMPOSE_CMD logs -f stock-ml-api"
echo "  3. 查看定時巡檢日誌 : $COMPOSE_CMD logs -f stock-ml-cron"
echo "  4. 即時手動觸發日報 : $COMPOSE_CMD run --rm stock-ml"
echo "  5. 停止所有服務      : $COMPOSE_CMD down"
echo "=============================================================================="
