#!/bin/bash
# 在另一台機器上執行此腳本來更新並重新安裝

echo "=== 更新 requirements.txt ==="
# 1. 先拉取最新的 requirements.txt
git pull origin main

echo ""
echo "=== 清理舊的安裝 ==="
# 2. 清理 pip cache
pip cache purge

echo ""
echo "=== 重新安裝依賴 ==="
# 3. 重新安裝（跳過已安裝的）
pip install -r requirements.txt --upgrade

echo ""
echo "=== 安裝完成 ==="
echo "如果看到任何錯誤，請檢查是否為可選套件（pymongo, mysql, statsforecast）"
