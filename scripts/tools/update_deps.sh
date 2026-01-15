#!/bin/bash
# 在另一台機器上執行此腳本來更新並重新安裝
# 建立 conda 環境（建議 Python 3.10 或 3.11）：
conda create -n stockml python=3.10
conda activate stockml
# 安裝 GPU 版 torch（A4000 支援 CUDA 12.1）：
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# 其他套件用 pip 安裝（建議先裝 numpy、pandas、tensorflow、keras、prophet、autogluon 等）：
pip install numpy pandas tensorflow keras prophet autogluon yfinance scikit-learn pandas-ta python-dotenv pymongo mysql-connector-python
# 執行你的 app.py 測試。
#如遇到「Illegal instruction」或 GPU 無法用，請回報 cat /proc/cpuinfo | grep -i avx 結果。



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
