# 使用 Miniconda 映像（預裝科學運算環境）
FROM continuumio/miniconda3:latest

# 設定工作目錄
WORKDIR /app

# 安裝 Python 3.11 和科學計算基礎套件（使用 conda）
RUN conda install python=3.11 numpy pandas scikit-learn -y && \
    conda clean -afy

# 複製需求文件
COPY requirements.txt .

# 安裝其他 Python 依賴（只保留 LSTM 需要的）
RUN pip install --no-cache-dir \
    python-dotenv requests urllib3 yfinance \
    torch torchvision \
    supabase

# 複製應用程式代碼
COPY . .

# 建立必要的目錄
RUN mkdir -p logs cache models

# 設定環境變數
ENV PYTHONUNBUFFERED=1

# 預設命令
CMD ["python", "main.py"]
