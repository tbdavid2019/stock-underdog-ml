# 使用 Miniconda 映像（預裝科學運算環境）
FROM continuumio/miniconda3:latest

# 設定工作目錄
WORKDIR /app

# 安裝 Python 3.11 和科學計算基礎套件（使用 conda）
RUN conda install python=3.11 numpy pandas scikit-learn -y && \
    conda clean -afy

# 複製需求文件
COPY requirements.txt .

# 检测 GPU 并安装 TensorFlow（完整版）
RUN pip install --no-cache-dir \
    python-dotenv requests urllib3 yfinance supabase

# 根据 GPU 情况安装 TensorFlow
RUN if command -v nvidia-smi >/dev/null 2>&1; then \
        echo "检测到 GPU，安装 TensorFlow GPU 版本..." && \
        pip install --no-cache-dir tensorflow==2.13.1; \
    else \
        echo "CPU 环境，安装 TensorFlow CPU 版本..." && \
        pip install --no-cache-dir tensorflow-cpu==2.13.1; \
    fi

# 複製應用程式代碼
COPY . .

# 建立必要的目錄
RUN mkdir -p logs cache models

# 设定环境变量
ENV PYTHONUNBUFFERED=1
# TensorFlow 环境变量
ENV TF_CPP_MIN_LOG_LEVEL=2

# 预设命令
CMD ["python", "main.py"]
