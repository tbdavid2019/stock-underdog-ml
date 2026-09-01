"""
api/main.py - FastAPI Application Entrypoint for Stock Prediction & Multi-Strategy Quant Engine
"""

import os
import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predictions, macro, stats
from api.schemas import HealthResponse
from data.duckdb_manager import DuckDBManager

app = FastAPI(
    title="Stock Quantitative Multi-Strategy Platform API",
    description="""
    🚀 **AI & 多維量化交易決策系統 REST API**
    
    支援基於 DuckDB 高效能時序資料庫之即時量化訊號查詢：
    * 🗡️ **玄鐵重劍策略**：MA60/120 均線趨勢回調技術買點
    * 🤖 **LSTM 深度學習**：下一交易日價格預測與漲跌幅排行榜
    * ⭐ **多維共振篩選**：技術 ∩ 籌碼 ∩ ML ∩ 估值 之 `🏆三重共振` / `土洋合買`
    * 🌐 **美股宏觀門檻**：S&P 500 / VIX / 費城半導體 即時曝險評估
    * 🦆 **DuckDB 引擎**：支援 44,000+ 歷史時序數據零拷貝秒級查詢
    """,
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 啟用跨來源資源共享 (CORS) 方便 Web 前端、儀表板與 MCP 工具存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載業務路由
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(macro.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Stock Quantitative Multi-Strategy API is running.",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
def health_check():
    db = DuckDBManager()
    count = db.get_row_count("predictions")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.datetime.now().isoformat(),
        version="2.2.0",
        duckdb_records=count
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
