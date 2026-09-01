"""
api/main.py - FastAPI Application Entrypoint for Stock Prediction & Multi-Strategy Quant Engine
"""

import os
import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from api.routes import predictions, macro, stats
from api.schemas import HealthResponse
from data.duckdb_manager import DuckDBManager

app = FastAPI(
    title="888 Stock Quant Platform API",
    description="""
    🚀 **888 Stock Quant 多維量化交易決策系統 REST API**
    
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

# 安全 CORS 配置：支援環境變數自訂白名單，預設不啟用通配符 credentials
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    allow_origins = ["*"]
    allow_credentials = False  # 安全規範：萬用字元禁止 credentials
else:
    allow_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載業務路由
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(macro.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """
    量化決策平台首頁：提供即時操盤儀表板與 Agent / MCP 對接指南。
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>888 Stock Quant Platform API is running. Visit <a href='/docs'>/docs</a>.</h1>")


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
def get_ai_plugin_manifest():
    """
    WebMCP / Plugin Standard Discovery Manifest
    """
    return JSONResponse(content={
        "schema_version": "v1",
        "name_for_model": "stock_quant_engine",
        "name_for_human": "888 Stock Quant",
        "description_for_model": "888 Stock Quant 專業級深度學習與多維量化決策系統 (宏觀風控、玄鐵均線、LSTM預測、三大法人籌碼、🏆三重共振)。提供每日台美股選股清單、目標價預測、法人籌碼鎖碼、均線波段買點與個股歷史走勢查詢。",
        "description_for_human": "888 Stock Quant Multi-Strategy Stock Trading Engine and Live Screener.",
        "auth": {
            "type": "none"
        },
        "api": {
            "type": "openapi",
            "url": "/openapi.json"
        },
        "logo_url": "https://img.icons8.com/color/96/bullish.png",
        "contact_email": "admin@david888.com",
        "legal_info_url": "https://david888.com"
    })


@app.get("/skill", response_class=PlainTextResponse, include_in_schema=False)
def get_agent_skill():
    """
    獲取 Agent Skill 規範檔 (SKILL.md) 供 AI 代理人直接學習量化分析工作流。
    """
    skill_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills", "stock-quant", "SKILL.md")
    if os.path.exists(skill_path):
        with open(skill_path, "r", encoding="utf-8") as f:
            return PlainTextResponse(content=f.read(), media_type="text/markdown; charset=utf-8")
    return PlainTextResponse(content="# Stock Quant Skill not found", status_code=404)


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
    port = int(os.getenv("API_PORT", "8088"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
