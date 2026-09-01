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

# 掛載原生 WebMCP (SSE 串流協議) 供遠端 Agent 即時連接
try:
    from mcp_server import mcp as mcp_instance
    from mcp.server.transport_security import TransportSecuritySettings
    mcp_instance.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"]
    )
    mcp_app = mcp_instance.sse_app()
    app.mount("/mcp", mcp_app)
except Exception as e:
    import logging
    logging.getLogger("stock_app.api").warning(f"⚠️ WebMCP SSE 掛載略過: {e}")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
def root():
    """
    量化決策平台首頁：提供即時操盤儀表板與 Agent / MCP 對接指南。
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>888 Stock Quant Platform API is running. Visit <a href='/docs'>/docs</a>.</h1>")


@app.get("/.well-known/mcp.json", include_in_schema=False)
@app.get("/mcp.json", include_in_schema=False)
def get_webmcp_manifest():
    """
    WebMCP Remote Discovery Manifest
    """
    return JSONResponse(content={
        "name": "stock-quant-engine",
        "description": "888 Stock Quant - 專業級深度學習與多維量化決策平台",
        "version": "2.2.0",
        "transport": "sse",
        "endpoints": {
            "sse": "/mcp/sse",
            "messages": "/mcp/messages/"
        },
        "tools": [
            "get_market_macro_regime",
            "get_triple_resonance_stocks",
            "get_xuantie_pullback_stocks",
            "get_lstm_top_predictions",
            "get_stock_history",
            "get_latest_market_snapshot"
        ]
    })


@app.get("/llms.txt", response_class=PlainTextResponse, include_in_schema=False)
@app.get("/llms-full.txt", response_class=PlainTextResponse, include_in_schema=False)
def get_llms_txt():
    """
    LLM Context & System Summary (https://llmstxt.org Standard)
    """
    content = """# 888 Stock Quant Platform

> 專業級台美股深度學習與多維量化交易決策平台。

## Overview
888 Stock Quant 整合總體經濟宏觀風控、玄鐵均線波段回調、LSTM 深度學習時序預測、三大法人籌碼鎖碼與 DuckDB 高效時序資料庫，提供即時量化選股與操盤指引。

## Agent & MCP Connectivity
- **WebMCP SSE Stream**: `/mcp/sse` (FastMCP Server-Sent Events bidirectional RPC)
- **WebMCP Manifest**: `/.well-known/mcp.json` & `/mcp.json`
- **Agent Skill Spec**: `/skill` (4-step quantitative trading workflow)
- **OpenAPI 3.1 Spec**: `/openapi.json`
- **Plugin Manifest**: `/.well-known/ai-plugin.json`
- **Swagger Docs**: `/docs`

## Core Quantitative Tools
- `get_market_macro_regime`: 取得 S&P500 / VIX / SOX 即時宏觀風控狀態與建議投資曝險比例。
- `get_triple_resonance_stocks`: 篩選三重共振（技術面 ∩ LSTM ∩ 籌碼 ∩ 估值）強勢標的。
- `get_xuantie_pullback_stocks`: 查詢 MA60/120 季線/半年線波段回調買點。
- `get_lstm_top_predictions`: 查詢 LSTM 深度學習下一交易日預測漲跌幅排行。
- `get_stock_history`: 查詢個股於 DuckDB 中的歷史時序預測軌跡與法人籌碼。
- `get_latest_market_snapshot`: 取得市場即時全景摘要與資料庫統計。

## Powered By
技術提供: [david888.com](https://david888.com)
"""
    return PlainTextResponse(content=content, media_type="text/markdown; charset=utf-8")


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
def get_ai_plugin_manifest():
    """
    Web Plugin Standard Discovery Manifest
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
