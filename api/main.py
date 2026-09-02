"""
api/main.py - FastAPI Application Entrypoint for Stock Prediction & Multi-Strategy Quant Engine
"""

import os
import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from api.routes import predictions, macro, stats, market
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
    * 🦆 **DuckDB 引擎**：支援 560,000+ 歷史時序與籌碼數據零拷貝秒級查詢
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

# 支援反向代理 HTTPS 標頭轉發 (避免 307 重定向至 http)
try:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
except Exception:
    pass

# 掛載業務路由
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(macro.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")
app.include_router(market.router, prefix="/api/v1")

STATIC_ASSET_DIR = os.path.join(os.path.dirname(__file__), "static")


def _static_asset(filename: str) -> FileResponse:
    """Serve a small public branding asset used by crawlers and home screens."""
    return FileResponse(os.path.join(STATIC_ASSET_DIR, filename))


@app.api_route("/favicon.svg", methods=["GET", "HEAD"], include_in_schema=False)
def favicon_svg():
    return _static_asset("favicon.svg")


@app.api_route("/favicon-32x32.png", methods=["GET", "HEAD"], include_in_schema=False)
def favicon_png():
    return _static_asset("favicon-32x32.png")


@app.api_route("/apple-touch-icon.png", methods=["GET", "HEAD"], include_in_schema=False)
def apple_touch_icon():
    return _static_asset("apple-touch-icon.png")


@app.api_route("/og-image.png", methods=["GET", "HEAD"], include_in_schema=False)
def og_image():
    return _static_asset("og-image.png")


@app.api_route("/site.webmanifest", methods=["GET", "HEAD"], include_in_schema=False)
def site_webmanifest():
    return _static_asset("site.webmanifest")

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
    app.mount("/mcp/sse", mcp_app)
except Exception as e:
    import logging
    logging.getLogger("stock_app.api").warning(f"⚠️ WebMCP SSE 掛載略過: {e}")


@app.api_route("/mcp", methods=["GET", "POST", "HEAD"], include_in_schema=False)
@app.api_route("/mcp/", methods=["GET", "POST", "HEAD"], include_in_schema=False)
def mcp_root_endpoint():
    """回應 WebMCP Interceptor / 瀏覽器擴充套件探測，消除 307 重定向"""
    return JSONResponse({
        "name": "stock-quant-engine",
        "status": "active",
        "transport": "sse",
        "endpoints": {
            "sse": "/mcp/sse",
            "manifest": "/.well-known/mcp.json"
        }
    })


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


@app.api_route("/.well-known/mcp.json", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/mcp.json", methods=["GET", "HEAD"], include_in_schema=False)
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
            "get_latest_market_snapshot",
            "get_top_institutional_flows"
        ]
    })


@app.api_route("/llms.txt", methods=["GET", "HEAD"], response_class=PlainTextResponse, include_in_schema=False)
def get_llms_txt():
    """
    LLM Context & System Summary (Conforming to https://llmstxt.org Standard)
    """
    content = """# 888 Stock Quant Platform

> 專業級台美股多維量化決策平台，整合總體經濟宏觀風控、玄鐵均線波段回調、LSTM 深度學習時序預測、三大法人籌碼鎖碼與 2md 繁中公司簡介與即時新聞。

888 Stock Quant 遵循 [llmstxt.org](https://llmstxt.org/) 規範，為大語言模型 (LLM)、AI Agents、Claude Desktop 與自動化投資系統提供標準化、結構化的量化訊號與資料字典。

## Quantitative Strategy Endpoints

- [Triple Resonance Recommendations](/api/v1/predictions/resonance): 取得多策略共振焦點推薦（技術面均線 ∩ LSTM看漲 ∩ 籌碼鎖碼 ∩ 估值合理）
- [Xuantie Heavy Sword Pullback](/api/v1/predictions/xuantie): 查詢玄鐵重劍策略 MA60/120 均線趨勢回調技術買點
- [LSTM Bullish Top Picks](/api/v1/predictions/lstm/top-bullish): 查詢 LSTM 深度學習下一交易日預測漲幅排行
- [LSTM Bearish Top Picks](/api/v1/predictions/lstm/top-bearish): 查詢 LSTM 深度學習下一交易日預測跌幅排行
- [Market Macro Risk Regime](/api/v1/macro/latest): 查詢台股加權指數/美股S&P500、VIX恐慌指數、費城半導體趨勢與建議投資曝險比例
- [Institutional Fund Flows](/api/v1/market/institutional/top): 查詢外資、投信、自營商三大法人買賣超與籌碼集中度排行
- [Company Profile & Live News](/api/v1/market/company-profile): 整合 2md 提取個股繁體中文公司簡介、核心業務與最新新聞
- [Batch Company Profiles](/api/v1/market/company-profiles/batch): 並發非同步批次預載多支股票之公司營運摘要
- [Stock History Trajectory](/api/v1/predictions/history/2330.TW): 查詢個股歷史時序預測軌跡與法人籌碼

## Agent & Developer Discovery Standards

- [WebMCP SSE Stream](/mcp/sse): FastMCP Server-Sent Events bidirectional RPC stream
- [OpenAPI 3.1 Specification](/openapi.json): Full REST API schema in standard OpenAPI 3.1 JSON format
- [WebMCP Manifest](/.well-known/mcp.json): Model Context Protocol (MCP) server discovery manifest
- [OpenAI Plugin Manifest](/.well-known/ai-plugin.json): Standard ChatGPT / GPT Actions plugin specification
- [Agent Trading Skill](/skill): 4-step quantitative trading workflow markdown skill specification
- [Interactive Swagger UI](/docs): Live API testing and interactive documentation
- [Alternative MCP Manifest](/mcp.json): WebMCP discovery alias

## Optional & Full Documentation

- [Full LLM Context](/llms-full.txt): Complete system specification, strategy mathematical formulation, DuckDB data dictionary, and multi-asset quant architecture
- [GitHub Repository](https://github.com/tbdavid2019/stock-underdog-ml): Open source repository and automated CI/CD pipelines

## Powered By
技術提供: [david888.com](https://david888.com) | Specification: [llmstxt.org](https://llmstxt.org)
"""
    return PlainTextResponse(content=content, media_type="text/markdown; charset=utf-8")


@app.api_route("/llms-full.txt", methods=["GET", "HEAD"], response_class=PlainTextResponse, include_in_schema=False)
def get_llms_full_txt():
    """
    LLM Full Context & Architecture Spec (Conforming to https://llmstxt.org Standard)
    """
    content = """# 888 Stock Quant Platform - Full System Specification

> 專業級台美股深度學習與多維量化決策平台全量架構規格與資料模型。

## 1. System Architecture & Core Philosophy
888 Stock Quant 定位為「**盤前買進決策指南 (Pre-Market Buy Guide)**」與「**純股市時序資料庫**」，每天於台股開盤前 (08:00 TWN / 00:00 UTC) 與美股開盤前 (20:30 TWN / 12:30 UTC) 自動運算完成並推播決策。

### Core Pillars:
1. **宏觀門檻與大盤風控 (Macro & Market Regime)**:
   - 🇹🇼 台股市場：加權指數 (`^TWII`) 站穩 MA60/MA20 + 費城半導體 (`^SOX`) + VIX 國際連動。
   - 🇺🇸 美股市場：S&P 500 (`SPY`) 站穩 MA60 + VIX 恐慌指數 + 費城半導體動態風控。
2. **波段策略 - 玄鐵重劍 (Xuantie Technical Pullback)**:
   - 篩選長線多頭排列（MA60/MA120 斜率向上）、中期回調至季線/半年線支撐帶（價格在 MA60/120 之 ±3% 內）之技術性買點。
3. **短線預測 - LSTM 深度學習 (LSTM Next-Day Forecast)**:
   - 基於 PyTorch LSTM 雙層架構，輸入過去 60 日 OHLCV 與技術指標，預測次日價格與漲跌幅度排行。
4. **籌碼策略 - 三大法人鎖碼 (Institutional Accumulation)**:
   - 分析外資、投信、自營商連續買超天數、買超佔比與股本集中度，篩選「土洋合買」主力加碼標的。
5. **多維交集 - 🏆 三重共振 (Triple Resonance)**:
   - 技術面買點 ∩ LSTM 看漲 ∩ 法人鎖碼 ∩ 估值合理 (PE < 25 / PB < 3.5) 之高勝率焦點標的。
6. **2md 繁中公司簡介與即時新聞 (2md Company Profile & News)**:
   - 整合 2md.aiurl.tw / 2md.glsoft.ai / create360.ai，實時提取公司全名、主要營運業務、董事長、市值與最新新聞頭條。

## 2. DuckDB Time-Series Data Tables
平台使用高效能 DuckDB (零拷貝 Columnar 時序儲存)：
- `tw_daily_bars`: 台股上市櫃全市場日 K 線 (`date`, `ticker`, `raw_code`, `name`, `open`, `high`, `low`, `close`, `volume`)。
- `tw_institutional_daily`: 三大法人每日買賣超明細 (`date`, `ticker`, `foreign_net`, `trust_net`, `dealer_net`, `total_net`, `foreign_ratio`)。
- `tw_broker_trades`: 券商分點主力進出追蹤。
- `predictions`: 歷史預測與多維策略訊號快照。

## 3. REST API Endpoint Reference
- `GET /api/v1/macro/latest?market=tw|us`: 大盤風控與建議曝險比例。
- `GET /api/v1/predictions/resonance?index_name=台灣50|台灣中型100|SP500&limit=60`: 三重共振標的。
- `GET /api/v1/predictions/xuantie?index_name=...&limit=60`: 玄鐵重劍波段買點。
- `GET /api/v1/predictions/lstm/top-bullish?index_name=...&limit=60`: LSTM 次日看漲排行。
- `GET /api/v1/predictions/lstm/top-bearish?index_name=...&limit=60`: LSTM 次日看跌排行。
- `GET /api/v1/market/institutional/top?order_by=total_net|trust_net|foreign_net&market=ALL|TWSE|TPEX&limit=30`: 法人買賣超排行。
- `GET /api/v1/market/company-profile?ticker=9945.TW`: 取得單一個股 2md 繁中公司簡介與即時新聞。
- `POST /api/v1/market/company-profiles/batch`: 並發批次預載多支股票之公司營運摘要。
- `GET /api/v1/predictions/history/{ticker}?limit=30`: 查詢個股歷史預測軌跡與法人籌碼。

## 4. WebMCP / Chrome WebMCP Standard
本系統符合 Google Chrome WebMCP 標準，前端頁面於 `window.document.modelContext` 自動註冊 8 組量化工具，並支援 `/mcp/sse` Server-Sent Events 雙向 RPC 串流。

Specification Conformance: https://llmstxt.org/
"""
    return PlainTextResponse(content=content, media_type="text/markdown; charset=utf-8")


@app.api_route("/.webmcp/bridge.js", methods=["GET", "HEAD"], response_class=PlainTextResponse, include_in_schema=False)
def get_webmcp_bridge():
    """
    Cloudflare & Chrome WebMCP Dynamic Bridge Script (https://blog.cloudflare.com/webmcp/)
    """
    js_content = """// Cloudflare & Chrome WebMCP Bridge
// https://blog.cloudflare.com/webmcp/
(async function() {
    const modelCtx = window.document?.modelContext || window.navigator?.modelContext;
    if (!modelCtx || typeof modelCtx.registerTool !== 'function') return;
    
    const tools = [
        {
            name: 'get_market_macro_regime',
            description: '查詢最新美股宏觀風控狀態與建議投資曝險比例',
            inputSchema: { type: 'object', properties: {} },
            execute: async () => JSON.stringify(await (await fetch('/api/v1/macro/latest')).json())
        },
        {
            name: 'get_triple_resonance_stocks',
            description: '篩選三重共振強勢股（技術面 ∩ LSTM ∩ 籌碼 ∩ 估值）',
            inputSchema: { type: 'object', properties: { index_name: { type: 'string' }, limit: { type: 'integer' } } },
            execute: async (args) => {
                let url = '/api/v1/predictions/resonance?limit=' + (args?.limit || 20);
                if (args?.index_name) url += '&index_name=' + encodeURIComponent(args.index_name);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_xuantie_pullback_stocks',
            description: '查詢玄鐵重劍策略均線波段回調買點標的',
            inputSchema: { type: 'object', properties: { index_name: { type: 'string' }, limit: { type: 'integer' } } },
            execute: async (args) => {
                let url = '/api/v1/predictions/xuantie?limit=' + (args?.limit || 20);
                if (args?.index_name) url += '&index_name=' + encodeURIComponent(args.index_name);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_lstm_top_predictions',
            description: '取得 LSTM 深度學習下一交易日預測漲跌幅排行',
            inputSchema: { type: 'object', properties: { direction: { type: 'string', enum: ['bullish', 'bearish'] }, limit: { type: 'integer' } }, required: ['direction'] },
            execute: async (args) => {
                const ep = args?.direction === 'bearish' ? 'top-bearish' : 'top-bullish';
                return JSON.stringify(await (await fetch('/api/v1/predictions/lstm/' + ep + '?limit=' + (args?.limit || 20))).json());
            }
        },
        {
            name: 'get_stock_history',
            description: '查詢個股歷史時序預測軌跡與法人籌碼',
            inputSchema: { type: 'object', properties: { ticker: { type: 'string' }, limit: { type: 'integer' } }, required: ['ticker'] },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/predictions/history/' + encodeURIComponent(args.ticker.toUpperCase()) + '?limit=' + (args?.limit || 30))).json());
            }
        }
    ];

    for (const tool of tools) {
        try {
            await modelCtx.registerTool(tool);
        } catch (e) {
            console.warn('[WebMCP] registerTool error:', e);
        }
    }
    console.log('[WebMCP] Cloudflare WebMCP Bridge initialized with 5 tools.');
})();
"""
    return PlainTextResponse(content=js_content, media_type="application/javascript; charset=utf-8")


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
