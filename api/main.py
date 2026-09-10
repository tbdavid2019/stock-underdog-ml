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
        },
        "tools": [
            "get_market_macro_regime",
            "get_triple_resonance_stocks",
            "get_xuantie_pullback_stocks",
            "get_lstm_top_predictions",
            "get_timesfm_top_predictions",
            "get_stock_history",
            "get_latest_market_snapshot",
            "get_top_institutional_flows",
            "get_broker_trades_for_stock",
            "get_company_profile",
            "get_fed_rate_monitor",
            "get_us_earnings_calendar",
            "get_economic_calendar",
            "get_commodities_summary",
            "resolve_stock_ticker",
            "get_polymarket_macro_sentiment"
        ]
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
        "description": "888 Stock Quant - 專業級深度學習與多維量化決策平台 (宏觀風控、玄鐵均線、LSTM預測、Google TimesFM時序大模型、三大法人籌碼、👑四重共振、🏆三重共振、🔮雙ML共振、Polymarket 真金白銀預測市場)",
        "version": "2.4.0",
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
            "get_timesfm_top_predictions",
            "get_stock_history",
            "get_latest_market_snapshot",
            "get_top_institutional_flows",
            "get_broker_trades_for_stock",
            "get_company_profile",
            "get_fed_rate_monitor",
            "get_us_earnings_calendar",
            "get_economic_calendar",
            "get_commodities_summary",
            "resolve_stock_ticker",
            "get_polymarket_macro_sentiment"
        ]
    })


@app.api_route("/llms.txt", methods=["GET", "HEAD"], response_class=PlainTextResponse, include_in_schema=False)
def get_llms_txt():
    """
    LLM Context & System Summary (Conforming to https://llmstxt.org Standard)
    """
    content = """# 888 Stock Quant Platform

> 專業級台美股深度學習與多維量化決策平台，整合總體經濟宏觀風控、玄鐵均線波段回調、LSTM 深度學習時序預測、Google Research TimesFM 時序大模型、三大法人籌碼鎖碼、7 大板塊資金輪動與 2MD 即時財經資訊。

888 Stock Quant 遵循 [llmstxt.org](https://llmstxt.org/) 規範，為大語言模型 (LLM)、AI Agents、Claude Desktop、Cursor、Antigravity 與自動化投資決策系統提供標準化、結構化之量化訊號與資料字典。本平台核心定位為「盤前買進決策指南 (Pre-Market Buy Guide)」，每日於台股開盤前 (08:00 TWN / 00:00 UTC) 與美股開盤前 (20:30 TWN / 12:30 UTC) 自動完成運算並推播。

## Core Strategy Matrix & Resonance Framework

- [Multi-Model Resonance Guidelines](/api/v1/predictions/resonance): 多策略交集共振機制
  - 👑 **四重共振 (Quadruple Resonance)**: 同時滿足「技術均線買點 ∩ 法人籌碼鎖碼 ∩ LSTM 看漲 ∩ TimesFM 看漲」之極高信心標的。
  - 🏆 **三重共振 (Triple Resonance)**: 滿足「技術均線買點 ∩ 法人籌碼鎖碼 ∩ (LSTM ∪ TimesFM) 看漲 ∩ 估值合理 (PE<25, PB<3.5)」。
  - 🔮 **雙ML共振 (Dual-ML Resonance)**: 微觀個股量價 LSTM 與宏觀預訓練 TimesFM 同步給出看漲訊號，雙重驗證勝率最高。
  - ⚖️ **高盈虧比 (High Risk/Reward)**: TimesFM 5日預測真實盈虧比 (P50 - P0) / (P0 - P10) >= 1.5，具備非對稱獲利空間。

## Quantitative Strategy Endpoints

- [Triple & Quad Resonance Recommendations](/api/v1/predictions/resonance): 取得多策略共振焦點推薦名單 (`index_name`, `limit`)
- [TimesFM Bullish Top Picks](/api/v1/predictions/timesfm/top-bullish): 查詢 Google TimesFM 時序大模型 5 日預測漲幅與盈虧比排行
- [TimesFM Bearish Top Picks](/api/v1/predictions/timesfm/top-bearish): 查詢 Google TimesFM 時序大模型 5 日預測跌幅避險/破底防禦排行
- [LSTM Bullish Top Picks](/api/v1/predictions/lstm/top-bullish): 查詢 LSTM 深度學習次日預測漲幅排行
- [LSTM Bearish Top Picks](/api/v1/predictions/lstm/top-bearish): 查詢 LSTM 深度學習次日預測跌幅排行
- [Xuantie Heavy Sword Pullback](/api/v1/predictions/xuantie): 查詢玄鐵重劍策略 MA60 季線 / MA120 半年線趨勢回調技術買點
- [Latest Batch Snapshot](/api/v1/predictions/latest): 取得全市場最新日報批次數據（含現價、目標價、均線、估值、標籤）
- [Stock History Trajectory](/api/v1/predictions/history/2330.TW): 查詢個股歷史時序預測軌跡、均線排列與法人籌碼歷程
- [Resolve Stock Symbol](/api/v1/predictions/resolve/台積電): 將中英文股票名稱（台積電、聯發科、NVDA、Tesla）模糊解析為標準代號

## Macroeconomic Risk & Investing Catalysts (2MD)

- [Market Macro Risk Regime](/api/v1/macro/latest): 查詢台股加權指數/美股S&P 500、VIX 恐慌指數、SOX 費半趨勢與動態投資曝險比例 (0~100%)
- [Investing Macro Summary](/api/v1/macro/investing/summary): 整合 2MD 一站式總經數據（FedWatch、美股財報、大宗商品、行事曆）
- [CME FedWatch Interest Rate Monitor](/api/v1/macro/investing/fed-rate): 查詢下次 FOMC 利率決策機率分布表與倒數天數
- [US Corporate Earnings Calendar](/api/v1/macro/investing/earnings-calendar): 查詢近期美股重量級企業財報行事曆（EPS、營收預估、市值）
- [Key Commodities Summary](/api/v1/macro/investing/commodities): 查詢黃金 (Gold)、銅博士 (Copper)、原油 (WTI) 實時報價與週期走勢
- [Global Economic Calendar](/api/v1/macro/investing/economic-calendar): 查詢全球重磅總經行事曆（CPI、非農 NFP、GDP、PCE）
- [Polymarket Real-Money Sentiment](/api/v1/macro/polymarket/sentiment): 透過 2MD 與 DoH (8.8.8.8) 查詢 Polymarket 真金白銀預測市場（聯準會降息機率、地緣關稅風險、科技七巨頭 AI 動向、美國經濟衰退預期）

## Institutional & Broker Fund Flows

- [Institutional Fund Flows](/api/v1/market/institutional/top): 查詢外資、投信、自營商三大法人買賣超排行與持股集中度
- [Broker Branch Trades](/api/v1/market/broker/summary/2330.TW): 查詢券商關鍵分點主力進出追蹤與多空明細
- [Company Profile & Live News](/api/v1/market/company-profile?ticker=2330.TW): 整合 2MD 提取個股繁體中文公司簡介、核心業務、市值與即時新聞
- [Batch Company Profiles](/api/v1/market/company-profiles/batch): 並發非同步批次預載多支股票之公司營運摘要

## Model Context Protocol (MCP) Tools

本平台提供 16 大標準 FastMCP 工具函數（支援 stdio 與 SSE 雙向通訊）：
- `get_market_macro_regime`: 評估大盤宏觀風控情境與建議投資曝險比例。
- `get_triple_resonance_stocks`: 查詢 👑四重共振、🏆三重共振與 🔮雙ML共振焦點股。
- `get_timesfm_top_predictions`: 查詢 Google TimesFM 時序大模型 5 日漲跌幅排行與盈虧比。
- `get_lstm_top_predictions`: 查詢 LSTM 深度學習次日漲跌幅排行。
- `get_xuantie_pullback_stocks`: 查詢玄鐵重劍 MA60/120 趨勢回調買點標的。
- `get_stock_history`: 查詢特定股票歷史時序量化預測軌跡與法人籌碼。
- `get_latest_market_snapshot`: 取得最新量化日報批次數據快照。
- `get_top_institutional_flows`: 查詢三大法人買賣超焦點股排行榜。
- `get_broker_trades_for_stock`: 查詢券商關鍵主力分點進出明細。
- `get_company_profile`: 透過 2MD 查詢個股繁體中文公司簡介與即時新聞。
- `get_fed_rate_monitor`: 透過 2MD 查詢 CME FedWatch 聯準會利率決策機率。
- `get_us_earnings_calendar`: 透過 2MD 查詢美股重量級企業財報行事曆。
- `get_economic_calendar`: 透過 2MD 查詢全球重大總經行事曆。
- `get_commodities_summary`: 透過 2MD 查詢黃金、原油、銅博士行情。
- `resolve_stock_ticker`: 將模糊搜尋之公司名稱快速解析為標準交易代號。
- `get_polymarket_macro_sentiment`: 透過 2MD/DoH 查詢 Polymarket 真金白銀預測市場之宏觀風控與重大事件情緒。

## Agent & Developer Discovery Standards

- [WebMCP SSE Stream](/mcp/sse): FastMCP Server-Sent Events bidirectional RPC stream
- [OpenAPI 3.1 Specification](/openapi.json): Full REST API schema in standard OpenAPI 3.1 JSON format
- [WebMCP Manifest](/.well-known/mcp.json): Model Context Protocol (MCP) server discovery manifest
- [OpenAI Plugin Manifest](/.well-known/ai-plugin.json): Standard ChatGPT / GPT Actions plugin specification
- [Agent Trading Skill](/skill): 4-step quantitative trading workflow markdown skill specification
- [Interactive Swagger UI](/docs): Live API testing and interactive documentation
- [Chrome WebMCP Dynamic Bridge](/.webmcp/bridge.js): Cloudflare & Chrome WebMCP registration script
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

> 專業級台美股深度學習與多維量化決策平台全量架構規格、數學模型定義與資料庫字典。

## 1. System Architecture & Core Philosophy
888 Stock Quant 定位為「**盤前買進決策指南 (Pre-Market Buy Guide)**」與「**純股市時序資料庫**」，每天於台股開盤前 (08:00 TWN / 00:00 UTC) 與美股開盤前 (20:30 TWN / 12:30 UTC) 自動運算完成並推播決策。系統透過 `--market auto` 實施時段防護，白天專注台股、夜間專注美股，非指定全市場時杜絕算力浪費。

### Core Quantitative Pillars:
1. **宏觀門檻與大盤風控 (Macro & Market Regime)**:
   - 🇹🇼 台股市場：加權指數 (`^TWII`) 站穩 MA60/MA20 + 費城半導體 (`^SOX`) + VIX 國際連動。
   - 🇺🇸 美股市場：S&P 500 (`SPY`) 站穩 MA60 + VIX 恐慌指數 + 費城半導體動態風控。
   - 曝險比例：全面多頭 (100%)、多頭回調 (85%)、避險防禦 (30~50%)、極度恐慌 (0%)。當費半破季線時，科技股部位強制上限 30%。
2. **波段策略 - 玄鐵重劍 (Xuantie Technical Pullback)**:
   - 順大勢：MA60 > MA120 或斜率向上。
   - 逆小勢：價格回調至季線/半年線支撐帶（價格在 MA60/120 之 ±3% 內，且未跌破超過 -1%）。
3. **短線預測 - LSTM 深度學習 (LSTM Next-Day Forecast)**:
   - 輸入過去 60 日 OHLCV 與技術指標，2 層雙向 LSTM 輸出次日收盤價預測與漲跌幅潛力。
4. **時序基礎大模型 - Google TimesFM (TimesFM 5-Day Forecast & Quantiles)**:
   - 採用 Google Research 預訓練 Decoder 架構時序基礎大模型 (TimesFM 2.5 500M)。
   - 輸出 1~5 日預測目標價軌跡，並計算 P10 (下行防守位)、P50 (中位預期) 與 P90 (上行獲利位)。
   - 真實盈虧比 (Risk/Reward Ratio): $RR = \\frac{P_{50} - P_0}{P_0 - P_{10}}$。當 $RR \\ge 1.5$ 且 $P_{50} > P_0$ 時判定為高勝率看漲標的。
5. **籌碼策略 - 三大法人鎖碼 (TWSE Institutional Accumulation)**:
   - 分析外資、投信、自營商買賣超。篩選「投信連買 >= 3 天」或「土洋合買（外資與投信同向買超）」主力標的。
6. **產業板塊資金輪動 (Sector Rotation)**:
   - 追蹤 7 大核心板塊 10D (40%) + 15D (30%) + 20D (30%) 動量加權資金流，鎖定市場前 3 大主流板塊。
7. **多維交集共振架構 (Multi-Model Resonance Hierarchy)**:
   - 👑 **四重共振**: 玄鐵技術買點 ∩ 法人鎖碼 ∩ LSTM 看漲 ∩ TimesFM 看漲 (勝率頂級)。
   - 🏆 **三重共振**: 玄鐵技術買點 ∩ 法人鎖碼 ∩ (LSTM ∪ TimesFM) 看漲 ∩ 估值合理 (PE<25 / PB<3.5)。
   - 🔮 **雙ML共振**: LSTM ∩ TimesFM 同步看漲（微觀記憶 ∩ 宏觀預訓練波形共振）。
   - ⚖️ **高盈虧比**: TimesFM 盈虧比 $RR \\ge 2.0$。

## 2. Mathematical Formulations

### 2.1 TimesFM Quantile Risk/Reward Ratio
給定現價 $P_0$、未來 5 日 TimesFM 中位數預測價 $P_{50}$、下行 10% 分位數價格 $P_{10}$：
$$\\text{Potential (\\%)} = \\frac{P_{50} - P_0}{P_0} \\times 100\\%$$
$$\\text{Risk/Reward Ratio} = \\begin{cases} \\frac{P_{50} - P_0}{P_0 - P_{10}}, & \\text{if } P_{50} > P_0 \\text{ and } P_0 > P_{10} \\\\ 0.0, & \\text{otherwise} \\end{cases}$$

### 2.2 Composite Weighted Score
綜合評分以動態權重計算，若大盤處於降級狀態則折減曝險比例：
$$\\text{Score}_{raw} = \\frac{\\sum_{s} w_s \\cdot S_s}{\\sum_s w_s}$$
$$\\text{Score}_{final} = \\text{Score}_{raw} \\times \\text{Exposure}$$

## 3. DuckDB Time-Series Data Tables

- `tw_daily_bars`:
  `date` (DATE), `ticker` (VARCHAR), `raw_code` (VARCHAR), `name` (VARCHAR), `open` (DOUBLE), `high` (DOUBLE), `low` (DOUBLE), `close` (DOUBLE), `volume` (BIGINT), `market` (VARCHAR)
- `tw_institutional_daily`:
  `date` (DATE), `ticker` (VARCHAR), `raw_code` (VARCHAR), `name` (VARCHAR), `foreign_net` (BIGINT), `trust_net` (BIGINT), `dealer_net` (BIGINT), `total_net` (BIGINT), `foreign_ratio` (DOUBLE), `market` (VARCHAR)
- `tw_broker_trades`:
  `date` (DATE), `ticker` (VARCHAR), `broker_name` (VARCHAR), `buy_shares` (BIGINT), `sell_shares` (BIGINT), `net_shares` (BIGINT)
- `predictions`:
  `index_name` (VARCHAR), `model_name` (VARCHAR), `strategy_type` (VARCHAR), `ticker` (VARCHAR), `current_price` (DOUBLE), `predicted_price` (DOUBLE), `potential` (DOUBLE), `ma5` (DOUBLE), `ma10` (DOUBLE), `ma60` (DOUBLE), `ma120` (DOUBLE), `ma250` (DOUBLE), `pullback_type` (VARCHAR), `pe` (DOUBLE), `pb` (DOUBLE), `forward_pe` (DOUBLE), `ev_ebitda` (DOUBLE), `period` (VARCHAR), `timestamp` (TIMESTAMP), `macro_regime` (VARCHAR), `trust_net_5d` (DOUBLE), `foreign_net_5d` (DOUBLE), `tags` (VARCHAR)

## 4. REST API Endpoints Specification

### 4.1 Quantitative Strategies
- `GET /api/v1/predictions/resonance?index_name=台灣50&limit=30`: 查詢三重/四重共振推薦清單。
- `GET /api/v1/predictions/timesfm/top-bullish?index_name=...&limit=20`: TimesFM 5日看漲榜與盈虧比。
- `GET /api/v1/predictions/timesfm/top-bearish?index_name=...&limit=20`: TimesFM 5日看跌避險榜。
- `GET /api/v1/predictions/lstm/top-bullish?index_name=...&limit=20`: LSTM 次日看漲榜。
- `GET /api/v1/predictions/lstm/top-bearish?index_name=...&limit=20`: LSTM 次日看跌榜。
- `GET /api/v1/predictions/xuantie?index_name=...&limit=20`: 玄鐵重劍 MA60/120 波段買點。
- `GET /api/v1/predictions/latest?index_name=...&limit=50`: 最新日報完整批次快照。
- `GET /api/v1/predictions/history/{ticker}?limit=30`: 單一標的歷史預測軌跡與籌碼時序。
- `GET /api/v1/predictions/resolve/{query}`: 模糊匹配股票名稱並解析為標準代號。

### 4.2 Macro & Investing Catalysts
- `GET /api/v1/macro/latest?market=tw|us`: 大盤風控與建議曝險比例。
- `GET /api/v1/macro/investing/summary`: 整合 CME FedWatch、美股財報、大宗商品與財經日曆。
- `GET /api/v1/macro/investing/fed-rate`: 聯準會利率決策機率分布與 FOMC 倒數。
- `GET /api/v1/macro/investing/earnings-calendar`: 美股重量級企業財報公布行事曆。
- `GET /api/v1/macro/investing/commodities`: 黃金、銅博士、WTI 原油行情與週期漲跌。
- `GET /api/v1/macro/investing/economic-calendar`: 全球重磅總經行事曆（CPI、非農等）。
- `GET /api/v1/macro/polymarket/sentiment`: Polymarket 真金白銀預測市場宏觀情緒（FOMC 利率、地緣關稅、科技巨頭、經濟衰退）。

### 4.3 Institutional & Company Profile
- `GET /api/v1/market/institutional/top?order_by=total_net&limit=30`: 三大法人買賣超排行。
- `GET /api/v1/market/broker/summary/{ticker}?days=20`: 券商關鍵分點主力累計買賣超。
- `GET /api/v1/market/company-profile?ticker=2330.TW`: 2MD 繁體中文公司簡介與即時新聞。
- `POST /api/v1/market/company-profiles/batch`: 並發批次預載多支股票之公司營運摘要。

## 5. Model Context Protocol (MCP) Server
- FastMCP Server: `mcp_server.py`
- WebMCP Endpoint: `/mcp/sse`
- WebMCP Manifest: `/.well-known/mcp.json`
- Tools (16 Native Tools): `get_market_macro_regime`, `get_triple_resonance_stocks`, `get_timesfm_top_predictions`, `get_lstm_top_predictions`, `get_xuantie_pullback_stocks`, `get_stock_history`, `get_latest_market_snapshot`, `get_top_institutional_flows`, `get_broker_trades_for_stock`, `get_company_profile`, `get_fed_rate_monitor`, `get_us_earnings_calendar`, `get_economic_calendar`, `get_commodities_summary`, `resolve_stock_ticker`, `get_polymarket_macro_sentiment`.

## 6. Recommended 4-Step Agent Trading Workflow
1. **檢查宏觀風控**: 呼叫 `get_market_macro_regime()` 決定整體建議曝險 (0%~100%)。
2. **首選共振焦點**: 呼叫 `get_triple_resonance_stocks()`，優先挑選含 `👑四重共振`、`🔮雙ML共振`、`高盈虧比` 標的。
3. **分流補充選股**:
   - 波段投資人: `get_xuantie_pullback_stocks()` (MA60/120 回踩買點)。
   - 大模型動量投資人: `get_timesfm_top_predictions()` (5日盈虧比優勢)。
   - 短線爆發投資人: `get_lstm_top_predictions()` (次日動量)。
4. **個股確認與防守**: 呼叫 `get_stock_history()` 與 `get_company_profile()` 檢視近期走勢、法人籌碼支撐與公司基本面。

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
            name: 'get_timesfm_top_predictions',
            description: '取得 Google Research TimesFM 時序大模型 5 日預測漲跌幅排行與盈虧比',
            inputSchema: { type: 'object', properties: { direction: { type: 'string', enum: ['bullish', 'bearish'] }, limit: { type: 'integer' }, index_name: { type: 'string' } }, required: ['direction'] },
            execute: async (args) => {
                const ep = args?.direction === 'bearish' ? 'top-bearish' : 'top-bullish';
                let url = '/api/v1/predictions/timesfm/' + ep + '?limit=' + (args?.limit || 20);
                if (args?.index_name) url += '&index_name=' + encodeURIComponent(args.index_name);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_stock_history',
            description: '查詢個股歷史時序預測軌跡與法人籌碼',
            inputSchema: { type: 'object', properties: { ticker: { type: 'string' }, limit: { type: 'integer' } }, required: ['ticker'] },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/predictions/history/' + encodeURIComponent(args.ticker.toUpperCase()) + '?limit=' + (args?.limit || 30))).json());
            }
        },
        {
            name: 'get_top_institutional_flows',
            description: '查詢三大法人買賣超焦點股排行榜',
            inputSchema: { type: 'object', properties: { order_by: { type: 'string', enum: ['total_net', 'trust_net', 'foreign_net'] }, limit: { type: 'integer' }, market: { type: 'string' } } },
            execute: async (args) => {
                const url = '/api/v1/market/institutional/top?order_by=' + (args?.order_by || 'total_net') + '&limit=' + (args?.limit || 20) + '&market=' + (args?.market || 'ALL');
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_fed_rate_monitor',
            description: '透過 2MD 查詢 Investing.com FedWatch 聯準會利率決策機率與 FOMC 倒數',
            inputSchema: { type: 'object', properties: { force_refresh: { type: 'boolean' } } },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/macro/investing/fed-rate?force_refresh=' + (args?.force_refresh ? 'true' : 'false'))).json());
            }
        },
        {
            name: 'get_us_earnings_calendar',
            description: '透過 2MD 查詢 Investing.com 美股重量級財報行事曆（EPS、營收預估）',
            inputSchema: { type: 'object', properties: { force_refresh: { type: 'boolean' } } },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/macro/investing/earnings-calendar?force_refresh=' + (args?.force_refresh ? 'true' : 'false'))).json());
            }
        },
        {
            name: 'get_latest_market_snapshot',
            description: '取得全市場最新日報批次數據快照',
            inputSchema: { type: 'object', properties: { index_name: { type: 'string' }, limit: { type: 'integer' } } },
            execute: async (args) => {
                let url = '/api/v1/predictions/latest?limit=' + (args?.limit || 50);
                if (args?.index_name) url += '&index_name=' + encodeURIComponent(args.index_name);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_broker_trades_for_stock',
            description: '查詢券商關鍵分點主力買賣超明細與累計走勢',
            inputSchema: { type: 'object', properties: { ticker: { type: 'string' }, days: { type: 'integer' } }, required: ['ticker'] },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/market/broker/summary/' + encodeURIComponent(args.ticker.toUpperCase()) + '?days=' + (args?.days || 20))).json());
            }
        },
        {
            name: 'get_company_profile',
            description: '透過 2MD 查詢個股繁體中文公司簡介與即時新聞',
            inputSchema: { type: 'object', properties: { ticker: { type: 'string' } }, required: ['ticker'] },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/market/company-profile?ticker=' + encodeURIComponent(args.ticker))).json());
            }
        },
        {
            name: 'get_economic_calendar',
            description: '透過 2MD 查詢全球重大總經行事曆（CPI、非農等）',
            inputSchema: { type: 'object', properties: { force_refresh: { type: 'boolean' } } },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/macro/investing/economic-calendar?force_refresh=' + (args?.force_refresh ? 'true' : 'false'))).json());
            }
        },
        {
            name: 'get_commodities_summary',
            description: '透過 2MD 查詢關鍵大宗商品（黃金、原油、期銅）即時行情',
            inputSchema: { type: 'object', properties: { force_refresh: { type: 'boolean' } } },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/macro/investing/commodities?force_refresh=' + (args?.force_refresh ? 'true' : 'false'))).json());
            }
        },
        {
            name: 'resolve_stock_ticker',
            description: '將中英文股票名稱或代號模糊解析為標準代號',
            inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
            execute: async (args) => {
                return JSON.stringify(await (await fetch('/api/v1/predictions/resolve/' + encodeURIComponent(args.query))).json());
            }
        },
        {
            name: 'get_polymarket_macro_sentiment',
            description: '透過 2MD 與 DoH 查詢 Polymarket 預測市場宏觀情緒與重大事件機率',
            inputSchema: { type: 'object', properties: { category: { type: 'string' }, force_refresh: { type: 'boolean' } } },
            execute: async (args) => {
                let url = '/api/v1/macro/polymarket/sentiment?force_refresh=' + (args?.force_refresh ? 'true' : 'false');
                if (args?.category) url += '&category=' + encodeURIComponent(args.category);
                return JSON.stringify(await (await fetch(url)).json());
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
    console.log('[WebMCP] Cloudflare WebMCP Bridge initialized with ' + tools.length + ' tools.');
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
        "description_for_model": "888 Stock Quant 專業級深度學習與多維量化決策系統 (宏觀風控、玄鐵均線、LSTM預測、Google TimesFM時序大模型、三大法人籌碼、👑四重共振、🏆三重共振、🔮雙ML共振、Polymarket 真金白銀預測市場)。提供每日台美股選股清單、目標價預測、法人籌碼鎖碼、均線波段買點與個股歷史走勢查詢。支援 16 大原生 MCP 量化工具與 WebMCP 協議。",
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
