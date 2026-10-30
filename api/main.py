"""
api/main.py - FastAPI Application Entrypoint for Stock Prediction & Multi-Strategy Quant Engine
"""

import os
import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from api.routes import predictions, macro, stats, market, decision, earnings_call, podcast, discovery
from api.schemas import HealthResponse
from data.duckdb_manager import DuckDBManager
from core.version import APP_VERSION

app = FastAPI(
    title="888 StockQuant Platform API",
    description="""
    🚀 **888 StockQuant 多維量化交易決策系統 REST API**
    
    支援基於 DuckDB 高效能時序資料庫之即時量化訊號查詢：
    * 🗡️ **玄鐵重劍策略**：MA60/120 均線趨勢回調技術買點
    * 🔮 **時序大模型矩陣**：Mistral Timestral-209M 與 Google TimesFM 雙時序基礎模型預測與波形共振
    * ⭐ **多維共振篩選**：技術 ∩ 籌碼 ∩ 雙大模型 ∩ 估值 之 `👑四重共振` / `🏆三重共振` / `土洋合買`
    * 🌐 **美股宏觀門檻**：S&P 500 / VIX / 費城半導體 即時曝險評估
    * 🦆 **DuckDB 引擎**：支援 560,000+ 歷史時序與籌碼數據零拷貝秒級查詢
    * 🏛️ **法說會催化劑**：台美雙軌法人說明會 (TWSE/TPEx + 美股巨頭) 時程與簡報追蹤
    * 🎙️ **Podcast 情報雷達**：頂級科技/財經 Podcast AI 繁中精煉與上市公司實體行情連動
    """,
    version=APP_VERSION,
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
app.include_router(decision.router, prefix="/api/v1")
app.include_router(earnings_call.router, prefix="/api/v1")
app.include_router(podcast.router, prefix="/api/v1")
app.include_router(discovery.router, prefix="/api/v1")

STATIC_ASSET_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# 掛載靜態檔案路由提供 /static/js/ 等靜態模組
try:
    from fastapi.staticfiles import StaticFiles

    class NoCacheStaticFiles(StaticFiles):
        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

    app.mount("/static", NoCacheStaticFiles(directory=STATIC_ASSET_DIR), name="static")
except Exception as e:
    import logging
    logging.getLogger("stock_app.api").warning(f"⚠️ StaticFiles 掛載略過: {e}")


def _static_asset(filename: str) -> FileResponse:
    """Serve a small public branding asset used by crawlers and home screens."""
    return FileResponse(os.path.join(STATIC_ASSET_DIR, filename))


def get_asset_fingerprint() -> str:
    """產生包含靜態腳本修改時間之指紋版本號，確保代碼更新時全網瀏覽器與 CDN 零快取殘留"""
    try:
        app_js_path = os.path.join(STATIC_ASSET_DIR, "js", "app.js")
        mtime = int(os.path.getmtime(app_js_path))
        return f"{APP_VERSION}.{mtime}"
    except Exception:
        return APP_VERSION


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


def _render_jinja_template(template_name: str = "index.html") -> str:
    """
    Render template resolving partials while strictly avoiding collision with Vue {{ }} syntax.
    Tries native Jinja2 with custom delimiters [[ ]], falling back to recursive include expansion.
    """
    fingerprint = get_asset_fingerprint()
    # 1. 優先嘗試 Jinja2 Environment 並以 [[ ]] 作為變數定界符，完全保護 Vue 的 {{ }}
    try:
        from jinja2 import Environment, FileSystemLoader
        jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            variable_start_string="[[",
            variable_end_string="]]",
            autoescape=False
        )
        template = jinja_env.get_template(template_name)
        return template.render(APP_VERSION=fingerprint).replace("__APP_VERSION__", fingerprint)
    except Exception:
        pass

    # 2. 備援處理：以正則表達式遞迴解析 {% include 'partials/...' %}
    import re
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        return ""
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _replace_includes(text: str) -> str:
        pattern = re.compile(r"{%\s*include\s+['\"]([^'\"]+)['\"]\s*%}")
        def _repl(match):
            sub_name = match.group(1)
            sub_path = os.path.join(TEMPLATES_DIR, sub_name)
            if os.path.exists(sub_path):
                with open(sub_path, "r", encoding="utf-8") as sf:
                    return _replace_includes(sf.read())
            return ""
        return pattern.sub(_repl, text)

    resolved = _replace_includes(content)
    return resolved.replace("__APP_VERSION__", fingerprint).replace("[[ APP_VERSION ]]", fingerprint)

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
            "get_timestral_top_predictions",
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
            "get_polymarket_macro_sentiment",
            "get_clef_stock_verdict"
        ]
    })


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
def root():
    """
    量化決策平台首頁：提供即時操盤儀表板與 Agent / MCP 對接指南。
    """
    content = _render_jinja_template("index.html")
    if content:
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>888 Stock Quant Platform API is running. Visit <a href='/docs'>/docs</a>.</h1>")


@app.api_route("/.well-known/mcp.json", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/mcp.json", methods=["GET", "HEAD"], include_in_schema=False)
def get_webmcp_manifest():
    """
    WebMCP Remote Discovery Manifest
    """
    return JSONResponse(content={
        "name": "stock-quant-engine",
        "description": "888 Stock Quant - 專業級深度學習與多維量化決策平台 (宏觀風控、玄鐵均線、Mistral Timestral-209M、Google TimesFM時序大模型、三大法人籌碼、👑四重共振、🏆三重共振、🔮雙時序共振、Polymarket 真金白銀預測市場、Clef System One 決策大模型)",
        "version": APP_VERSION,
        "transport": "sse",
        "endpoints": {
            "sse": "/mcp/sse",
            "messages": "/mcp/messages/"
        },
        "tools": [
            "get_market_macro_regime",
            "get_triple_resonance_stocks",
            "get_xuantie_pullback_stocks",
            "get_timestral_top_predictions",
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
            "get_polymarket_macro_sentiment",
            "get_clef_stock_verdict",
            "get_upcoming_earnings_calls",
            "get_ticker_earnings_calls",
            "get_company_network",
            "open_company_network",
            "get_company_tenders"
        ]
    })


@app.api_route("/llms.txt", methods=["GET", "HEAD"], response_class=PlainTextResponse, include_in_schema=False)
def get_llms_txt():
    """
    LLM Context & System Summary (Conforming to https://llmstxt.org Standard)
    """
    content = """# 888 Stock Quant Platform

> 專業級台美股深度學習與多維量化決策平台，整合總體經濟宏觀風控、玄鐵均線波段回調、Mistral Timestral-209M 時序大模型、Google Research TimesFM 時序大模型、雙時序模型波形共振、三大法人籌碼鎖碼、7 大板塊資金輪動與 2MD 即時財經資訊。

> ⚠️ **Continuous Feature Evolution Active**: Caller LLM Agents and automated systems should dynamically fetch the latest live specifications from `https://stockdata.david888.com/llms.txt` and `https://stockdata.david888.com/skill` at the start of each session/task to leverage newly released AI models, FastMCP tools, and multi-market data endpoints.

888 Stock Quant 遵循 [llmstxt.org](https://llmstxt.org/) 規範，為大語言模型 (LLM)、AI Agents、Claude Desktop、Cursor、Antigravity 與自動化投資決策系統提供標準化、結構化之量化訊號與資料字典。本平台核心定位為「盤前買進決策指南 (Pre-Market Buy Guide)」，每日於台股開盤前 (08:00 TWN / 00:00 UTC) 與美股開盤前 (20:30 TWN / 12:30 UTC) 自動完成運算並推播。

## Core Strategy Matrix & Resonance Framework

- [Multi-Model Resonance Guidelines](/api/v1/predictions/resonance): 多策略交集共振機制
  - 👑 **四重共振 (Quadruple Resonance)**: 同時滿足「玄鐵季線回踩 ∩ 法人籌碼鎖碼 ∩ TimesFM 看漲 ∩ Timestral 看漲」四大頂級條件，為系統最高勝率極選。
  - 🏆 **三重共振 (Triple Resonance)**: 滿足「技術均線買點 ∩ 法人籌碼鎖碼 ∩ (TimesFM ∪ Timestral) 看漲 ∩ 估值合理 (PE<25, PB<3.5)」。
  - 🔮 **雙模型共振 (Dual-Foundation Model Resonance)**: Google TimesFM (200M) 與 Mistral Timestral (209M) 雙時序大模型波形共識，目標價 $P_{res} = 0.5 \\times P_{TimesFM} + 0.5 \\times P_{Timestral}$ 同步看漲。
  - 🗡️ **玄鐵均線波段 (Xuantie Moving Average)**: 「順大勢（MA60/120多頭）、逆小勢（回踩均線支撐）」，精準捕捉波段回踩起漲點。
  - ⚖️ **高盈虧比 (High Risk/Reward)**: 時序模型 Horizon 預測真實盈虧比 $(P_{50,h} - P_0) / (P_0 - P_{10,h}) \\ge 1.5$，具備非對稱獲利防守空間。

## Quantitative Strategy Endpoints

- [Resonance Focus Picks](/api/v1/predictions/resonance): 取得雙時序大模型與多策略共振推薦名單 (`index_name`, `limit`)
- [Timestral Bullish Top Picks](/api/v1/predictions/timestral/top-bullish): 查詢 Mistral Timestral-209M 時序大模型 5 日預測漲幅與盈虧比排行
- [Timestral Bearish Defense Picks](/api/v1/predictions/timestral/top-bearish): 查詢 Mistral Timestral-209M 時序大模型 5 日預測跌幅避險/破底防禦排行
- [TimesFM Bullish Top Picks](/api/v1/predictions/timesfm/top-bullish): 查詢 Google TimesFM 時序大模型 5 日預測漲幅與盈虧比排行
- [TimesFM Bearish Defense Picks](/api/v1/predictions/timesfm/top-bearish): 查詢 Google TimesFM 時序大模型 5 日預測跌幅避險/破底防禦排行
- [Xuantie Heavy Sword Pullback](/api/v1/predictions/xuantie): 查詢玄鐵重劍策略 MA60 季線 / MA120 半年線趨勢回調技術買點
- [Latest Batch Snapshot](/api/v1/predictions/latest): 取得全市場最新日報批次數據（含現價、目標價、均線、估值、標籤）
- [Stock History Trajectory](/api/v1/predictions/history/2330.TW): 查詢個股歷史時序預測軌跡、均線排列與法人籌碼歷程
- [Resolve Stock Symbol](/api/v1/predictions/resolve/台積電): 將中英文股票名稱（台積電、聯發科、NVDA、Tesla）模糊解析為標準代號

## Macroeconomic Risk & Investing Catalysts (2MD)

- [Market Macro Risk Regime](/api/v1/macro/latest): 查詢台股加權指數/美股S&P 500、VIX 恐慌指數、SOX 費半趨勢與動態投資曝險比例 (0~100%)
- [Investing Macro Summary](/api/v1/macro/investing/summary): 整合 2MD 一站式總經數據（FedWatch、美股財報、大宗商品、行事曆）
- [CME FedWatch Interest Rate Monitor](/api/v1/macro/investing/fed-rate): 查詢下次 FOMC 利率決策機率分布表與倒數天數
- [US Corporate Earnings Calendar](/api/v1/macro/investing/earnings-calendar): 查詢近期美股重量級企業財報行事曆（EPS、營收預估、市值）
- [Key Commodities Summary](/api/v1/macro/investing/commodities): 查詢黃金 (Gold)、銅 (Copper)、原油 (WTI) 實時報價與週期走勢
- [Global Economic Calendar](/api/v1/macro/investing/economic-calendar): 查詢全球重磅總經行事曆（CPI、非農 NFP、GDP、PCE）
- [Polymarket Real-Money Sentiment](/api/v1/macro/polymarket/sentiment): 透過 2MD 與 DoH (8.8.8.8) 查詢 Polymarket 真金白銀預測市場；機率欄位統一為 0~100 百分比，上游失敗會回傳 `success=false`，可用舊快取時標記 `stale=true`

## Institutional & Broker Fund Flows

- [Institutional Fund Flows](/api/v1/market/institutional/top): 查詢外資、投信、自營商三大法人買賣超排行與持股集中度
- [Broker Branch Trades](/api/v1/market/broker/summary/2330.TW): 查詢券商關鍵分點主力進出追蹤與多空明細
- [Company Profile & Live News](/api/v1/market/company-profile?ticker=2330.TW): 整合 2MD 提取個股繁體中文公司簡介、核心業務、市值與即時新聞
- [Batch Company Profiles](/api/v1/market/company-profiles/batch): 並發非同步批次預載多支股票之公司營運摘要

## Global Corporate Earnings Calls & Guidance Intelligence (法說會與官方指引)

整合 BigGo 財經 (finance.biggo.com.tw) 與公開資訊觀測站 (MOPS) 官方後端 API，涵蓋台灣 (TWSE/TPEx)、美股 (US)、日股 (JP)、韓股 (KR) 與全球重點市場，秒級聚合跨市場法說會時程、高管營收/毛利率/資本支出官方財測指引 (Guidance)、法說簡報 PDF 原始下載、Webcast 直播回放與機構問答精華：
- [Upcoming Earnings Calls](/api/v1/earnings-call/upcoming?market=ALL&days=14): 查詢近期跨市場即將召開之法說會清單，支援市場篩選 (`market: ALL / TW / US / JP / KR / OTHER`)、權值股篩選 (`is_weight_stock: true`) 與天數範圍 (`days`)，自動換算雙時區 (CST/EST) 與倒數日。
- [Monthly Earnings Calendar](/api/v1/earnings-call/calendar?year=2026&month=10&market=ALL): 查詢指定月之每日法說會召開場次聚合統計與公司清單。
- [Stock Earnings History & Presentation](/api/v1/earnings-call/2330.TW): 查詢特定標的 (如 2330.TW、NVDA、7203.T) 歷史與未來法說會日程、歷次官方簡報 PDF、影音回放與管理層指引。
- [Trigger Full Sync](/api/v1/earnings-call/sync): 觸發後台執行 6 視窗並行多市場法說會資料抓取並全量寫入 DuckDB。

## AI Tech Radar & Discovery Engine (AI 產業鏈前沿挖掘機)

定時主動爬取學術界 arXiv 前沿論文（cs.AI / cs.RO / cs.LG / cs.CV），由底層技術突破點映射至 5 大硬體瓶頸（CPO 光互連、HBM/KV-cache、液冷散熱、具身智能機器人、先進封裝 CoWoS），結合代號正規化、Universe 白名單硬檢驗與 7 維度確定性評分模型：
- [Discovery Candidates](/api/v1/discovery/candidates): 查詢前沿技術瓶頸映射至台美股核心供應鏈之候選標的清單與 7 維多因子評分拆解
- [arXiv Research Papers](/api/v1/discovery/papers): 查詢前沿突破論文摘要、架構瓶頸標籤與關聯標的代碼
- [Industry Live Signals](/api/v1/discovery/signals): 查詢前沿事件快訊流與情緒評分
- [Discovery History Trends](/api/v1/discovery/history): 查詢候選數量、論文數量與評分時序軌跡
- [Discovery Pipeline Status](/api/v1/discovery/status): 取得管線健康度、快照基準日與 Stale 降級標記

## Model Context Protocol (MCP) Tools

本平台提供 20 大標準 FastMCP 工具函數（支援 stdio 與 SSE 雙向通訊）：
- `get_market_macro_regime`: 評估大盤宏觀風控情境與建議投資曝險比例。
- `get_triple_resonance_stocks`: 查詢 👑四重共振、🏆三重共振與 🔮雙模型共振焦點股。
- `get_timestral_top_predictions`: 查詢 Mistral Timestral-209M 時序大模型 5 日漲跌幅排行與蒙地卡羅區間。
- `get_timesfm_top_predictions`: 查詢 Google TimesFM 時序大模型 5 日漲跌幅排行與盈虧比。
- `get_xuantie_pullback_stocks`: 查詢玄鐵重劍 MA60/120 趨勢回調買點標的。
- `get_stock_history`: 查詢特定股票歷史時序量化預測軌跡與法人籌碼。
- `get_latest_market_snapshot`: 取得最新量化日報批次數據快照。
- `get_top_institutional_flows`: 查詢三大法人買賣超焦點股排行榜。
- `get_broker_trades_for_stock`: 查詢券商關鍵主力分點進出明細。
- `get_company_profile`: 透過 2MD 查詢個股繁體中文公司簡介與即時新聞，並整合 Taiwan Company Network 官方資料。
- `get_company_network`: 查詢台股企業之經濟部官方統編、資本額、董事長、上游控股、下游轉投資企業與互動式股權網絡圖。
- `get_fed_rate_monitor`: 透過 2MD 查詢 CME FedWatch 聯準會利率決策機率。
- `get_us_earnings_calendar`: 透過 2MD 查詢美股重量級企業財報行事曆。
- `get_economic_calendar`: 透過 2MD 查詢全球重大總經行事曆。
- `get_commodities_summary`: 透過 2MD 查詢黃金、原油、銅行情。
- `resolve_stock_ticker`: 將模糊搜尋之公司名稱快速解析為標準交易代號。
- `get_polymarket_macro_sentiment`: 透過 2MD/DoH 查詢 Polymarket 真金白銀預測市場之宏觀風控與重大事件情緒。
- `get_clef_stock_verdict`: 透過 Clef-Flash System One 決策大模型（3 級 Fallback）推論交易動作機率分佈、確信度評分與勝率預期。
- `get_upcoming_earnings_calls`: 查詢跨市場 (台/美/日/韓/全球) 近期即將召開之法人說明會、雙時區對齊、倒數天數、簡報 PDF 與影音回放。
- `get_ticker_earnings_calls`: 查詢指定個股歷史與即將召開之法說會歷程、官方財測指引 (營收/毛利率/資本支出) 與簡報下載。
- `open_company_network`: 在瀏覽器中導航開啟特定台股企業之互動式集團股權拓撲關係圖譜 (Taiwan Company Network)。
- `get_company_tenders`: 透過 ToBid 查詢台股企業之政府電子採購網得標案件統計、累計金額、主要機關金主與競標對手情報。

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

> ⚠️ **Continuous Feature Evolution Active**: Caller LLM Agents should dynamically fetch the latest live specifications from `https://stockdata.david888.com/llms.txt` and `https://stockdata.david888.com/skill` before execution to leverage newly released models, tools, and endpoints.

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
3. **時序基礎大模型 - Mistral Timestral-209M (Timestral 5-Day Forecast & Monte Carlo Bands)**:
   - 採用 Mistral AI 209M 參數架構開源時序基礎大模型。
   - 輸入過去 96+ 交易日高精度 OHLCV 價格序列，自回歸產生未來 5 日目標價路徑。
   - 結合蒙地卡羅抽樣輸出 P10 (下行防守位)、P50 (中位預期) 與 P90 (樂觀目標位)。
4. **時序基礎大模型 - Google TimesFM (TimesFM 5-Day Forecast & Quantiles)**:
   - 採用 Google Research 預訓練 Decoder 架構時序基礎大模型 (TimesFM 2.0 200M / 2.5 500M)。
   - 輸出 1~5 日預測目標價軌跡，並以設定 horizon 的最後一天計算 P10 (下行防守位)、P50 (中位預期) 與 P90 (上行參考位)。
   - 真實盈虧比 (Risk/Reward Ratio): $RR = \\frac{P_{50,h} - P_0}{P_0 - P_{10,h}}$。當 $RR \\ge 1.5$ 且 horizon P50 $> P_0$ 時判定為高勝率看漲標的；次日潛力僅供診斷。
5. **籌碼策略 - 三大法人鎖碼 (TWSE Institutional Accumulation)**:
   - 分析外資、投信、自營商買賣超。篩選「投信連買 >= 3 天」或「土洋合買（外資與投信同向買超）」主力標的。
6. **產業板塊資金輪動 (Sector Rotation)**:
   - 追蹤 7 大核心板塊 10D (40%) + 15D (30%) + 20D (30%) 動量加權資金流，鎖定市場前 3 大主流板塊。
7. **多維交集共振架構 (Multi-Model Resonance Hierarchy)**:
   - 👑 **四重共振**: 玄鐵技術買點 ∩ 法人鎖碼 ∩ TimesFM 看漲 ∩ Timestral 看漲 (系統最高勝率極選)。
   - 🏆 **三重共振**: 玄鐵技術買點 ∩ 法人鎖碼 ∩ (TimesFM ∪ Timestral) 看漲 ∩ 估值合理 (PE<25 / PB<3.5)。
   - 🔮 **雙模型共振**: TimesFM ∩ Timestral 同步看漲，波形共識目標價 $P_{res} = 0.5 \\times P_{TimesFM} + 0.5 \\times P_{Timestral}$。
   - 🗡️ **玄鐵均線波段**: 「順大勢、逆小勢」，MA60/120 多頭排列下精準回踩支撐之波段買點。
   - ⚖️ **高盈虧比**: 時序大模型盈虧比 $RR \\ge 1.5$。

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
- `earnings_calls`:
  `id` (VARCHAR), `market` (VARCHAR: TW/US/JP/KR/OTHER), `ticker` (VARCHAR), `name` (VARCHAR), `call_date` (VARCHAR: YYYY-MM-DD), `call_time` (VARCHAR), `timezone` (VARCHAR), `meeting_type` (VARCHAR), `location` (VARCHAR), `webcast_url` (VARCHAR), `presentation_url` (VARCHAR), `transcript_url` (VARCHAR), `language` (VARCHAR), `eps_estimate` (DOUBLE), `eps_actual` (DOUBLE), `revenue_estimate` (DOUBLE), `revenue_actual` (DOUBLE), `is_weight_stock` (BOOLEAN), `financial_period` (VARCHAR), `guidance_summary` (VARCHAR), `guidance_outlook` (VARCHAR), `guidance_margin` (VARCHAR), `guidance_capex` (VARCHAR), `guidance_qa` (VARCHAR), `source` (VARCHAR)
- `macro_snapshots`:
  `category` (VARCHAR: PRIMARY KEY, 如 fed_rate_monitor, investing_summary, commodities_summary, economic_calendar, polymarket_macro_sentiment), `data_json` (VARCHAR: JSON 結構), `source` (VARCHAR), `updated_at` (VARCHAR: YYYY-MM-DD HH:MM:SS)
- `discovery_candidates`:
  `ticker`, `market`, `name`, `theme_id`, `theme_name`, `total_score` (DOUBLE), `recommendation` (VARCHAR), `review_status`, `in_pool` (BOOLEAN), 7-factor breakdown scores, `catalyst_summary`, `us_anchor`, `run_id`, `as_of`
- `discovery_papers`:
  `paper_id`, `title`, `authors`, `abstract`, `category`, `published_date`, `bottleneck_tag`, `related_tickers`, `url`, `significance_score`, `as_of`
- `discovery_signals`:
  `signal_id`, `ticker`, `theme_id`, `source_type`, `title`, `content`, `url`, `sentiment_score`, `detected_at`, `as_of`
- `discovery_history`:
  `run_id`, `as_of`, `total_candidates`, `propose_add_count`, `observe_count`, `already_in_pool_count`, `paper_count`, `signal_count`, `avg_total_score`

## 4. REST API Endpoints Specification

### 4.1 Quantitative Strategies
- `GET /api/v1/predictions/resonance?index_name=台灣50&limit=30`: 查詢雙時序大模型共振與多策略焦點推薦清單。
- `GET /api/v1/predictions/timestral/top-bullish?index_name=...&limit=20`: Mistral Timestral 5日看漲榜與蒙地卡羅區間。
- `GET /api/v1/predictions/timestral/top-bearish?index_name=...&limit=20`: Mistral Timestral 5日看跌避險榜。
- `GET /api/v1/predictions/timesfm/top-bullish?index_name=...&limit=20`: Google TimesFM 5日看漲榜與盈虧比。
- `GET /api/v1/predictions/timesfm/top-bearish?index_name=...&limit=20`: Google TimesFM 5日看跌避險榜。
- `GET /api/v1/predictions/xuantie?index_name=...&limit=20`: 玄鐵重劍 MA60/120 波段買點。
- `GET /api/v1/predictions/latest?index_name=...&limit=50`: 最新日報完整批次快照。
- `GET /api/v1/predictions/history/{ticker}?limit=30`: 單一標的歷史預測軌跡與籌碼時序。
- `GET /api/v1/predictions/resolve/{query}`: 模糊匹配股票名稱並解析為標準代號。

### 4.2 Macro & Investing Catalysts
- `GET /api/v1/macro/latest?market=tw|us`: 大盤風控與建議曝險比例。
- `GET /api/v1/macro/investing/summary`: 整合 CME FedWatch、美股財報、大宗商品與財經日曆。
- `GET /api/v1/macro/investing/fed-rate`: 聯準會利率決策機率分布與 FOMC 倒數。
- `GET /api/v1/macro/investing/earnings-calendar`: 美股重量級企業財報公布行事曆。
- `GET /api/v1/macro/investing/commodities`: 黃金、銅、WTI 原油行情與週期漲跌。
- `GET /api/v1/macro/investing/economic-calendar`: 全球重磅總經行事曆（CPI、非農等）。
- `GET /api/v1/macro/polymarket/sentiment`: Polymarket 真金白銀預測市場宏觀情緒（FOMC 利率、地緣關稅、科技巨頭、經濟衰退）；機率為 0~100 百分比，並回傳 `success`/`stale`/`error` 狀態。

### 4.3 Institutional & Company Profile
- `GET /api/v1/market/institutional/top?order_by=total_net&limit=30`: 三大法人買賣超排行。
- `GET /api/v1/market/broker/summary/{ticker}?days=20`: 券商關鍵分點主力累計買賣超。
- `GET /api/v1/market/company-profile?ticker=2330.TW`: 2MD 繁體中文公司簡介與即時新聞。
- `POST /api/v1/market/company-profiles/batch`: 並發批次預載多支股票之公司營運摘要。

### 4.4 Global Corporate Earnings Calls & Guidance (法說會與官方指引)
- `GET /api/v1/earnings-call/upcoming?market=ALL&days=14&is_weight_stock=true&limit=50`: 查詢近期跨市場法說會，支援 market (ALL/TW/US/JP/KR/OTHER)、雙時區換算與倒數天數。
- `GET /api/v1/earnings-call/calendar?year=2026&month=10&market=ALL`: 月度法說會日程聚合統計與每日公司摘要。
- `GET /api/v1/earnings-call/{ticker}?limit=20`: 個股歷史法說會歷程、官方財測指引 (營收/毛利率/資本支出) 與簡報 PDF 原始連結。
- `POST /api/v1/earnings-call/sync`: 觸發背景非同步多視窗跨市場全量法說會同步。

## 5. Model Context Protocol (MCP) Server
- FastMCP Server: `mcp_server.py`
- WebMCP Endpoint: `/mcp/sse`
- WebMCP Manifest: `/.well-known/mcp.json`
- Tools (22 Native Tools): `get_market_macro_regime`, `get_triple_resonance_stocks`, `get_timestral_top_predictions`, `get_timesfm_top_predictions`, `get_xuantie_pullback_stocks`, `get_stock_history`, `get_latest_market_snapshot`, `get_top_institutional_flows`, `get_broker_trades_for_stock`, `get_company_profile`, `get_company_network`, `get_fed_rate_monitor`, `get_us_earnings_calendar`, `get_economic_calendar`, `get_commodities_summary`, `resolve_stock_ticker`, `get_polymarket_macro_sentiment`, `get_clef_stock_verdict`, `get_upcoming_earnings_calls`, `get_ticker_earnings_calls`, `open_company_network`, `get_company_tenders`.

## 6. Recommended 4-Step Agent Trading Workflow
1. **檢查宏觀風控**: 呼叫 `get_market_macro_regime()` 決定整體建議曝險 (0%~100%)。
2. **首選共振焦點**: 呼叫 `get_triple_resonance_stocks()`，優先挑選含 `👑四重共振`、`🔮雙模型共振`、`高盈虧比` 標的。
3. **分流補充選股**:
   - 雙時序基礎模型: `get_timestral_top_predictions()` (Timestral 看漲) 與 `get_timesfm_top_predictions()` (TimesFM 盈虧比優勢)。
   - 波段回調買點: `get_xuantie_pullback_stocks()` (MA60/120 回踩買點)。
   - 避險防守觀察: `get_timestral_top_predictions(direction='bearish')`。
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
            description: '篩選三重共振強勢股（技術面 ∩ 雙大模型 ∩ 籌碼 ∩ 估值）',
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
            name: 'get_timestral_top_predictions',
            description: '取得 Mistral Timestral-209M 時序大模型 5 日預測漲跌幅排行與蒙地卡羅區間',
            inputSchema: { type: 'object', properties: { direction: { type: 'string', enum: ['bullish', 'bearish'] }, limit: { type: 'integer' }, index_name: { type: 'string' } }, required: ['direction'] },
            execute: async (args) => {
                const ep = args?.direction === 'bearish' ? 'top-bearish' : 'top-bullish';
                let url = '/api/v1/predictions/timestral/' + ep + '?limit=' + (args?.limit || 20);
                if (args?.index_name) url += '&index_name=' + encodeURIComponent(args.index_name);
                return JSON.stringify(await (await fetch(url)).json());
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
                if (args?.category && args.category !== 'all') url += '&category=' + encodeURIComponent(args.category);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_clef_stock_verdict',
            description: '透過 Clef-Flash System One 決策模型推論交易動作機率、確信度評分與勝率預期',
            inputSchema: {
                type: 'object',
                properties: {
                    ticker: { type: 'string', description: '股票代號 (如 2330.TW, NVDA)' },
                    macro_regime: { type: 'string', description: '宏觀情境 (bull, pullback, defense, panic)' },
                    api_key: { type: 'string', description: '決策 API 授權金鑰 (若伺服器設定存取控制時提供)' }
                },
                required: ['ticker']
            },
            execute: async (args) => {
                const headers = { 'Content-Type': 'application/json' };
                const token = (args && args.api_key) || (typeof window !== 'undefined' && (window.__DECISION_API_KEY__ || window.__API_KEY__ || localStorage.getItem('decision_api_key')));
                if (token) {
                    headers['X-API-Key'] = token;
                }
                return JSON.stringify(await (await fetch('/api/v1/decision/evaluate', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({ ticker: args.ticker, macro_regime: args.macro_regime || 'bull' })
                })).json());
            }
        },
        {
            name: 'get_upcoming_earnings_calls',
            description: '查詢近期跨市場 (台/美/日/韓/全球) 即將召開之法人說明會時程、雙時區與簡報 PDF',
            inputSchema: {
                type: 'object',
                properties: {
                    market: { type: 'string', description: '市場類別 (ALL, TW, US, JP, KR, OTHER)' },
                    days: { type: 'integer', description: '查詢未來天數範圍' },
                    is_weight_stock: { type: 'boolean', description: '是否僅顯示權值股' },
                    limit: { type: 'integer', description: '返回上限筆數' }
                }
            },
            execute: async (args) => {
                let url = '/api/v1/earnings-call/upcoming?market=' + encodeURIComponent(args?.market || 'ALL') + '&days=' + (args?.days || 14) + '&limit=' + (args?.limit || 50);
                if (typeof args?.is_weight_stock === 'boolean') url += '&is_weight_stock=' + args.is_weight_stock;
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_ticker_earnings_calls',
            description: '查詢個股 (如 2330.TW, NVDA, 7203.T) 歷史與未來法說會、官方財測指引與簡報下載',
            inputSchema: {
                type: 'object',
                properties: {
                    ticker: { type: 'string', description: '股票代號 (如 2330.TW, NVDA)' },
                    limit: { type: 'integer', description: '返回上限場次' }
                },
                required: ['ticker']
            },
            execute: async (args) => {
                let url = '/api/v1/earnings-call/' + encodeURIComponent(args.ticker) + '?limit=' + (args?.limit || 20);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'get_company_network',
            description: '查詢台股企業之經濟部統編、資本額、董事長、上游控股、下游轉投資企業與互動式股權網絡圖',
            inputSchema: {
                type: 'object',
                properties: {
                    ticker: { type: 'string', description: '股票代號或名稱 (如 2330.TW, 2317, 台積電)' }
                },
                required: ['ticker']
            },
            execute: async (args) => {
                let url = '/api/v1/market/company-network?ticker=' + encodeURIComponent(args.ticker);
                return JSON.stringify(await (await fetch(url)).json());
            }
        },
        {
            name: 'open_company_network',
            description: '在瀏覽器中開啟特定台股企業之互動式集團股權拓撲關係圖譜',
            inputSchema: {
                type: 'object',
                properties: {
                    company: { type: 'string', description: '公司正式全名、股票代號或別名 (如 2330, 台積電)' },
                    direction: { type: 'string', enum: ['both', 'upstream', 'downstream'], description: '關係展開方向，預設 both' }
                },
                required: ['company']
            },
            execute: async (args) => {
                let url = '/api/v1/market/company-network?ticker=' + encodeURIComponent(args.company);
                let res = await (await fetch(url)).json();
                let net = res?.data || res;
                let graphUrl = net?.web_url || ('https://taiwan-company-network.david888.com/graph?company=' + encodeURIComponent(args.company));
                if (typeof window !== 'undefined' && window.open) {
                    window.open(graphUrl, '_blank');
                }
                return JSON.stringify({ navigated: true, company: args.company, url: graphUrl, direction: args?.direction || 'both', message: '已在瀏覽器新分頁中開啟「' + args.company + '」之集團股權關係圖譜。' });
            }
        },
        {
            name: 'get_company_tenders',
            description: '查詢台股企業之政府採購得標統計、得標率、累計金額、主要機關金主與競標對手情報 (ToBid)',
            inputSchema: {
                type: 'object',
                properties: {
                    ticker: { type: 'string', description: '股票代號或名稱 (如 2634.TW, 漢翔, 華城)' }
                },
                required: ['ticker']
            },
            execute: async (args) => {
                let url = '/api/v1/market/company-tender?ticker=' + encodeURIComponent(args.ticker);
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
        "description_for_model": "888 Stock Quant 專業級深度學習與多維量化決策系統 (宏觀風控、玄鐵均線、Mistral Timestral-209M、Google TimesFM時序大模型、三大法人籌碼、👑四重共振、🏆三重共振、🔮雙模型共振、Polymarket 真金白銀預測市場)。提供每日台美股選股清單、目標價預測、法人籌碼鎖碼、均線波段買點與個股歷史走勢查詢。支援 16 大原生 MCP 量化工具與 WebMCP 協議。",
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
        version=APP_VERSION,
        duckdb_records=count
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8088"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
