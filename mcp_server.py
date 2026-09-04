#!/usr/bin/env python3
"""
mcp_server.py - Model Context Protocol (MCP) Server for Stock Quant Platform

Provides native AI Agent Tools for Claude Desktop, Cursor, Antigravity, Open-WebUI, etc.
Connects directly to the embedded DuckDB high-speed quantitative time-series database.
"""

import os
import sys
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from data.duckdb_manager import DuckDBManager
from data.macro import MacroRegimeAnalyzer

# Initialize FastMCP Server with name and instructions
mcp = FastMCP(
    "stock-quant-engine",
    instructions="888 Stock Quant - 專業級深度學習與多維量化決策平台 (宏觀風控、玄鐵均線、LSTM預測、三大法人籌碼、🏆三重共振)"
)

# Global DuckDB Manager
db = DuckDBManager()


@mcp.tool(
    name="get_market_macro_regime",
    description="即時評估台股大盤或美股宏觀市場風控狀態（台股加權指數、S&P 500、VIX 恐慌指數、SOX 費城半導體），取得當前市場情境（全面多頭/多頭回調/避險防禦/極度恐慌）與建議投資曝險比例（0.0 ~ 1.0）。"
)
def get_market_macro_regime(
    market: str = "us",
    index_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query current macroeconomic and market regime status and exposure level for TW or US market.
    """
    target = index_name or market
    state = MacroRegimeAnalyzer.evaluate_market(target)
    return {
        "success": True,
        "market": state.market,
        "regime_name": state.regime_name,
        "exposure": state.exposure,
        "vix": state.vix,
        "spy_above_ma60": state.spy_above_ma60,
        "sox_above_ma60": state.sox_above_ma60,
        "twii_above_ma60": state.twii_above_ma60,
        "tech_exposure_cap": state.tech_exposure_cap,
        "warnings": state.warnings,
        "summary": state.summary
    }


@mcp.tool(
    name="get_triple_resonance_stocks",
    description="查詢多策略交集與 🏆 三重共振焦點股票。篩選同時符合「技術面 MA60/120 買點 ∩ LSTM 看漲 ∩ 投信/外資主力連續買超鎖碼 ∩ 低PE估值」之最高信心標的。"
)
def get_triple_resonance_stocks(
    index_name: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Query Triple Resonance and multi-strategy intersection stock picks.
    
    Args:
        index_name: 指數名稱篩選 (如 '台灣50', '台灣中型100', 'SP500', 或 None 代表全部)
        limit: 回傳標的數量上限 (預設: 20)
    """
    records = db.get_resonance_candidates(index_name=index_name, limit=limit)
    return {
        "success": True,
        "count": len(records),
        "data": records
    }


@mcp.tool(
    name="get_xuantie_pullback_stocks",
    description="查詢「玄鐵重劍」技術面均線回調波段買點標的。依據『順大勢、逆小勢』原則，篩選中長期均線多頭排列且現價剛好回踩 MA60 季線或 MA120 半年線之支撐個股。"
)
def get_xuantie_pullback_stocks(
    index_name: Optional[str] = None,
    pullback_type: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Query XuanTie trend pullback technical buy points (MA60/120 support).
    
    Args:
        index_name: 指數名稱篩選 (如 '台灣50', '台灣中型100', 'SP500')
        pullback_type: 均線回調類型篩選 (如 'MA60', 'MA120', 或 None)
        limit: 回傳數量上限 (預設: 20)
    """
    records = db.get_xuantie_candidates(index_name=index_name, pullback_type=pullback_type, limit=limit)
    return {
        "success": True,
        "count": len(records),
        "data": records
    }


@mcp.tool(
    name="get_lstm_top_predictions",
    description="查詢 LSTM 深度學習模型預測之漲幅 TOP N 潛力榜或跌幅 TOP N 避險做空榜。包含預測明日目標價、潛在漲跌幅 %、本益比 PE 與股價淨值比 PB。"
)
def get_lstm_top_predictions(
    index_name: Optional[str] = None,
    direction: str = "bullish",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Query LSTM price prediction rankings.
    
    Args:
        index_name: 指數名稱篩選 (如 '台灣50', '台灣中型100', 'SP500')
        direction: 'bullish' (看漲排行) 或 'bearish' (看跌排行)
        limit: 取得前 N 檔 (預設: 10)
    """
    if direction.lower() == "bearish":
        records = db.get_top_bearish(index_name=index_name, limit=limit)
    else:
        records = db.get_top_bullish(index_name=index_name, limit=limit)
        
    return {
        "success": True,
        "direction": direction,
        "count": len(records),
        "data": records
    }


@mcp.tool(
    name="get_stock_history",
    description="查詢特定股票代號（如 '2330.TW', '2454.TW', 'AAPL', 'NVDA'）的時間序列歷史量化預測軌跡、均線狀態與歷史信號。"
)
def get_stock_history(
    ticker: str,
    limit: int = 30
) -> Dict[str, Any]:
    """
    Query historical prediction trajectory and metrics for a specific ticker.
    
    Args:
        ticker: 股票代號 (如 '2330.TW', 'AAPL')
        limit: 歷史天數/筆數 (預設: 30)
    """
    requested_ticker = ticker.strip()
    resolved = db.resolve_ticker(requested_ticker)
    if not resolved:
        return {
            "success": False,
            "ticker": requested_ticker,
            "count": 0,
            "data": [],
            "error": f"查無法辨識的股票或公司名稱：{requested_ticker}"
        }

    records = db.get_ticker_history(ticker=resolved["ticker"], limit=limit)
    return {
        "success": bool(records),
        "query": requested_ticker,
        "ticker": resolved["ticker"],
        "name": resolved.get("name"),
        "count": len(records),
        "data": records,
        "error": None if records else f"查無標的 {resolved['ticker']} 之歷史預測記錄"
    }


@mcp.tool(
    name="get_latest_market_snapshot",
    description="取得最新量化日報批次數據。包含所有標的之現價、預測價、均線排列、本益比/股價淨值比、三大法人買賣超與綜合標籤。"
)
def get_latest_market_snapshot(
    index_name: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get latest full quantitative batch snapshot.
    
    Args:
        index_name: 指數名稱篩選 (如 '台灣50', '台灣中型100', 'SP500')
        limit: 回傳數量 (預設: 50)
    """
    res = db.get_latest_predictions(index_name=index_name, limit=limit, batch_only=True)
    return {
        "success": True,
        "batch_date": res.get("batch_date"),
        "latest_batch_timestamp": res.get("latest_batch_timestamp"),
        "is_stale": res.get("is_stale", False),
        "count": len(res.get("records", [])),
        "data": res.get("records", [])
    }


@mcp.tool(
    name="get_top_institutional_flows",
    description="查詢台股三大法人（外資、投信、自營商）淨買超或淨賣超排行榜與持股時序。"
)
def get_top_institutional_flows(
    order_by: str = "total_net",
    sort_dir: str = "desc",
    market: str = "ALL",
    limit: int = 20
) -> Dict[str, Any]:
    """
    Query top institutional net buyers/sellers from DuckDB.
    
    Args:
        order_by: 排序欄位 ('total_net', 'foreign_net', 'trust_net', 'dealer_net')
        sort_dir: 排序方向 ('desc' 買超榜, 'asc' 賣超榜)
        market: 市場篩選 ('ALL', 'TWSE', 'TPEX')
        limit: 回傳數量 (預設: 20)
    """
    valid_cols = {"total_net", "foreign_net", "trust_net", "dealer_net"}
    if order_by not in valid_cols:
        order_by = "total_net"
    sort_order = "ASC" if sort_dir.lower() == "asc" else "DESC"

    try:
        with db._get_connection() as con:
            max_date = con.execute("SELECT MAX(date) FROM tw_institutional_daily").fetchone()[0]
            if not max_date:
                return {"success": True, "count": 0, "data": []}
            
            where_clauses = ["date = ?"]
            params = [str(max_date)]
            if market.upper() in ["TWSE", "TPEX"]:
                where_clauses.append("market = ?")
                params.append(market.upper())
            
            where_sql = " AND ".join(where_clauses)
            sql = f"""
            SELECT date, ticker, raw_code, name, foreign_net, trust_net, dealer_net, total_net, foreign_ratio, market
            FROM tw_institutional_daily
            WHERE {where_sql}
            ORDER BY {order_by} {sort_order}
            LIMIT ?;
            """
            params.append(limit)
            df = con.execute(sql, params).df()
            records = df.to_dict(orient="records")
            return {
                "success": True,
                "date": str(max_date),
                "count": len(records),
                "data": records
            }
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@mcp.tool(
    name="get_broker_trades_for_stock",
    description="查詢特定股票（如 '2330.TW', '2454.TW'）在過去 N 天內，各關鍵券商分點（外資外商分點與內資主力分點）之累計買賣超排行榜與統計摘要。"
)
def get_broker_trades_for_stock(
    ticker: str,
    days: int = 20
) -> Dict[str, Any]:
    """
    Query broker branches top buyers and sellers for a ticker.
    
    Args:
        ticker: 股票代號 (例如 '2330.TW' 或 '2330')
        days: 統計天數 (預設: 20)
    """
    summary = db.get_broker_top_summary(ticker=ticker, limit_days=days)
    return {
        "success": True,
        "ticker": ticker,
        "days": days,
        "data": summary
    }


@mcp.tool(
    name="get_company_profile",
    description="透過 2md.aiurl.tw 與 Yahoo 股市實時查詢股票（如 9945.TW, 2330.TW, AAPL）之繁體中文公司簡介、核心營運業務、產業別、市值與最新即時新聞。"
)
def get_company_profile(
    ticker: str
) -> Dict[str, Any]:
    """
    Fetch company profile, business operations, and recent news using 2md.aiurl.tw.
    """
    from data.company_profile import CompanyProfileService
    profile = CompanyProfileService.get_company_profile(ticker, db=db)
    return {
        "success": True,
        "ticker": profile["ticker"],
        "name": profile["name"],
        "industry": profile.get("industry", "綜合產業"),
        "business_summary": profile.get("business_summary", ""),
        "chairman": profile.get("chairman", ""),
        "market_cap": profile.get("market_cap", ""),
        "recent_news": profile.get("recent_news", []),
        "links": profile.get("links", {})
    }



@mcp.tool(
    name="get_fed_rate_monitor",
    description="透過 2MD 查詢 Investing.com FedWatch 聯準會利率監控工具（CME FedWatch）。取得下次 FOMC 會議日期、倒數天數、降息/升息概率分布表（Target Rate, Current %, Prev Day %, Prev Week %）與政策預期摘要。"
)
def get_fed_rate_monitor(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetch CME FedWatch interest rate probabilities and FOMC countdown from Investing.com via 2MD.
    """
    from data.investing_service import InvestingService
    return InvestingService.get_fed_rate_monitor(force_refresh=force_refresh)


@mcp.tool(
    name="get_us_earnings_calendar",
    description="透過 2MD 查詢 Investing.com 美股重量級企業財報行事曆。取得近期即將公布財報之美股個股（代號、公司名稱、財報公布日、EPS 預估值、營收預估值、市值規模）。"
)
def get_us_earnings_calendar(force_refresh: bool = False, limit: int = 15) -> Dict[str, Any]:
    """
    Fetch upcoming US corporate earnings releases from Investing.com via 2MD.
    """
    from data.investing_service import InvestingService
    data = InvestingService.get_earnings_calendar(force_refresh=force_refresh)
    if "earnings" in data and isinstance(data["earnings"], list):
        data["earnings"] = data["earnings"][:limit]
    return data


@mcp.tool(
    name="get_economic_calendar",
    description="透過 2MD 查詢 Investing.com 全球重大財經行事曆（CPI、非農 NFP、PCE、GDP、FOMC 等）。取得時間、國家、重要度等級、前值、預測值與實際值。"
)
def get_economic_calendar(force_refresh: bool = False, limit: int = 15) -> Dict[str, Any]:
    """
    Fetch upcoming high-impact economic calendar events from Investing.com via 2MD.
    """
    from data.investing_service import InvestingService
    data = InvestingService.get_economic_calendar(force_refresh=force_refresh)
    if "events" in data and isinstance(data["events"], list):
        data["events"] = data["events"][:limit]
    return data


if __name__ == "__main__":
    # Standard FastMCP entrypoint
    mcp.run()

