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
    description="專業級 AI 深度學習與多維量化決策平台 (宏觀風控、玄鐵均線、LSTM預測、三大法人籌碼、🏆三重共振)"
)

# Global DuckDB Manager
db = DuckDBManager()


@mcp.tool(
    name="get_market_macro_regime",
    description="即時評估全球與美股宏觀市場風控狀態（監控 S&P 500、VIX 恐慌指數、SOX 費城半導體），取得當前市場情境（全面多頭/多頭回調/避險防禦/極度恐慌）與建議投資曝險比例（0.0 ~ 1.0）。"
)
def get_market_macro_regime() -> Dict[str, Any]:
    """
    Query current global macroeconomic risk status and exposure level.
    """
    state = MacroRegimeAnalyzer.evaluate_us_market()
    return {
        "success": True,
        "regime_name": state.regime_name,
        "exposure": state.exposure,
        "vix": state.vix,
        "spy_above_ma60": state.spy_above_ma60,
        "sox_above_ma60": state.sox_above_ma60,
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
    records = db.get_ticker_history(ticker=ticker.strip(), limit=limit)
    return {
        "success": True,
        "ticker": ticker.strip(),
        "count": len(records),
        "data": records
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


if __name__ == "__main__":
    # Standard FastMCP entrypoint
    mcp.run()
