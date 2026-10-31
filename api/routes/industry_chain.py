"""
api/routes/industry_chain.py - 跨市場 AI 產業價值鏈與實體關係圖譜 REST API

提供產業主題清單、三欄拓撲結構 (上游 ➔ 中游 ➔ 下游) 與 DuckDB 即時量化預測/籌碼動態融合。
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Depends

from api.schemas import (
    IndustryThemesResponse,
    IndustryThemeItem,
    IndustryGraphResponse,
    IndustryGraphNode,
    IndustryGraphEdge,
)
from data.industry_chain_fetcher import IndustryChainFetcher
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.api.industry_chain")

router = APIRouter(prefix="/industry-chain", tags=["Industry Value Chain & Graph"])

_fetcher = IndustryChainFetcher()


def get_duckdb() -> DuckDBManager:
    return DuckDBManager()


@router.get(
    "/themes",
    response_model=IndustryThemesResponse,
    summary="取得可用的 AI 產業價值鏈主題清單與統計數量"
)
def get_industry_themes():
    """
    回傳所有支援的產業價值鏈主題 (如 ALL, HBM_STORAGE, CPO_OPTICAL, POWER_COOLING, 先進封裝等)
    包含各主題的上中下游標的數量統計。
    """
    themes = _fetcher.get_available_themes()
    return IndustryThemesResponse(
        success=True,
        total_themes=len(themes),
        data=themes
    )


@router.get(
    "/graph",
    response_model=IndustryGraphResponse,
    summary="取得指定主題之產業價值鏈三欄拓撲圖譜 (動態融合量化預測與籌碼)"
)
def get_industry_graph(
    theme: str = Query("ALL", description="產業主題代號 (如 ALL, HBM_STORAGE, CPO_OPTICAL, POWER_COOLING, ADV_PACKAGING, AI_CHIPS, SERVER_ODM, ROBOTICS)"),
    relation_type: str = Query("ALL", description="關係篩選 (ALL, 產業鏈, 資本開支, 大客戶依賴, 主題關聯)"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢產業鏈結構圖譜，依據上中下游三欄分組，
    並自動自 DuckDB 時序資料庫注入最新收盤價、漲跌幅、Timestral/TimesFM 隔日目標價與投信連買籌碼標籤。
    """
    graph_data = _fetcher.get_theme_graph(theme=theme, relation_filter=relation_type, duckdb_mgr=db)

    # 收集圖譜內所有股票代號進行批次量化行情與籌碼補強
    all_raw_nodes = (
        graph_data["upstream_nodes"] +
        graph_data["midstream_nodes"] +
        graph_data["downstream_nodes"]
    )
    tickers = [n["ticker"] for n in all_raw_nodes if n.get("ticker")]

    enriched_map = {}
    if getattr(db, "enabled", False) and tickers:
        try:
            enriched_map = db.get_enriched_industry_quotes(tickers)
        except Exception as e:
            logger.warning("動態注入口量化指標失敗: %s", e)

    def _enrich_node(raw_node: Dict[str, Any]) -> IndustryGraphNode:
        t = raw_node["ticker"]
        t_clean = t.split(".")[0] if "." in t else t
        quote_info = enriched_map.get(t_clean, {}) or enriched_map.get(t, {})

        return IndustryGraphNode(
            ticker=raw_node["ticker"],
            company=raw_node["company"],
            market=raw_node["market"],
            layer=raw_node["layer"],
            role=raw_node["role"],
            key_focus=raw_node.get("key_focus"),
            theme_tags=raw_node.get("theme_tags", []),
            current_price=quote_info.get("current_price"),
            change_pct=quote_info.get("change_pct"),
            timestral_potential=quote_info.get("timestral_potential"),
            timesfm_potential=quote_info.get("timesfm_potential"),
            institutional_streak=quote_info.get("institutional_streak"),
            dual_model_consensus=quote_info.get("dual_model_consensus"),
        )

    enriched_upstream = [_enrich_node(n) for n in graph_data["upstream_nodes"]]
    enriched_midstream = [_enrich_node(n) for n in graph_data["midstream_nodes"]]
    enriched_downstream = [_enrich_node(n) for n in graph_data["downstream_nodes"]]

    edges = [
        IndustryGraphEdge(
            source=e["source"],
            target=e["target"],
            relation_type=e["relation_type"],
            desc=e.get("desc"),
            weight=e.get("weight", 1.0)
        )
        for e in graph_data["edges"]
    ]

    return IndustryGraphResponse(
        success=True,
        theme=graph_data["theme"],
        relation_filter=graph_data["relation_filter"],
        total_nodes=len(enriched_upstream) + len(enriched_midstream) + len(enriched_downstream),
        upstream_nodes=enriched_upstream,
        midstream_nodes=enriched_midstream,
        downstream_nodes=enriched_downstream,
        edges=edges,
        summary=graph_data.get("summary", {})
    )
