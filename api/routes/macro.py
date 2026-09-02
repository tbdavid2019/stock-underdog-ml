"""
api/routes/macro.py - Global Macro Regime & Risk Exposure Endpoints
"""

from fastapi import APIRouter, Depends
from api.schemas import MacroRegimeResponse
from data.macro import MacroRegimeAnalyzer
from data.duckdb_manager import DuckDBManager
import datetime

router = APIRouter(prefix="/macro", tags=["Macro Regime & Risk Gate"])


@router.get("/latest", response_model=MacroRegimeResponse, summary="取得當前大盤/宏觀風控狀態與建議曝險")
def get_latest_macro(market: str = "us", index_name: Optional[str] = None):
    """
    即時評估台股或美股大盤核心指標，回傳當前市場情境與建議倉位比例。
    支援 market=tw (加權指數/費半連動) 或 market=us (S&P500/VIX/費半)。
    """
    target = index_name or market
    state = MacroRegimeAnalyzer.evaluate_market(target)
    
    return MacroRegimeResponse(
        market=state.market,
        regime_name=state.regime_name,
        exposure=state.exposure,
        vix=state.vix,
        spy_above_ma60=state.spy_above_ma60,
        sox_above_ma60=state.sox_above_ma60,
        twii_above_ma60=state.twii_above_ma60,
        warnings=state.warnings,
        timestamp=datetime.datetime.now().isoformat()
    )
