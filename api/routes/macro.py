"""
api/routes/macro.py - Global Macro Regime & Risk Exposure Endpoints
"""

from fastapi import APIRouter, Depends
from api.schemas import MacroRegimeResponse
from data.macro import MacroRegimeAnalyzer
from data.duckdb_manager import DuckDBManager
import datetime

router = APIRouter(prefix="/macro", tags=["Macro Regime & Risk Gate"])


@router.get("/latest", response_model=MacroRegimeResponse, summary="取得當前美股宏觀風控狀態與建議曝險")
def get_latest_macro():
    """
    即時評估美股三大核心指標（S&P 500、VIX 恐慌指數、SOX 費城半導體），回傳當前全球環境與建議倉位比例。
    """
    analyzer = MacroRegimeAnalyzer()
    state = analyzer.evaluate()
    
    return MacroRegimeResponse(
        regime_name=state.regime_name,
        exposure=state.exposure,
        vix=state.vix_value,
        spy_above_ma60=state.spy_above_ma60,
        sox_above_ma60=state.sox_above_ma60,
        warnings=state.warnings,
        timestamp=datetime.datetime.now().isoformat()
    )
