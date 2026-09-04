"""
api/routes/macro.py - Global Macro Regime & Risk Exposure Endpoints
"""

from typing import Optional
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
    同時整合 Investing.com 之聯準會降息機率與重磅催化劑事件。
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
        fed_rate=state.fed_rate,
        commodities=state.commodities,
        earnings_calendar=state.earnings_calendar,
        economic_calendar=state.economic_calendar,
        catalyst_alerts=state.catalyst_alerts,
        timestamp=datetime.datetime.now().isoformat()
    )


@router.get("/investing/summary", summary="取得 Investing.com 宏觀數據與美股行事曆彙整")
def get_investing_summary(refresh: bool = False, force_refresh: bool = False):
    """取得聯準會利率機率、美股重量級財報、大宗原物料與重磅總經行事曆之彙整資料"""
    from data.investing_service import InvestingService
    do_refresh = refresh or force_refresh
    data = InvestingService.get_investing_macro_summary(force_refresh=do_refresh)
    return {"success": True, "data": data}


@router.get("/investing/fed-rate", summary="取得 CME 聯準會利率監控工具 (FedWatch) 降息機率")
def get_fed_rate(refresh: bool = False, force_refresh: bool = False):
    """取得下次 FOMC 會議時間與各目標利率區間機率分布"""
    from data.investing_service import InvestingService
    do_refresh = refresh or force_refresh
    data = InvestingService.get_fed_rate_monitor(force_refresh=do_refresh)
    return {"success": True, "data": data}


@router.get("/investing/earnings-calendar", summary="取得美股重要企業財報行事曆")
def get_earnings_calendar(refresh: bool = False, force_refresh: bool = False):
    """取得近期即將公佈財報的美股企業、預估 EPS、預估營收與市值"""
    from data.investing_service import InvestingService
    do_refresh = refresh or force_refresh
    data = InvestingService.get_earnings_calendar(force_refresh=do_refresh)
    return {"success": True, "data": data}


@router.get("/investing/economic-calendar", summary="取得重磅總體經濟數據行事曆")
def get_economic_calendar(refresh: bool = False, force_refresh: bool = False):
    """取得美國與全球重要總經指標公佈行事曆 (CPI, PCE, 非農, FOMC, GDP)"""
    from data.investing_service import InvestingService
    do_refresh = refresh or force_refresh
    data = InvestingService.get_economic_calendar(force_refresh=do_refresh)
    return {"success": True, "data": data}


@router.get("/investing/commodities", summary="取得關鍵大宗商品與原物料行情")
def get_commodities(refresh: bool = False, force_refresh: bool = False):
    """取得黃金、銅博士、WTI 原油之即時行情與週期漲跌"""
    from data.investing_service import InvestingService
    do_refresh = refresh or force_refresh
    data = InvestingService.get_commodities_summary(force_refresh=do_refresh)
    return {"success": True, "data": data}


