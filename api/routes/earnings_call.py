"""
api/routes/earnings_call.py - 法人說明會與財報催化劑 REST API 路由
"""

import datetime
import re
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Path, HTTPException, BackgroundTasks

from api.schemas import EarningsCallItem, UpcomingCallsResponse, EarningsCalendarResponse
from data.duckdb_manager import DuckDBManager
from data.earnings_fetcher import EarningsCallFetcher

router = APIRouter(prefix="/earnings-call", tags=["Earnings Call 法人說明會"])

duckdb_manager = DuckDBManager()


def _clean_str(val: Any) -> Optional[str]:
    """將資料庫/Pandas回傳值安全轉為字串，自動清除 NaN、None 與空字串"""
    if val is None:
        return None
    if isinstance(val, float) and (val != val):
        return None
    s = str(val).strip()
    return s if s else None


def _clean_float(val: Any) -> Optional[float]:
    """將資料庫/Pandas回傳值安全轉為浮點數，自動清除 NaN 與非數值"""
    if val is None:
        return None
    if isinstance(val, float):
        return None if (val != val) else val
    try:
        f = float(val)
        return None if (f != f) else f
    except (ValueError, TypeError):
        return None


def _enrich_call_item(row: Dict[str, Any], today: datetime.date) -> EarningsCallItem:
    """轉換資料庫字典為附帶雙時區與倒數之 Pydantic 物件"""
    call_date_str = str(row.get("call_date", ""))
    call_time_str = str(row.get("call_time", "14:00") or "14:00")
    timezone_str = str(row.get("timezone", "Asia/Taipei"))

    # 計算倒數天數
    countdown = None
    status = "upcoming"
    try:
        parts = [int(p) for p in call_date_str.split("-")]
        c_date = datetime.date(parts[0], parts[1], parts[2])
        diff = (c_date - today).days
        countdown = diff
        if diff == 0:
            status = "today"
        elif diff < 0:
            status = "past"
        else:
            status = "upcoming"
    except Exception:
        pass

    # 產生雙時區表示 (精確利用 zoneinfo 處理夏令/冬令時差)
    datetime_cst = None
    datetime_est = None
    try:
        from zoneinfo import ZoneInfo
        time_part = call_time_str.strip() if call_time_str else "14:00"
        if len(time_part) == 5:
            time_part += ":00"
        elif len(time_part) == 4 and ":" not in time_part:
            time_part = f"{time_part[:2]}:{time_part[2:]}:00"

        dt_naive = datetime.datetime.strptime(f"{call_date_str} {time_part[:8]}", "%Y-%m-%d %H:%M:%S")
        if timezone_str == "America/New_York":
            dt_ny = dt_naive.replace(tzinfo=ZoneInfo("America/New_York"))
            dt_cst_obj = dt_ny.astimezone(ZoneInfo("Asia/Taipei"))
            datetime_est = dt_ny.strftime("%Y-%m-%d %H:%M %Z")
            datetime_cst = dt_cst_obj.strftime("%Y-%m-%d %H:%M CST")
        else:
            dt_tw = dt_naive.replace(tzinfo=ZoneInfo("Asia/Taipei"))
            dt_ny_obj = dt_tw.astimezone(ZoneInfo("America/New_York"))
            datetime_cst = dt_tw.strftime("%Y-%m-%d %H:%M CST")
            datetime_est = dt_ny_obj.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        datetime_cst = f"{call_date_str} {call_time_str} CST"
        datetime_est = f"{call_date_str} {call_time_str} EST"

    return EarningsCallItem(
        id=str(row.get("id", "")),
        market=str(row.get("market", "TW")),
        ticker=str(row.get("ticker", "")),
        name=str(row.get("name", "")),
        call_date=call_date_str,
        call_time=call_time_str,
        timezone=timezone_str,
        datetime_cst=datetime_cst,
        datetime_est=datetime_est,
        countdown_days=countdown,
        status=status,
        meeting_type=row.get("meeting_type") or "online",
        location=_clean_str(row.get("location")),
        webcast_url=_clean_str(row.get("webcast_url")),
        presentation_url=_clean_str(row.get("presentation_url")),
        transcript_url=_clean_str(row.get("transcript_url")),
        language=_clean_str(row.get("language")) or "zh-TW",
        eps_estimate=_clean_float(row.get("eps_estimate")),
        eps_actual=_clean_float(row.get("eps_actual")),
        revenue_estimate=_clean_float(row.get("revenue_estimate")),
        revenue_actual=_clean_float(row.get("revenue_actual")),
        is_weight_stock=bool(row.get("is_weight_stock", False)),
        remarks=_clean_str(row.get("remarks")),
        financial_period=_clean_str(row.get("financial_period")),
        guidance_summary=_clean_str(row.get("guidance_summary")),
        guidance_outlook=_clean_str(row.get("guidance_outlook")),
        guidance_margin=_clean_str(row.get("guidance_margin")),
        guidance_capex=_clean_str(row.get("guidance_capex")),
        guidance_qa=_clean_str(row.get("guidance_qa")),
        source=_clean_str(row.get("source")) or "BigGo / MOPS"
    )


@router.get("/upcoming", response_model=UpcomingCallsResponse, summary="取得全球跨市場 (台/美/日/韓) 即將舉行的法說會清單")
def get_upcoming_calls(
    market: str = Query("ALL", description="市場篩選: ALL / TW / US / JP / KR / OTHER"),
    days: int = Query(14, ge=1, le=90, description="查詢未來天數範圍 (預設 14 天)"),
    start_date: Optional[str] = Query(None, description="起始日期 (YYYY-MM-DD，若指定則優先於 days)"),
    end_date: Optional[str] = Query(None, description="截止日期 (YYYY-MM-DD)"),
    is_weight_stock: Optional[bool] = Query(None, description="是否僅顯示重量級權值股"),
    limit: int = Query(50, ge=1, le=200, description="返回筆數上限")
):
    """
    查詢近期即將召開之法人說明會，自動換算雙時區與倒數剩餘天數。
    """
    mkt = str(market) if isinstance(market, str) else "ALL"
    s_date = str(start_date) if isinstance(start_date, str) and start_date else None
    e_date = str(end_date) if isinstance(end_date, str) and end_date else None
    w_stock = is_weight_stock if isinstance(is_weight_stock, bool) else None
    day_cnt = int(days) if isinstance(days, int) else 14
    lim = int(limit) if isinstance(limit, int) else 50

    today = datetime.date.today()
    s_date = s_date or (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    e_date = e_date or (today + datetime.timedelta(days=day_cnt)).strftime("%Y-%m-%d")

    records = duckdb_manager.get_earnings_calls(
        market=mkt,
        start_date=s_date,
        end_date=e_date,
        is_weight_stock=w_stock,
        limit=lim
    )

    # 若特定市場或冷啟動時查無紀錄，即時觸發一次同步以獲取最新法說會
    if not records and (start_date is None or not isinstance(start_date, str)):
        fetcher = EarningsCallFetcher(duckdb_manager)
        fetcher.fetch_all_and_sync()
        records = duckdb_manager.get_earnings_calls(
            market=mkt,
            start_date=s_date,
            end_date=e_date,
            is_weight_stock=w_stock,
            limit=lim
        )

    # 若在預設滑動視窗內該市場暫無法說會排程 (例如日韓或低頻市場)，回溯抓取最近召開之歷史法說會，避免清單空白
    if not records and (start_date is None or not isinstance(start_date, str)):
        records = duckdb_manager.get_earnings_calls(
            market=mkt,
            is_weight_stock=w_stock,
            order_desc=True,
            limit=lim
        )

    enriched = [_enrich_call_item(r, today) for r in records]

    # 排序：優先呈現今天 (today) 與即將召開 (upcoming，依倒數天數升序)，隨後呈現近期已召開 (past，依日期降序)
    def _sort_key(call: EarningsCallItem):
        prio = 0 if call.status == "today" else (1 if call.status == "upcoming" else 2)
        countdown = call.countdown_days if (call.countdown_days is not None and call.countdown_days >= 0) else 999
        date_num = int(call.call_date.replace("-", "")) if (call.call_date and re.match(r"^\d{4}-\d{2}-\d{2}$", call.call_date)) else 0
        return (prio, countdown, -date_num)

    enriched.sort(key=_sort_key)

    return UpcomingCallsResponse(
        success=True,
        market=mkt.upper(),
        total_count=len(enriched),
        data=enriched
    )


@router.get("/calendar", response_model=EarningsCalendarResponse, summary="取得月度法說會日曆聚合場次")
def get_earnings_calendar(
    year: Optional[int] = Query(None, description="年份 (預設當前年)"),
    month: Optional[int] = Query(None, ge=1, le=12, description="月份 (1-12，預設當前月)"),
    market: str = Query("ALL", description="市場: ALL / TW / US / JP / KR / OTHER")
):
    """
    查詢指定月之每日法說會場次計數，供行事曆檢視視覺化。
    """
    today = datetime.date.today()
    y = year if isinstance(year, int) else today.year
    m = month if isinstance(month, int) else today.month
    mkt = str(market) if isinstance(market, str) else "ALL"

    data = duckdb_manager.get_earnings_calendar(year=y, month=m, market=mkt)
    return EarningsCalendarResponse(
        success=True,
        year=y,
        month=m,
        market=mkt.upper(),
        data=data
    )


@router.get("/{ticker}", response_model=UpcomingCallsResponse, summary="取得單一標的之法說會歷程與未來時程")
def get_ticker_calls(
    ticker: str = Path(..., description="股票代碼 (如 2330.TW 或 NVDA)"),
    limit: int = Query(20, ge=1, le=100, description="最多返回場次")
):
    """
    查詢特定個股之歷史與預約法說會資訊，包含歷次官方簡報 PDF 下載入口與影音回放。
    """
    lim = int(limit) if isinstance(limit, int) else 20
    today = datetime.date.today()
    records = duckdb_manager.get_earnings_calls(
        ticker=ticker,
        limit=lim
    )
    if not records:
        fetcher = EarningsCallFetcher(duckdb_manager)
        records = fetcher.fetch_ticker_calls(ticker)
        if records:
            records = records[:lim]

    enriched = [_enrich_call_item(r, today) for r in records]
    return UpcomingCallsResponse(
        success=True,
        market="TICKER",
        total_count=len(enriched),
        data=enriched
    )


@router.post("/sync", summary="觸發跨市場法說會資料全量同步")
def trigger_sync(background_tasks: BackgroundTasks):
    """
    手動或排程觸發法說會資訊同步 (TWSE/TPEx OpenAPI + 美股行事曆)。
    """
    fetcher = EarningsCallFetcher(duckdb_manager)
    background_tasks.add_task(fetcher.fetch_all_and_sync)
    return {
        "success": True,
        "message": "法說會資料同步已於背景排隊執行",
        "timestamp": datetime.datetime.now().isoformat()
    }
