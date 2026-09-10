"""
backtest/resolver.py - 回測交易日解析與時區/午夜邊界處理器
"""

import datetime
from typing import Optional, Tuple
import pandas as pd

try:
    import zoneinfo
    TZ_TAIPEI = zoneinfo.ZoneInfo("Asia/Taipei")
    TZ_NEW_YORK = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    TZ_TAIPEI = datetime.timezone(datetime.timedelta(hours=8))
    TZ_NEW_YORK = datetime.timezone(datetime.timedelta(hours=-4))  # EDT fallback


def is_taiwan_ticker(ticker: str) -> bool:
    """判斷標的是否屬於台股 (TWSE / TPEX)"""
    t = ticker.upper()
    return t.endswith(".TW") or t.endswith(".TWO") or (t.isdigit() and len(t) >= 4)


def parse_prediction_timestamp(ts_val: any) -> datetime.datetime:
    """
    規格化解析預測時間戳記，確保具備時區資訊 (預設 Asia/Taipei)
    """
    if isinstance(ts_val, datetime.datetime):
        dt = ts_val
    else:
        ts_clean = str(ts_val).replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(ts_clean)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_TAIPEI)
    return dt


def get_target_trading_date(pred_dt: datetime.datetime, ticker: str) -> str:
    """
    根據預測產生的時間與標的所屬市場，精確推算預測所針對的「目標交易日 (YYYY-MM-DD)」:
    
    1. 台股市場 (09:00 ~ 13:30 Asia/Taipei):
       - 13:30 以前產生 (含 08:00 盤前指南): 針對當日 (Today, Day T)
       - 13:30 以後產生 (盤後批次): 針對下一交易日 (Day T+1 起算)
       
    2. 美股市場 (09:30 ~ 16:00 America/New_York):
       - 轉換至 America/New_York 美東時間判定
       - 16:00 以前產生 (含台北 20:30 美股盤前、以及台北午夜 00:00~04:00 盤中):
         針對紐約當日 (Today in NY, Day T)
       - 16:00 以後產生 (美股收盤後): 針對下一交易日 (Day T+1 in NY 起算)
    """
    if is_taiwan_ticker(ticker):
        dt_local = pred_dt.astimezone(TZ_TAIPEI)
        # 台股 13:30 收盤
        if dt_local.hour < 13 or (dt_local.hour == 13 and dt_local.minute < 30):
            return dt_local.strftime("%Y-%m-%d")
        else:
            return (dt_local + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        dt_local = pred_dt.astimezone(TZ_NEW_YORK)
        # 美股 16:00 收盤 (美東時間)
        if dt_local.hour < 16:
            return dt_local.strftime("%Y-%m-%d")
        else:
            return (dt_local + datetime.timedelta(days=1)).strftime("%Y-%m-%d")


def is_market_closed(trading_date_str: str, ticker: str, now_dt: Optional[datetime.datetime] = None) -> bool:
    """
    檢查指定交易日該市場是否已經收盤
    """
    target_date = datetime.date.fromisoformat(trading_date_str)
    if is_taiwan_ticker(ticker):
        now_local = now_dt.astimezone(TZ_TAIPEI) if now_dt else datetime.datetime.now(TZ_TAIPEI)
        current_date = now_local.date()
        if target_date < current_date:
            return True
        if target_date == current_date:
            return now_local.hour > 13 or (now_local.hour == 13 and now_local.minute >= 30)
        return False
    else:
        now_local = now_dt.astimezone(TZ_NEW_YORK) if now_dt else datetime.datetime.now(TZ_NEW_YORK)
        current_date = now_local.date()
        if target_date < current_date:
            return True
        if target_date == current_date:
            return now_local.hour >= 16
        return False


def find_actual_close_price(
    hist_df: pd.DataFrame,
    target_date_str: str,
    ticker: str,
    now_dt: Optional[datetime.datetime] = None,
    max_search_days: int = 10
) -> Tuple[Optional[float], Optional[str]]:
    """
    從歷史日 K 棒中尋找 >= target_date_str 的最近已收盤交易日之收盤價。
    徹底解決 DatetimeIndex 時區比對失效與盤前日跳過之問題。
    """
    if hist_df is None or hist_df.empty or "Close" not in hist_df.columns:
        return None, None

    # 將歷史日 K 索引標準化為 'YYYY-MM-DD' 映射表
    hist_date_map = {}
    for idx, row in hist_df.iterrows():
        try:
            if hasattr(idx, "strftime"):
                d_str = idx.strftime("%Y-%m-%d")
            else:
                d_str = str(idx)[:10]
            val = float(row["Close"])
            hist_date_map[d_str] = val
        except Exception:
            continue

    target_dt = datetime.date.fromisoformat(target_date_str)

    for i in range(max_search_days):
        check_str = (target_dt + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        if check_str in hist_date_map:
            # 必須確認該交易日已經正式收盤，不可拿盤中或盤前未定價的資料當做驗證結果
            if is_market_closed(check_str, ticker, now_dt=now_dt):
                return hist_date_map[check_str], check_str
            else:
                # 該交易日是今天但尚未收盤，跳過以待收盤後驗證
                return None, None

    return None, None
