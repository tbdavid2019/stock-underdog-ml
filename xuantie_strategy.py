"""
玄鐵重劍平均線策略 (Xuan Tie MA Strategy)
核心：順大勢（長線向上）+ 逆小勢（回調買點）
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from logger import logger


def calculate_ma_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    計算移動平均線指標
    
    Args:
        data: 股票數據 DataFrame (必須包含 Close 欄位)
    
    Returns:
        添加了 MA 指標的 DataFrame
    """
    df = data.copy()
    
    # 計算各週期移動平均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['MA250'] = df['Close'].rolling(window=250).mean()
    
    return df


def check_major_trend_up(data: pd.DataFrame, lookback: int = 10) -> bool:
    """
    判斷大勢是否向上 (Filter 1)
    
    檢查條件：
    1. MA60 在過去 N 天內向上傾斜
    2. 當前股價在 MA60 之上
    
    Args:
        data: 包含 MA60 的 DataFrame
        lookback: 回看天數 (預設 10 天，放寬標準)
    
    Returns:
        True 如果大勢向上
    """
    if len(data) < lookback + 1:
        return False
    
    # 取最近的數據
    recent = data.tail(lookback + 1)
    
    # 檢查 MA60 是否存在且有效
    if 'MA60' not in recent.columns or recent['MA60'].isna().all():
        return False
    
    # 1. MA60 向上傾斜：當前 MA60 > N 天前的 MA60
    ma60_current = recent['MA60'].iloc[-1]
    ma60_past = recent['MA60'].iloc[0]
    
    if pd.isna(ma60_current) or pd.isna(ma60_past):
        return False
    
    ma60_up = ma60_current > ma60_past
    
    # 2. 股價在 MA60 之上
    price_above_ma60 = recent['Close'].iloc[-1] > ma60_current
    
    return ma60_up and price_above_ma60


def check_minor_pullback(data: pd.DataFrame, tolerance: float = 0.05) -> Tuple[bool, str]:
    """
    判斷小勢回調 (Filter 2)
    
    檢查條件：
    股價接近或短暫跌破 MA60/MA120 (±5%)
    
    Args:
        data: 包含 MA 指標的 DataFrame
        tolerance: 容忍範圍 (預設 ±5%，放寬標準)
    
    Returns:
        (是否回調, 回調類型描述)
    """
    if len(data) == 0:
        return False, ""
    
    latest = data.iloc[-1]
    
    # 安全取值函數
    def safe_float(value):
        if pd.isna(value):
            return None
        if hasattr(value, 'iloc'):  # Series
            return float(value.iloc[0])
        return float(value)
    
    close_price = safe_float(latest['Close'])
    if close_price is None:
        return False, ""
    
    # 檢查 MA120 回調
    if 'MA120' in data.columns:
        ma120 = safe_float(latest['MA120'])
        if ma120 is not None:
            diff_pct = (close_price - ma120) / ma120
            if -tolerance <= diff_pct <= tolerance:
                return True, f"MA120回調 ({diff_pct*100:.1f}%)"
    
    # 檢查 MA60 回調
    if 'MA60' in data.columns:
        ma60 = safe_float(latest['MA60'])
        if ma60 is not None:
            diff_pct = (close_price - ma60) / ma60
            if -tolerance <= diff_pct <= tolerance:
                return True, f"MA60回調 ({diff_pct*100:.1f}%)"
    
    return False, ""


def check_xuantie_signal(data: pd.DataFrame, ticker: str = "", lookback: int = 10, tolerance: float = 0.05) -> Dict:
    """
    綜合判斷玄鐵重劍買入信號
    
    Args:
        data: 股票歷史數據
        ticker: 股票代碼
        lookback: MA250 斜率檢查天數 (預設 10 天)
        tolerance: 回調容忍範圍 (預設 ±5%)
    
    Returns:
        信號字典，包含：
        - signal: 是否買入 (True/False)
        - major_trend: 大勢是否向上
        - pullback: 是否回調
        - pullback_type: 回調類型
        - current_price: 當前價格
        - ma250: MA250 值
    """
    # 計算 MA 指標
    df = calculate_ma_indicators(data)
    
    if len(df) < 70:
        return {
            'signal': False,
            'reason': '數據不足 (需要至少70天)',
            'ticker': ticker
        }
    
    # 1. 檢查大勢
    major_trend_up = check_major_trend_up(df, lookback=lookback)
    
    # 2. 檢查小勢回調
    pullback, pullback_type = check_minor_pullback(df, tolerance=tolerance)
    
    # 3. 綜合判斷
    signal = major_trend_up and pullback
    
    latest = df.iloc[-1]
    
    result = {
        'ticker': ticker,
        'signal': signal,
        'major_trend': major_trend_up,
        'pullback': pullback,
        'pullback_type': pullback_type,
        'current_price': float(latest['Close']),
        'ma5': float(latest['MA5']) if not pd.isna(latest['MA5']) else None,
        'ma10': float(latest['MA10']) if not pd.isna(latest['MA10']) else None,
        'ma60': float(latest['MA60']) if not pd.isna(latest['MA60']) else None,
        'ma120': float(latest['MA120']) if not pd.isna(latest['MA120']) else None,
        'ma250': float(latest['MA250']) if not pd.isna(latest['MA250']) else None,
    }
    
    return result


def filter_stocks_by_xuantie(stock_list: list, period: str = "1y", lookback: int = 10, tolerance: float = 0.05) -> pd.DataFrame:
    """
    批量篩選符合玄鐵重劍策略的股票
    
    Args:
        stock_list: 股票代碼列表
        period: 數據週期
        lookback: MA250 斜率檢查天數 (預設 10 天)
        tolerance: 回調容忍範圍 (預設 ±5%)
    
    Returns:
        符合條件的股票 DataFrame
    """
    from data_loader import get_stock_data
    
    results = []
    
    logger.info(f"🗡️  開始玄鐵重劍策略篩選... ({len(stock_list)} 支股票)")
    logger.info(f"   參數: lookback={lookback}天, tolerance=±{tolerance*100:.0f}%")
    
    for i, ticker in enumerate(stock_list, 1):
        try:
            # 獲取數據
            data = get_stock_data(ticker, period)
            
            if data.empty or len(data) < 70:
                logger.debug(f"[{i}/{len(stock_list)}] {ticker} 數據不足")
                continue
            
            # 檢查信號
            signal_info = check_xuantie_signal(data, ticker, lookback=lookback, tolerance=tolerance)
            
            if signal_info['signal']:
                logger.info(f"✅ [{i}/{len(stock_list)}] {ticker} - {signal_info['pullback_type']}")
                results.append(signal_info)
            else:
                reason = []
                if not signal_info['major_trend']:
                    reason.append("大勢未向上")
                if not signal_info['pullback']:
                    reason.append("未回調")
                logger.debug(f"❌ [{i}/{len(stock_list)}] {ticker} - {', '.join(reason)}")
                
        except Exception as e:
            logger.error(f"處理 {ticker} 時出錯: {e}")
    
    if results:
        df = pd.DataFrame(results)
        logger.info(f"🎯 篩選完成！符合條件: {len(results)}/{len(stock_list)} 支")
        return df
    else:
        logger.info(f"⚠️  篩選完成！無符合條件的股票")
        return pd.DataFrame()
