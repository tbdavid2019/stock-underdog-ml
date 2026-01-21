"""
雙軌策略主程序 - 整合 LSTM 預測 + 玄鐵重劍策略
策略 1: 玄鐵重劍 (波段操作，持有 2-4 週)
策略 2: LSTM 預測 (短線操作，持有 1-7 天)
"""
import datetime
import torch
import pandas as pd
from config import config
from database import SupabaseManager
from data_loader import (
    get_stock_data, 
    get_tw0050_stocks, 
    get_tw0051_stocks, 
    get_sp500_stocks
)
from models.lstm import prepare_data, train_lstm_model, predict_next_day
from xuantie_strategy import filter_stocks_by_xuantie, check_xuantie_signal
from logger import logger
from notifier_dual import send_dual_strategy_results
from database import SupabaseManager
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_lstm_predictions(stock_list, period="6mo"):
    """
    執行 LSTM 預測
    
    Returns:
        List of (ticker, potential, current_price, predicted_price)
    """
    results = []
    total = len(stock_list)
    
    logger.info(f"🤖 開始 LSTM 預測... ({total} 支股票)")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        
        for ticker in stock_list:
            future = executor.submit(process_single_stock_lstm, ticker, period)
            futures[future] = ticker
        
        completed = 0
        for future in as_completed(futures):  # 無總體超時限制，讓所有股票都有機會完成
            ticker = futures[future]
            completed += 1
            
            try:
                # 單支股票超時保護：最多等待 260 秒
                result = future.result(timeout=260)
                if result:
                    results.append(result)
                    logger.info(f"✅ [{completed}/{total}] {ticker} - 預測漲幅: {result['potential']:+.2f}%")
                else:
                    logger.debug(f"❌ [{completed}/{total}] {ticker} - 預測失敗")
            except TimeoutError:
                logger.warning(f"⏱️ [{completed}/{total}] {ticker} - 單支股票超時 (>60秒)，跳過")
            except Exception as e:
                logger.error(f"❌ [{completed}/{total}] {ticker} - 處理錯誤: {e}")
    
    # 按潛力排序
    results.sort(key=lambda x: x['potential'], reverse=True)
    return results


def get_fundamental_data(ticker):
    """
    獲取基本面數據 (PE/PB/EV/EBITDA)
    
    Returns:
        dict with pe, pb, forward_pe, ev_ebitda
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'pb': info.get('priceToBook'),
            'ev_ebitda': info.get('enterpriseToEbitda')
        }
    except:
        return {'pe': None, 'forward_pe': None, 'pb': None, 'ev_ebitda': None}


def process_single_stock_lstm(ticker, period):
    """
    單支股票 LSTM 預測
    
    Returns:
        dict with ticker, potential, current_price, predicted_price, pe, pb
    """
    try:
        data = get_stock_data(ticker, period)
        
        if data.empty or len(data) < 30:
            return None
        
        current_price = float(data['Close'].iloc[-1])
        
        # 準備數據
        X, y, scaler = prepare_data(data)
        if len(X) < 10:
            return None
        
        # 訓練模型
        model = train_lstm_model(X, y)
        
        # 預測（傳入原始 DataFrame）
        predicted_price = predict_next_day(model, data, scaler)
        
        if predicted_price is None or predicted_price <= 0:
            return None
        
        # 計算潛力
        potential = ((predicted_price - current_price) / current_price) * 100
        
        # 獲取基本面
        fundamentals = get_fundamental_data(ticker)
        
        return {
            'ticker': ticker,
            'potential': potential,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'pe': fundamentals['pe'],
            'pb': fundamentals['pb'],
            'ev_ebitda': fundamentals['ev_ebitda']
        }
        
    except Exception as e:
        logger.debug(f"LSTM 預測 {ticker} 失敗: {e}")
        return None


def run_dual_strategy(index_name, stock_list, period="6mo"):
    """
    執行雙軌策略分析
    
    Args:
        index_name: 指數名稱
        stock_list: 股票列表
        period: 數據週期
    
    Returns:
        dict with xuantie_results, lstm_results, overlap_results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 分析指數: {index_name} ({len(stock_list)} 支股票)")
    logger.info(f"{'='*60}\n")
    
    # ============= 軌道 1: 玄鐵重劍策略 =============
    logger.info("🗡️  【軌道 1】玄鐵重劍策略 (波段操作)")
    xuantie_df = filter_stocks_by_xuantie(
        stock_list, 
        period=period,
        lookback=10,
        tolerance=0.05
    )
    
    xuantie_stocks = set(xuantie_df['ticker'].tolist()) if not xuantie_df.empty else set()
    
    # 為玄鐵結果加入 PE/PB/EV/EBITDA
    if not xuantie_df.empty:
        for idx, row in xuantie_df.iterrows():
            fundamentals = get_fundamental_data(row['ticker'])
            xuantie_df.at[idx, 'pe'] = fundamentals['pe']
            xuantie_df.at[idx, 'pb'] = fundamentals['pb']
            xuantie_df.at[idx, 'ev_ebitda'] = fundamentals['ev_ebitda']
    
    logger.info(f"✅ 玄鐵策略符合: {len(xuantie_stocks)} 支\n")
    
    # ============= 軌道 2: LSTM 預測 =============
    logger.info("🤖 【軌道 2】LSTM 預測 (短線操作)")
    lstm_results = run_lstm_predictions(stock_list, period)
    logger.info(f"✅ LSTM 預測完成: {len(lstm_results)} 支\n")
    
    # ============= 找出雙重符合 =============
    overlap = []
    for result in lstm_results:
        ticker = result['ticker']
        if ticker in xuantie_stocks:
            # 找出玄鐵策略的詳細信息
            xuantie_info = xuantie_df[xuantie_df['ticker'] == ticker].iloc[0]
            overlap.append({
                'ticker': ticker,
                'lstm_potential': result['potential'],
                'current_price': result['current_price'],
                'predicted_price': result['predicted_price'],
                'pullback_type': xuantie_info['pullback_type'],
                'ma60': xuantie_info['ma60'],
                'pe': result['pe'],
                'pb': result['pb'],
                'ev_ebitda': result['ev_ebitda']
            })
    
    overlap_df = pd.DataFrame(overlap) if overlap else pd.DataFrame()
    
    return {
        'xuantie_results': xuantie_df,
        'lstm_results': lstm_results,
        'overlap_results': overlap_df
    }


def format_value(val, decimal=2):
    """格式化數值，處理 None"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.{decimal}f}"
    return str(val)


def print_dual_strategy_report(index_name, results):
    """
    美化輸出雙軌策略報告 - 表格式
    """
    xuantie_df = results['xuantie_results']
    lstm_results = results['lstm_results']
    overlap_df = results['overlap_results']
    
    logger.info(f"\n{'='*100}")
    logger.info(f"📊 投資建議報告 - {index_name}")
    logger.info(f"{'='*100}\n")
    
    # ====== 軌道 1: 玄鐵重劍 ======
    logger.info("🗡️  【波段操作】玄鐵重劍策略 (持有 2-4 週) - 技術面買點")
    logger.info(f"   符合條件: {len(xuantie_df)} 支\n")
    
    if not xuantie_df.empty:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'價格':>8} {'MA60':>8} {'回調類型':<18} {'PE':>6} {'PB':>6}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*8} {'-'*8} {'-'*18} {'-'*6} {'-'*6}")
        for idx, row in xuantie_df.head(10).iterrows():
            logger.info(
                f"   {idx+1:<4} {row['ticker']:<10} "
                f"{row['current_price']:>8.2f} "
                f"{format_value(row.get('ma60')):>8} "
                f"{row['pullback_type']:<18} "
                f"{format_value(row.get('pe')):>6} "
                f"{format_value(row.get('pb')):>6}"
            )
    else:
        logger.info("   (本期無符合條件的股票)")
    
    logger.info("")
    
    # ====== 軌道 2: LSTM 預測 ======
    logger.info("🤖 【短線操作】LSTM 預測 (持有 1-7 天) - 預測漲幅排行")
    logger.info(f"   預測完成: {len(lstm_results)} 支\n")
    
    if lstm_results:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'預測漲幅':>10} {'現價':>8} {'→':^3} {'預測價':>8} {'PE':>6} {'PB':>6}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*3} {'-'*8} {'-'*6} {'-'*6}")
        for i, result in enumerate(lstm_results[:10], 1):
            logger.info(
                f"   {i:<4} {result['ticker']:<10} "
                f"{result['potential']:>+9.2f}% "
                f"{result['current_price']:>8.2f} {'→':^3} "
                f"{result['predicted_price']:>8.2f} "
                f"{format_value(result.get('pe')):>6} "
                f"{format_value(result.get('pb')):>6}"
            )
    else:
        logger.info("   (本期無預測結果)")
    
    logger.info("")
    
    # ====== 雙重符合 ======
    logger.info("⭐ 【優先推薦】技術面買點 + LSTM 看好 + 基本面檢視")
    logger.info(f"   雙重符合: {len(overlap_df)} 支\n")
    
    if not overlap_df.empty:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'LSTM':>8} {'回調':>6} {'MA60':>8} {'PE':>6} {'PB':>6} {'綜合評價'}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*20}")
        for idx, row in overlap_df.iterrows():
            # 簡單評分
            score_parts = []
            if row['lstm_potential'] > 3:
                score_parts.append("LSTM強")
            pe_val = row.get('pe')
            if pe_val and pe_val < 20:
                score_parts.append("低PE")
            pb_val = row.get('pb')
            if pb_val and pb_val < 3:
                score_parts.append("低PB")
            
            score = " | ".join(score_parts) if score_parts else "觀察"
            
            logger.info(
                f"   {idx+1:<4} {row['ticker']:<10} "
                f"{row['lstm_potential']:>+7.2f}% "
                f"{row['pullback_type'][:6]:>6} "
                f"{format_value(row.get('ma60')):>8} "
                f"{format_value(row.get('pe')):>6} "
                f"{format_value(row.get('pb')):>6} "
                f"{score}"
            )
    else:
        logger.info("   (本期無雙重符合的股票)")
    
    logger.info(f"\n{'='*100}\n")


def main():
    """主程序"""
    try:
        logger.info("🚀 啟動雙軌策略分析系統...")
        logger.info(f"⏰ 執行時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 初始化資料庫
        db_manager = SupabaseManager()
        
        # 設定
        period = "6mo"
        
        # 指數清單（僅美股 SP500 完整測試）
        indices = {
            "SP500": get_sp500_stocks()
        }
        
        # 對每個指數執行雙軌策略
        all_results = {}
        
        for index_name, stock_list in indices.items():
            results = run_dual_strategy(index_name, stock_list, period)
            all_results[index_name] = results
            
            # 輸出報告
            print_dual_strategy_report(index_name, results)
            
            # 發送通知
            send_dual_strategy_results(index_name, results)
            
            # 保存到資料庫
            if db_manager.enabled:
                db_manager.save_dual_strategy_results(index_name, results, period)
        
        logger.info("✅ 雙軌策略分析完成！")
        
    except Exception as e:
        logger.error(f"⚠️ 錯誤: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
