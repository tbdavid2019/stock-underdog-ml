"""
Main entry point for stock prediction application
Uses modular components for cleaner architecture
"""
import datetime
import torch
import pandas as pd
from config import config
from database import SupabaseManager
from data_loader import get_stock_data, download_many, get_tw0050_stocks, get_tw0051_stocks, get_sp500_stocks, get_nasdaq_stocks, get_sox_stocks, get_dji_stocks, get_index_name_map
from models.lstm import prepare_data, train_lstm_model, predict_next_day
from notifier import send_results, send_to_telegram
from logger import logger
from parallel_processor import process_single_stock
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_top_and_bottom_10_potential_stocks(period, selected_indices, db_manager=None):
    """
    依所選指數，回傳各模型潛力排行榜（前 / 後 10）
    結構範例：
    {
        "台灣50": {
            "🥇 前十名 LSTM ":    [ (ticker, pot, curr, pred), ... ],
            "📉 後十名 LSTM ":    [ ... ],
            ...
            "🚀 前十名 TabNet":     [ ... ],
            "⛔ 後十名 TabNet":     [ ... ]
        }, ...
    }
    """
    results = {}

    # --- 指數 → 股票清單 ---------------------------------
    index_stock_map = {
        "台灣50":      get_tw0050_stocks(),
        "台灣中型100": get_tw0051_stocks(),
        "SP500":       get_sp500_stocks(),
        "NASDAQ":      get_nasdaq_stocks(),
        "費城半導體":   get_sox_stocks(),
        "道瓊":        get_dji_stocks(),
    }

    # --- 全域設定 ---------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n計算潛力股... (Period={period})")

    for index_name, stock_list in index_stock_map.items():
        stock_predictions = {}  # 每個指數都重新初始化
        if index_name not in selected_indices:
            continue
        logger.info(f"\n=== 處理指數: {index_name} ===")

        # -------- 序列模型容器 --------
        lstm_preds = []

        # ======== 跑時間序列模型 (Parallel) ========
        logger.info(f"啟動並行處理 (Max Workers: 5)... 分析 {len(stock_list)} 支股票")
        
        completed = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock = {executor.submit(process_single_stock, tic, period): tic for tic in stock_list}
            
            for future in as_completed(future_to_stock):
                tic = future_to_stock[future]
                completed += 1
                try:
                    res = future.result(timeout=60)  # 每支股票最多60秒
                    if 'lstm' in res: 
                        lstm_preds.append(res['lstm'])
                        logger.info(f"✅ [{completed}/{len(stock_list)}] {tic} 完成")
                    else:
                        logger.info(f"⚠️  [{completed}/{len(stock_list)}] {tic} 無結果")
                except TimeoutError:
                    logger.warning(f"⏱️  [{completed}/{len(stock_list)}] 超時跳過 {tic} (60秒)")
                except Exception as e:
                    logger.error(f"❌ [{completed}/{len(stock_list)}] {tic} 失敗: {e}")

        # --- Database：時間序列模型 ---------------------------
        if db_manager and db_manager.enabled:
            if lstm_preds:
                db_manager.save_predictions(index_name, lstm_preds, "LSTM", period)

        # --- 組排行榜（時間序列） -------------------------
        stock_predictions = {}

        stock_predictions.update({
            "🥇 前五名 LSTM 🧠": sorted(lstm_preds, key=lambda x: x[1], reverse=True)[:5],
            "📉 後五名 LSTM 🧠": sorted(lstm_preds, key=lambda x: x[1])[:5],
        })

        # -------- 收尾 --------
        if stock_predictions:
            results[index_name] = stock_predictions

    return results


def main():
    """Main execution function"""
    try:
        # Initialize Database manager (Supabase)
        db_manager = SupabaseManager()

        calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        period = "6mo"
        selected_indices = ["台灣50", "台灣中型100", "SP500"]

        print("計算潛力股...")
        analysis_results = get_top_and_bottom_10_potential_stocks(period, selected_indices, db_manager)

        # Process and send results for each index separately
        for index_name, stock_predictions in analysis_results.items():
            print(f"處理並發送結果: {index_name}")
            name_map = get_index_name_map(index_name)
            send_results(index_name, stock_predictions, name_map=name_map)

    except Exception as e:
        print(f"錯誤: {str(e)}")
        send_to_telegram(f"⚠️ 錯誤: {str(e)}")

    finally:
        # Close DB connection if needed
        pass
        

if __name__ == "__main__":
    main()
