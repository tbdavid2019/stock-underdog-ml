"""
Main entry point for stock prediction application
Uses modular components for cleaner architecture
"""
import datetime
import torch
import pandas as pd
from config import config
from database import SupabaseManager
from data_loader import get_stock_data, download_many, get_tw0050_stocks, get_tw0051_stocks, get_sp500_stocks, get_nasdaq_stocks, get_sox_stocks, get_dji_stocks
from models.lstm import prepare_data, train_lstm_model, predict_stock
from models.transformer import train_transformer_model, predict_transformer
from models.prophet_model import train_prophet_model, predict_with_prophet
from models.chronos_model import prepare_chronos_data, train_and_predict_chronos
from models.cross_section import CROSS_MODELS, import_model, build_cross_xy, train_tabnet, train_cross_loop
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
        lstm_preds, prophet_preds = [], []
        transformer_preds, chronos_preds = [], []

        # ======== 先跑橫斷面模型（Cross Sectional Models） ========
        if config.use_cross:
            logger.info("啟動 Cross Section 策略...")
            try:
                raw_df = download_many(stock_list, config.cross_period)
                Xc, yc, meta_c = build_cross_xy(raw_df)

                # 修正 mask_last = Series vs Series 錯誤
                max_date = pd.Timestamp(meta_c['Date'].max())
                logger.debug(f"[LOG] max_date: {max_date}, type: {type(max_date)}")
                logger.debug(f"[LOG] meta_c['Date'] head: {meta_c['Date'].head()}, dtype: {meta_c['Date'].dtype}")
                mask_last = meta_c['Date'].values == max_date
                logger.debug(f"[LOG] mask_last: {mask_last}, shape: {mask_last.shape}, type: {type(mask_last)}")
                meta_last = meta_c[mask_last].reset_index(drop=True)
                logger.debug(f"[LOG] meta_last shape: {meta_last.shape}, columns: {meta_last.columns}")

                latest_close = (
                    raw_df[raw_df['Date'] == max_date]
                    .groupby('Ticker')['Close']
                    .first()
                )
                logger.debug(f"[LOG] latest_close index: {latest_close.index}, type: {type(latest_close)}")

                # 執行 Cross 模型（TabNet，SFM，ADDModel）
                for m_path, cls_list in CROSS_MODELS:
                    ModelClass = import_model(m_path, cls_list)
                    if ModelClass is None:
                        continue
                    logger.info(f"🔍 Cross 訓練 {ModelClass.__name__} …")
                    try:
                        if ModelClass.__name__ == "TabNet":
                            preds_all = train_tabnet(Xc, yc, epochs=config.cross_epochs, device=device)
                        else:
                            preds_all = train_cross_loop(ModelClass, Xc, yc, config.cross_epochs, device)
                        logger.debug(f"[LOG] preds_all shape: {getattr(preds_all, 'shape', None)}, type: {type(preds_all)}")
                        preds_last = preds_all[mask_last]
                        logger.debug(f"[LOG] preds_last shape: {getattr(preds_last, 'shape', None)}, type: {type(preds_last)}")

                        # 組 TabNet / SFM / ADDModel 結果
                        records = [
                            (
                                tic,
                                p,                               # 預測潛力
                                float(latest_close[tic]),        # 現價
                                float(latest_close[tic] * (1+p)) # 預測價
                            )
                            for tic, p in zip(meta_last['Ticker'], preds_last)
                        ]
                        logger.debug(f"[LOG] records sample: {records[:3]}")

                        # 寫 Databse (Supabase)
                        if db_manager and db_manager.enabled:
                            db_manager.save_predictions(index_name, records, ModelClass.__name__, config.cross_period)
                        else:
                            logger.info("DB Manager not initialized or enabled.")

                        # 排行榜
                        stock_predictions.update({
                            f"🚀 前五名 {ModelClass.__name__}": sorted(records, key=lambda x:x[1], reverse=True)[:5],
                            f"⛔ 後五名 {ModelClass.__name__}": sorted(records, key=lambda x:x[1])[:5],
                        })
                        logger.debug(f"[DEBUG] stock_predictions keys after update: {list(stock_predictions.keys())}")
                        logger.debug(f"[DEBUG] stock_predictions lens after update: {[len(v) for v in stock_predictions.values()]}")

                        if len(preds_last) == 0:
                            logger.warning(f"{ModelClass.__name__} 沒有產生預測結果")
                            continue

                    except Exception as e:
                        logger.error(f"{ModelClass.__name__} 失敗: {e}")
                        continue

            except Exception as e:
                logger.error(f"Cross‑section 流程錯誤: {e}")

        # ======== 跑時間序列模型 (Parallel) ========
        logger.info(f"啟動並行處理 (Max Workers: 5)... 分析 {len(stock_list)} 支股票")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock = {executor.submit(process_single_stock, tic, period): tic for tic in stock_list}
            
            for future in as_completed(future_to_stock):
                tic = future_to_stock[future]
                try:
                    res = future.result()
                    if 'lstm' in res: lstm_preds.append(res['lstm'])
                    if 'transformer' in res: transformer_preds.append(res['transformer'])
                    if 'prophet' in res: prophet_preds.append(res['prophet'])
                    if 'chronos' in res: chronos_preds.append(res['chronos'])
                    
                    # Optional: Progress logging
                    # print(f"完成: {tic}")
                except Exception as e:
                    logger.error(f"處理失敗 {tic}: {e}")

        # --- Database：時間序列模型 ---------------------------
        if db_manager and db_manager.enabled:
            if lstm_preds:
                db_manager.save_predictions(index_name, lstm_preds, "LSTM", period)
            if config.use_prophet and prophet_preds:
                db_manager.save_predictions(index_name, prophet_preds, "Prophet", period)
            if config.use_transformer and transformer_preds:
                db_manager.save_predictions(index_name, transformer_preds, "Transformer", period)
            if config.use_chronos and chronos_preds:
                db_manager.save_predictions(index_name, chronos_preds, "Chronos-Bolt", config.chronos_period)

        # --- 組排行榜（時間序列） -------------------------
        stock_predictions = stock_predictions if 'stock_predictions' in locals() else {}

        stock_predictions.update({
            "🥇 前五名 LSTM 🧠": sorted(lstm_preds, key=lambda x: x[1], reverse=True)[:5],
            "📉 後五名 LSTM 🧠": sorted(lstm_preds, key=lambda x: x[1])[:5],
        })
        if config.use_prophet and prophet_preds:
            stock_predictions.update({
                "🚀 前五名 Prophet 🔮": sorted(prophet_preds, key=lambda x: x[1], reverse=True)[:5],
                "⛔ 後五名 Prophet 🔮": sorted(prophet_preds, key=lambda x: x[1])[:5],
            })
        if config.use_transformer and transformer_preds:
            stock_predictions.update({
                "🚀 前五名 Transformer 🔄": sorted(transformer_preds, key=lambda x: x[1], reverse=True)[:5],
                "⛔ 後五名 Transformer 🔄": sorted(transformer_preds, key=lambda x: x[1])[:5],
            })
        if config.use_chronos and chronos_preds:
            stock_predictions.update({
                "🚀 前五名 Chronos-Bolt ⚡": sorted(chronos_preds, key=lambda x: x[1], reverse=True)[:5],
                "⛔ 後五名 Chronos-Bolt ⚡": sorted(chronos_preds, key=lambda x: x[1])[:5],
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
            send_results(index_name, stock_predictions)

    except Exception as e:
        print(f"錯誤: {str(e)}")
        send_to_telegram(f"⚠️ 錯誤: {str(e)}")

    finally:
        # Close DB connection if needed
        pass
        

if __name__ == "__main__":
    main()
