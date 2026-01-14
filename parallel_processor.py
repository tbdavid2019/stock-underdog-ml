from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pandas as pd
from config import config
from data_loader import get_stock_data
from models.lstm import prepare_data, train_lstm_model, predict_next_day
from logger import logger


def process_single_stock(tic, period):
    """
    Process a single stock: download data, train LSTM model, return predictions.
    Returns a dict with key: 'lstm'
    Value is a tuple: (ticker, potential, current_price, predicted_price)
    """
    results = {}
    
    try:
        # Download data once
        try:
            data = get_stock_data(tic, period)
            if len(data) < 60:
                return results
        except Exception as e:
            logger.warning(f"Data download failed for {tic}: {e}")
            return results

        # ----- LSTM (Next-Day Prediction) -----
        try:
            X, y, scaler = prepare_data(data)
            lstm_model = train_lstm_model(X, y)
            
            # Predict NEXT trading day's price
            predicted_next_day = predict_next_day(lstm_model, data, scaler)
            
            # Current price (today's close)
            cur = float(data['Close'].iloc[-1].item()) if hasattr(data['Close'].iloc[-1], 'item') else float(data['Close'].iloc[-1])
            
            # Potential = (predicted_tomorrow - current_today) / current_today
            pot = (predicted_next_day - cur) / cur
            
            results['lstm'] = (tic, pot, cur, predicted_next_day)
        except Exception as e:
            logger.debug(f"LSTM 失敗 {tic}: {e}")
            pass
    except Exception as e:
        logger.error(f"Error processing {tic}: {e}")
            
    return results
