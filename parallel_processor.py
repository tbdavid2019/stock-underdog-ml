from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from config import config
from data_loader import get_stock_data
from models.lstm import prepare_data, train_lstm_model, predict_stock
from models.transformer import train_transformer_model, predict_transformer
from models.prophet_model import train_prophet_model, predict_with_prophet
from models.chronos_model import prepare_chronos_data
from logger import logger

def process_single_stock(tic, period):
    """
    Process a single stock: download data, train models, return predictions.
    Returns a dict with keys: 'lstm', 'transformer', 'prophet', 'chronos'
    Values are tuples: (ticker, potential, current_price, predicted_price)
    """
    results = {}
    
    # Download data once
    try:
        data = get_stock_data(tic, period)
        if len(data) < 60:
            return results
    except Exception as e:
        logger.warning(f"Data download failed for {tic}: {e}")
        return results

    # ----- LSTM -----
    try:
        X, y, scaler = prepare_data(data)
        lstm_model = train_lstm_model(X, y)
        lstm_series = predict_stock(lstm_model, data, scaler)
        
        # Handle types safely
        cur = float(data['Close'].iloc[-1].item()) if hasattr(data['Close'].iloc[-1], 'item') else float(data['Close'].iloc[-1])
        pred = float(lstm_series.max())
        pot = (pred - cur) / cur
        results['lstm'] = (tic, pot, cur, pred)
    except Exception as e:
        logger.debug(f"LSTM 失敗 {tic}: {e}")
        pass

    # ----- Transformer -----
    if config.use_transformer:
        try:
            tf_data = get_stock_data(tic, config.transformer_period)
            X_tf, y_tf, tf_scaler = prepare_data(tf_data)
            tf_shape = (X_tf.shape[1], X_tf.shape[2])
            tf_model = train_transformer_model(X_tf, y_tf, tf_shape)
            tf_series = predict_transformer(tf_model, tf_data, tf_scaler)
            
            cur = float(tf_data['Close'].iloc[-1].item()) if hasattr(tf_data['Close'].iloc[-1], 'item') else float(tf_data['Close'].iloc[-1])
            pred = float(tf_series.max())
            pot = (pred - cur) / cur
            results['transformer'] = (tic, pot, cur, pred)
        except Exception as e:
            logger.debug(f"Transformer 失敗 {tic}: {e}")
            pass

    # ----- Prophet -----
    if config.use_prophet:
        try:
            p_model = train_prophet_model(data)
            p_fore = predict_with_prophet(p_model, data)
            
            cur = float(data['Close'].iloc[-1].item())
            pred = float(p_fore['yhat'].max())
            pot = (pred - cur) / cur
            results['prophet'] = (tic, pot, cur, pred)
        except Exception as e:
            logger.debug(f"Prophet 失敗 {tic}: {e}")
            pass

    # ----- Chronos-Bolt -----
    if config.use_chronos:
        try:
            ch_data = get_stock_data(tic, config.chronos_period)
            if len(ch_data) >= 60:
                ts_df = prepare_chronos_data(ch_data)
                from autogluon.timeseries import TimeSeriesPredictor
                predictor = TimeSeriesPredictor(
                    prediction_length=10, freq="D", target="target", verbosity=0)
                predictor.fit(ts_df, hyperparameters={
                    "Chronos": {"model_path": "autogluon/chronos-bolt-base"}
                })
                preds = predictor.predict(ts_df)
                
                cur = float(ch_data['Close'].iloc[-1].item())
                pred = float(preds.values.max())
                pot = (pred - cur) / cur
                results['chronos'] = (tic, pot, cur, pred)
        except Exception as e:
            logger.debug(f"Chronos 失敗 {tic}: {e}")
            pass
            
    return results
