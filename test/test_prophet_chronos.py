#!/usr/bin/env python3
"""
Test Prophet and Chronos models only (skip LSTM)
"""
import sys
sys.path.insert(0, '/home/ec2-user/stock-underdog-ml')

from data_loader import get_tw0050_stocks, get_stock_data
from models.prophet_model import train_prophet_model, predict_with_prophet
from models.chronos_model import prepare_chronos_data
from config import config
from logger import logger

# Test with just 3 stocks to save time
test_stocks = ['2330.TW', '2317.TW', '2454.TW']

print("=== Testing Prophet Model ===")
for ticker in test_stocks:
    try:
        logger.info(f"Testing Prophet on {ticker}")
        data = get_stock_data(ticker, "6mo")
        if len(data) < 60:
            logger.warning(f"Insufficient data for {ticker}")
            continue
            
        model = train_prophet_model(data)
        forecast = predict_with_prophet(model, data)
        
        cur = float(data['Close'].iloc[-1])
        pred = float(forecast['yhat'].max())
        potential = (pred - cur) / cur
        
        print(f"✅ {ticker}: Current={cur:.2f}, Predicted={pred:.2f}, Potential={potential*100:.2f}%")
    except Exception as e:
        print(f"❌ Prophet failed for {ticker}: {e}")

print("\n=== Testing Chronos Model ===")
for ticker in test_stocks:
    try:
        logger.info(f"Testing Chronos on {ticker}")
        data = get_stock_data(ticker, "6mo")
        if len(data) < 60:
            logger.warning(f"Insufficient data for {ticker}")
            continue
            
        ts_df = prepare_chronos_data(data)
        from autogluon.timeseries import TimeSeriesPredictor
        
        predictor = TimeSeriesPredictor(
            prediction_length=10, 
            freq="D", 
            target="target",
            verbosity=0
        )
        predictor.fit(ts_df, hyperparameters={
            "Chronos": {"model_path": "autogluon/chronos-bolt-base"}
        })
        preds = predictor.predict(ts_df)
        
        cur = float(data['Close'].iloc[-1])
        pred = float(preds.values.max())
        potential = (pred - cur) / cur
        
        print(f"✅ {ticker}: Current={cur:.2f}, Predicted={pred:.2f}, Potential={potential*100:.2f}%")
    except Exception as e:
        print(f"❌ Chronos failed for {ticker}: {e}")

print("\n✅ Model testing complete!")
