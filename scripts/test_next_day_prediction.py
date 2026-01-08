#!/usr/bin/env python3
"""
Quick test: Run prediction on 3 stocks and verify next-day backtest
"""
import os
import sys
sys.path.insert(0, '/home/ec2-user/stock-underdog-ml')

from dotenv import load_dotenv
from data_loader import get_stock_data
from models.lstm import prepare_data, train_lstm_model, predict_next_day
from supabase import create_client
import datetime

load_dotenv()

print("=== Testing Next-Day Prediction ===\n")

# Test stocks
test_tickers = ["2330.TW", "AAPL", "TSLA"]

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

predictions = []

for ticker in test_tickers:
    try:
        print(f"Processing {ticker}...")
        
        # Download data
        data = get_stock_data(ticker, "6mo")
        
        # Train model
        X, y, scaler = prepare_data(data)
        model = train_lstm_model(X, y)
        
        # Predict next day
        predicted_next_day = predict_next_day(model, data, scaler)
        current_price = float(data['Close'].iloc[-1])
        potential = (predicted_next_day - current_price) / current_price
        
        print(f"  Current: ${current_price:.2f}")
        print(f"  Predicted (next day): ${predicted_next_day:.2f}")
        print(f"  Potential: {potential*100:.2f}%\n")
        
        # Save to database
        record = {
            'index_name': 'TEST',
            'model_name': 'LSTM',
            'ticker': ticker,
            'current_price': current_price,
            'predicted_price': predicted_next_day,
            'potential': potential,
            'period': '1day',
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        supabase.table('predictions').insert(record).execute()
        predictions.append(record)
        
    except Exception as e:
        print(f"  ❌ Error: {e}\n")

print(f"✅ Saved {len(predictions)} predictions to database")
print("\n📝 Next steps:")
print("1. Wait for next trading day")
print("2. Run: python backtest/backtest.py")
print("3. Check results: python backtest/analyze_backtest.py")
