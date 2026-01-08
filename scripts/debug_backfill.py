#!/usr/bin/env python3
"""
Debug: Check why backfill skipped everything
"""
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
import yfinance as yf

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Get a sample prediction
response = supabase.table('predictions').select('*').is_('actual_price', 'null').limit(5).execute()
predictions = response.data

print("=== Debug Backfill Issue ===\n")

for pred in predictions:
    print(f"Ticker: {pred['ticker']}")
    print(f"Prediction Date: {pred['timestamp']}")
    print(f"Current Price: {pred['current_price']}")
    print(f"Predicted Price: {pred['predicted_price']}")
    
    # Try to download data
    try:
        pred_date = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
        print(f"Parsed Date: {pred_date.strftime('%Y-%m-%d')}")
        
        stock = yf.Ticker(pred['ticker'])
        hist = stock.history(period='1mo')
        
        print(f"Downloaded {len(hist)} days of data")
        print(f"Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
        print(f"Dates in index: {[d.date() for d in hist.index[:5]]}")
        
        # Check if next day exists
        from datetime import timedelta
        next_day = pred_date + timedelta(days=1)
        print(f"Looking for: {next_day.date()}")
        print(f"Found: {next_day.date() in hist.index}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 60)
    print()
