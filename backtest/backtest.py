#!/usr/bin/env python3
"""
Improved Backtesting for NEXT-DAY Predictions
Fetches actual prices from the NEXT trading day after prediction
"""
import os
from dotenv import load_dotenv
from supabase import create_client
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("=== Next-Day Backtesting ===\n")

# Fetch predictions without actual prices
print("1. Fetching predictions to verify...")
response = supabase.table('predictions').select('*').is_('actual_price', 'null').execute()
predictions = response.data

print(f"Found {len(predictions)} predictions without actual prices\n")

if len(predictions) == 0:
    print("No predictions to verify!")
    exit(0)

# Group by ticker
tickers = {}
for pred in predictions:
    ticker = pred['ticker']
    if ticker not in tickers:
        tickers[ticker] = []
    tickers[ticker].append(pred)

print(f"2. Downloading next-day actual prices for {len(tickers)} unique tickers...")

updated = 0
skipped = 0

for ticker, ticker_preds in tickers.items():
    try:
        # Download recent historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        
        for pred in ticker_preds:
            try:
                pred_date = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                
                # Find NEXT trading day after prediction
                # Start from the day after prediction
                search_date = pred_date + timedelta(days=1)
                max_search_days = 10  # Search up to 10 days for next trading day
                
                actual_price = None
                actual_date = None
                
                for i in range(max_search_days):
                    check_date = search_date + timedelta(days=i)
                    if check_date.date() in hist.index:
                        actual_price = float(hist.loc[check_date.date()]['Close'])
                        actual_date = check_date
                        break
                
                if actual_price is None:
                    # No trading day found (maybe too recent or market closed)
                    skipped += 1
                    continue
                
                # Calculate metrics
                current = float(pred['current_price'])
                predicted = float(pred['predicted_price'])
                
                # Accuracy: how well did we predict the next day
                absolute_error = abs(predicted - actual_price)
                percentage_error = ((predicted - actual_price) / actual_price) * 100
                
                # Direction accuracy: did we predict up/down correctly?
                predicted_direction = 1 if predicted > current else -1
                actual_direction = 1 if actual_price > current else -1
                direction_correct = 1.0 if predicted_direction == actual_direction else 0.0
                
                # Update record
                supabase.table('predictions').update({
                    'actual_price': actual_price,
                    'actual_date': actual_date.isoformat(),
                    'accuracy': direction_correct,  # 1.0 if direction correct, 0.0 otherwise
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error
                }).eq('id', pred['id']).execute()
                
                updated += 1
                direction_symbol = "✅" if direction_correct == 1.0 else "❌"
                print(f"  {direction_symbol} {ticker}: Pred {predicted:.2f}, Actual {actual_price:.2f}, Error {percentage_error:.2f}%")
            
            except Exception as e:
                print(f"  ⚠️ {ticker} (single prediction): {e}")
                continue
    
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")

print(f"\n✅ Updated {updated} predictions")
print(f"⏭️ Skipped {skipped} predictions (no next trading day data yet)")

# Generate performance report
print("\n3. Model Performance Summary:")
response = supabase.table('predictions').select('model_name, accuracy, percentage_error').not_.is_('actual_price', 'null').execute()
results = response.data

if results:
    df = pd.DataFrame(results)
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        direction_acc = (model_df['accuracy'].sum() / len(model_df)) * 100
        avg_error = model_df['percentage_error'].abs().mean()
        
        print(f"\n{model}:")
        print(f"  Direction Accuracy: {direction_acc:.1f}%")
        print(f"  Avg Absolute Error: {avg_error:.2f}%")
        print(f"  Total Verified: {len(model_df)}")
else:
    print("No verified predictions yet!")
