#!/usr/bin/env python3
"""
Backfill historical predictions with next-day actual prices
Test on 1/6 predictions using 1/7 actual prices
"""
import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import time

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

start_time = time.time()

print("=== Backfilling Historical Predictions ===\n")

# Fetch all predictions without actual prices
print("1. Fetching predictions to backfill...")
response = supabase.table('predictions').select('*').is_('actual_price', 'null').execute()
predictions = response.data

print(f"Found {len(predictions)} predictions without actual prices\n")

if len(predictions) == 0:
    print("No predictions to backfill!")
    exit(0)

# Group by ticker and date
ticker_groups = {}
for pred in predictions:
    ticker = pred['ticker']
    if ticker not in ticker_groups:
        ticker_groups[ticker] = []
    ticker_groups[ticker].append(pred)

print(f"2. Processing {len(ticker_groups)} unique tickers...\n")

updated = 0
skipped = 0
errors = 0

# Track by index
index_stats = {}

for ticker, ticker_preds in ticker_groups.items():
    try:
        # Download historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo')  # Last month should cover all predictions
        
        for pred in ticker_preds:
            try:
                pred_date = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                index_name = pred.get('index_name', 'Unknown')
                
                # Find NEXT trading day
                search_date = pred_date + timedelta(days=1)
                max_search_days = 10
                
                actual_price = None
                actual_date = None
                
                for i in range(max_search_days):
                    check_date = search_date + timedelta(days=i)
                    # FIX: hist.index contains datetime.date, not datetime.datetime
                    if check_date.date() in [d.date() if hasattr(d, 'date') else d for d in hist.index]:
                        # Find the exact date in index
                        for idx_date in hist.index:
                            idx_date_obj = idx_date.date() if hasattr(idx_date, 'date') else idx_date
                            if idx_date_obj == check_date.date():
                                actual_price = float(hist.loc[idx_date]['Close'])
                                actual_date = check_date
                                break
                        if actual_price is not None:
                            break
                
                if actual_price is None:
                    skipped += 1
                    continue
                
                # Calculate metrics
                current = float(pred['current_price'])
                predicted = float(pred['predicted_price'])
                
                absolute_error = abs(predicted - actual_price)
                percentage_error = ((predicted - actual_price) / actual_price) * 100
                
                # Direction accuracy
                predicted_direction = 1 if predicted > current else -1
                actual_direction = 1 if actual_price > current else -1
                direction_correct = 1.0 if predicted_direction == actual_direction else 0.0
                
                # Update database
                supabase.table('predictions').update({
                    'actual_price': actual_price,
                    'actual_date': actual_date.isoformat(),
                    'accuracy': direction_correct,
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error
                }).eq('id', pred['id']).execute()
                
                updated += 1
                
                # Track stats by index
                if index_name not in index_stats:
                    index_stats[index_name] = {
                        'total': 0,
                        'correct_direction': 0,
                        'errors': []
                    }
                
                index_stats[index_name]['total'] += 1
                if direction_correct == 1.0:
                    index_stats[index_name]['correct_direction'] += 1
                index_stats[index_name]['errors'].append(abs(percentage_error))
                
                if updated % 50 == 0:
                    print(f"  Progress: {updated} updated, {skipped} skipped...")
            
            except Exception as e:
                errors += 1
                continue
    
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        errors += 1

elapsed_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"✅ Backfill Complete!")
print(f"{'='*60}")
print(f"Updated: {updated}")
print(f"Skipped: {skipped}")
print(f"Errors: {errors}")
print(f"Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print()

# Print stats by index
print("📊 Accuracy by Index:")
print(f"{'='*60}")
for index_name, stats in sorted(index_stats.items()):
    if stats['total'] > 0:
        direction_acc = (stats['correct_direction'] / stats['total']) * 100
        avg_error = sum(stats['errors']) / len(stats['errors'])
        print(f"\n{index_name}:")
        print(f"  Total: {stats['total']}")
        print(f"  Direction Accuracy: {direction_acc:.1f}%")
        print(f"  Avg Absolute Error: {avg_error:.2f}%")

print(f"\n{'='*60}")
print("Next step: python backtest/analyze_backtest.py")
