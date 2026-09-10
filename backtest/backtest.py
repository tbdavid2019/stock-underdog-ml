#!/usr/bin/env python3
"""
Improved Backtesting for NEXT-DAY Predictions
Fetches actual prices from the NEXT trading day after prediction
"""
import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path to import logger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger
from backtest.resolver import (
    parse_prediction_timestamp,
    get_target_trading_date,
    find_actual_close_price
)

# Setup logger for backtest
logger = setup_logger('backtest', 'logs/backtest.log')

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

logger.info("=== Next-Day Backtesting ===\n")

# Fetch predictions without actual prices
logger.info("1. Fetching predictions to verify...")
response = supabase.table('predictions').select('*').is_('actual_price', 'null').execute()
predictions = response.data

logger.info(f"Found {len(predictions)} predictions without actual prices\n")

if len(predictions) == 0:
    logger.info("No predictions to verify!")
    exit(0)

# Group by ticker
tickers = {}
for pred in predictions:
    ticker = pred['ticker']
    if ticker not in tickers:
        tickers[ticker] = []
    tickers[ticker].append(pred)

logger.info(f"2. Downloading next-day actual prices for {len(tickers)} unique tickers...")

updated = 0
skipped = 0

for ticker, ticker_preds in tickers.items():
    try:
        # Download recent historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        
        for pred in ticker_preds:
            try:
                pred_date = parse_prediction_timestamp(pred['timestamp'])
                target_date_str = get_target_trading_date(pred_date, ticker)

                actual_price, actual_date_str = find_actual_close_price(
                    hist_df=hist,
                    target_date_str=target_date_str,
                    ticker=ticker
                )

                if actual_price is None:
                    # No closed trading day found (maybe too recent or market not closed yet)
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
                    'actual_date': f"{actual_date_str}T00:00:00+00:00",
                    'accuracy': direction_correct,  # 1.0 if direction correct, 0.0 otherwise
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error
                }).eq('id', pred['id']).execute()
                
                updated += 1
                direction_symbol = "✅" if direction_correct == 1.0 else "❌"
                logger.info(f"  {direction_symbol} {ticker}: Pred {predicted:.2f}, Actual {actual_price:.2f}, Error {percentage_error:.2f}%")
            
            except Exception as e:
                logger.warning(f"  ⚠️ {ticker} (single prediction): {e}")
                continue
    
    except Exception as e:
        logger.error(f"  ❌ {ticker}: {e}")

logger.info(f"\n✅ Updated {updated} predictions")
logger.info(f"⏭️ Skipped {skipped} predictions (no next trading day data yet)")

# Generate performance report
logger.info("\n3. Model Performance Summary:")
response = supabase.table('predictions').select('model_name, accuracy, percentage_error').not_.is_('actual_price', 'null').execute()
results = response.data

if results:
    df = pd.DataFrame(results)
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        direction_acc = (model_df['accuracy'].sum() / len(model_df)) * 100
        avg_error = model_df['percentage_error'].abs().mean()
        
        logger.info(f"\n{model}:")
        logger.info(f"  Direction Accuracy: {direction_acc:.1f}%")
        logger.info(f"  Avg Absolute Error: {avg_error:.2f}%")
        logger.info(f"  Total Verified: {len(model_df)}")
else:
    logger.info("No verified predictions yet!")
