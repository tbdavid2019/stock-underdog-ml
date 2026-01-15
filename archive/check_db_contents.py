#!/usr/bin/env python3
"""
Check what data is actually in the database
"""
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("=== Checking Database Contents ===\n")

# Get all predictions
response = supabase.table('predictions').select('*').execute()
data = response.data

if not data:
    print("No predictions found!")
    exit(0)

print(f"Total predictions: {len(data)}\n")

# Check date range
timestamps = [datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) for p in data if p.get('timestamp')]
if timestamps:
    earliest = min(timestamps)
    latest = max(timestamps)
    print(f"Prediction date range:")
    print(f"  Earliest: {earliest.strftime('%Y-%m-%d')}")
    print(f"  Latest: {latest.strftime('%Y-%m-%d')}")
    print()

# Check backtest status
with_actual = sum(1 for p in data if p.get('actual_price') is not None)
without_actual = len(data) - with_actual

print(f"Backtest status:")
print(f"  With actual_price: {with_actual}")
print(f"  Without actual_price: {without_actual}")
print()

# Show sample
print("Sample predictions (first 5):")
for i, pred in enumerate(data[:5], 1):
    ts = pred.get('timestamp', 'N/A')
    if ts != 'N/A':
        ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).strftime('%Y-%m-%d')
    print(f"{i}. {pred.get('ticker')} - {pred.get('model_name')}")
    print(f"   Predicted on: {ts}")
    print(f"   Current: {pred.get('current_price')}, Predicted: {pred.get('predicted_price')}")
    print(f"   Actual: {pred.get('actual_price', 'NULL')}")
    print()
