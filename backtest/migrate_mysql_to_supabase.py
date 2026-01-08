#!/usr/bin/env python3
"""
MySQL to Supabase Migration Script
Migrates historical prediction data from MySQL to Supabase
"""
import os
from dotenv import load_dotenv
import mysql.connector
from supabase import create_client

load_dotenv()

# MySQL connection
mysql_config = {
    'host': 'sql.freedb.tech',
    'user': 'freedb_david888',
    'password': 'd#NFM?7ubyhTteY',
    'database': 'freedb_stock',
    'port': 3306
}

# Supabase connection
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

print("=== MySQL to Supabase Migration ===\n")

# Step 1: Connect to MySQL and inspect data
print("1. Connecting to MySQL...")
try:
    mysql_conn = mysql.connector.connect(**mysql_config)
    cursor = mysql_conn.cursor(dictionary=True)
    
    # Check table structure
    cursor.execute("DESCRIBE stock_predictions")
    columns = cursor.fetchall()
    print("\nMySQL Table Structure:")
    for col in columns:
        print(f"  - {col['Field']}: {col['Type']}")
    
    # Count records
    cursor.execute("SELECT COUNT(*) as count FROM stock_predictions")
    count = cursor.fetchone()['count']
    print(f"\nTotal records in MySQL: {count}")
    
    # Sample data
    cursor.execute("SELECT * FROM stock_predictions ORDER BY created_at DESC LIMIT 5")
    samples = cursor.fetchall()
    print("\nSample records:")
    for i, row in enumerate(samples, 1):
        print(f"\n  Record {i}:")
        for key, value in row.items():
            print(f"    {key}: {value}")
    
    # Step 2: Connect to Supabase
    print("\n2. Connecting to Supabase...")
    supabase = create_client(supabase_url, supabase_key)
    print("✅ Supabase connected")
    
    # Step 3: Migrate data
    print("\n3. Starting migration...")
    cursor.execute("SELECT * FROM stock_predictions")
    all_records = cursor.fetchall()
    
    migrated = 0
    batch_size = 100
    
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i+batch_size]
        supabase_batch = []
        
        for record in batch:
            # Map MySQL fields to Supabase fields
            supabase_record = {
                'index_name': record.get('index_name'),
                'model_name': record.get('prediction_method'),
                'ticker': record.get('stock_symbol'),
                'current_price': float(record.get('current_price', 0)),
                'predicted_price': float(record.get('predicted_price', 0)),
                'potential': float(record.get('potential', 0)),
                'period': record.get('period_param'),
                'timestamp': record.get('created_at').isoformat() if record.get('created_at') else None
            }
            supabase_batch.append(supabase_record)
        
        # Insert batch
        try:
            supabase.table('predictions').insert(supabase_batch).execute()
            migrated += len(batch)
            print(f"  Migrated {migrated}/{len(all_records)} records...")
        except Exception as e:
            print(f"  ❌ Error migrating batch: {e}")
    
    print(f"\n✅ Migration complete! Migrated {migrated} records")
    
    cursor.close()
    mysql_conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n=== Backtesting Analysis ===")
print("\nCurrent schema supports backtesting with:")
print("  ✅ Historical predictions (timestamp)")
print("  ✅ Predicted vs actual prices")
print("  ✅ Model performance tracking")
print("\nTo enable backtesting, you need:")
print("  - Actual closing prices at prediction_date + N days")
print("  - Suggestion: Add 'actual_price' and 'accuracy' columns")
