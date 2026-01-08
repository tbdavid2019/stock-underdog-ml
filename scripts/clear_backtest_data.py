#!/usr/bin/env python3
"""
Clear backtest data using Supabase service_role key
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("=== Clearing Invalid Backtest Data ===\n")

# Get current status
response = supabase.table('predictions').select('id, actual_price').execute()
total = len(response.data)
with_actual = sum(1 for p in response.data if p.get('actual_price') is not None)

print(f"Current status:")
print(f"  Total predictions: {total}")
print(f"  With actual_price: {with_actual}")
print(f"  Without actual_price: {total - with_actual}\n")

if with_actual == 0:
    print("✅ No backtest data to clear!")
    exit(0)

# Clear backtest fields
print(f"Clearing {with_actual} backtest records...")

# Use RPC or direct update
# Note: Supabase Python client doesn't support UPDATE without WHERE easily
# So we'll update all records
try:
    # Get all IDs with actual_price
    ids_to_update = [p['id'] for p in response.data if p.get('actual_price') is not None]
    
    # Update in batches
    batch_size = 100
    updated = 0
    
    for i in range(0, len(ids_to_update), batch_size):
        batch_ids = ids_to_update[i:i+batch_size]
        
        for record_id in batch_ids:
            supabase.table('predictions').update({
                'actual_price': None,
                'actual_date': None,
                'accuracy': None,
                'absolute_error': None,
                'percentage_error': None
            }).eq('id', record_id).execute()
            updated += 1
        
        print(f"  Updated {updated}/{len(ids_to_update)}...")
    
    print(f"\n✅ Cleared {updated} backtest records")
    
    # Verify
    response = supabase.table('predictions').select('id, actual_price').execute()
    remaining = sum(1 for p in response.data if p.get('actual_price') is not None)
    print(f"\nVerification:")
    print(f"  Remaining with actual_price: {remaining}")
    
except Exception as e:
    print(f"❌ Error: {e}")
