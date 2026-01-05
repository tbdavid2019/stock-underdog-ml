#!/usr/bin/env python3
"""
Direct Supabase connection test to diagnose PGRST205 error
"""
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"URL: {url}")
print(f"Key (first 20 chars): {key[:20] if key else 'None'}...")
print()

# Test 1: Check if supabase library is installed
try:
    from supabase import create_client
    print("✅ Supabase library imported successfully")
except ImportError as e:
    print(f"❌ Failed to import supabase: {e}")
    exit(1)

# Test 2: Create client
try:
    client = create_client(url, key)
    print("✅ Supabase client created successfully")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    exit(1)

# Test 3: Try to query the table (SELECT instead of INSERT)
print("\n--- Testing SELECT query ---")
try:
    response = client.table("predictions").select("*").limit(1).execute()
    print(f"✅ SELECT query successful!")
    print(f"   Data: {response.data}")
except Exception as e:
    print(f"❌ SELECT query failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Test 4: Try INSERT
print("\n--- Testing INSERT query ---")
try:
    test_data = {
        "index_name": "DEBUG_TEST",
        "model_name": "TEST",
        "ticker": "TEST",
        "current_price": 100.0,
        "predicted_price": 110.0,
        "potential": 0.1,
        "period": "test"
    }
    response = client.table("predictions").insert(test_data).execute()
    print(f"✅ INSERT successful!")
    print(f"   Response: {response.data}")
except Exception as e:
    print(f"❌ INSERT failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Test 5: List all tables (if possible with REST API)
print("\n--- Attempting to introspect schema ---")
try:
    # Try to access a system table or use RPC
    response = client.rpc("version").execute()
    print(f"✅ RPC call successful: {response.data}")
except Exception as e:
    print(f"⚠️  RPC introspection not available: {e}")
