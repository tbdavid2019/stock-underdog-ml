#!/usr/bin/env python3
"""
Direct Supabase connection test to diagnose PGRST205 error
"""
from dotenv import load_dotenv
import os


def main():
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
        return

    # Test 2: Create client
    try:
        client = create_client(url, key)
        print("✅ Supabase client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return

    # Test 3: Try to query the table (SELECT instead of INSERT)
    print("\n--- Testing SELECT query ---")
    try:
        response = client.table("predictions").select("*").limit(1).execute()
        print(f"✅ SELECT query successful!")
        print(f"   Data: {response.data}")
    except Exception as e:
        print(f"❌ SELECT query failed: {e}")


if __name__ == "__main__":
    main()
