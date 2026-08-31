#!/usr/bin/env python3
"""
Test Supabase with service_role key (full admin access)
"""
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    publishable_key = os.getenv("SUPABASE_KEY")

    key_to_use = service_key if service_key else publishable_key
    key_type = "SERVICE_ROLE" if service_key else "PUBLISHABLE"
    url = os.getenv("SUPABASE_URL")

    print(f"URL: {url}")
    print(f"Using {key_type} key")
    print(f"Key (first 25 chars): {key_to_use[:25] if key_to_use else 'None'}...")
    print()

    try:
        from supabase import create_client
        client = create_client(url, key_to_use)
        print("✅ Client created")
        
        # Test SELECT
        print("\n--- Testing SELECT ---")
        try:
            response = client.table("predictions").select("*").limit(5).execute()
            print(f"✅ SELECT successful! Found {len(response.data)} rows")
            if response.data:
                print(f"   Sample: {response.data[0]}")
        except Exception as e:
            print(f"❌ SELECT failed: {e}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")


if __name__ == "__main__":
    main()
