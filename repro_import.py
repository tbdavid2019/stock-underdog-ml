import database
print(f"HAS_SUPABASE: {database.HAS_SUPABASE}")

try:
    from supabase import create_client
    print("Direct import: SUCCESS")
except ImportError as e:
    print(f"Direct import: FAILED - {e}")
except Exception as e:
    print(f"Direct import: ERROR - {e}")
