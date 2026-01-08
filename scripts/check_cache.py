#!/usr/bin/env python3
"""
Check stock list cache status
"""
from stock_cache import get_cache_status

print("=== Stock List Cache Status ===\n")

status = get_cache_status()

if not status:
    print("No cache found. Will fetch from API on first run.\n")
else:
    for index_name, info in status.items():
        print(f"{index_name}:")
        print(f"  Stock Count: {info['stock_count']}")
        print(f"  Age: {info['age_days']} days")
        print(f"  Status: {'⚠️ Expired' if info['expired'] else '✅ Fresh'}")
        print(f"  Last Updated: {info['timestamp']}")
        print()
