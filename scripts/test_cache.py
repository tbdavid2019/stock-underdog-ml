#!/usr/bin/env python3
"""
Test cache functionality
"""
import sys
sys.path.insert(0, '/home/ec2-user/stock-underdog-ml')

from data_loader import get_tw0050_stocks, get_sp500_stocks

print("=== Testing Stock List Cache ===\n")

# Test 1: First fetch (should hit API)
print("Test 1: First fetch (should create cache)")
print("-" * 50)
tw50 = get_tw0050_stocks()
print(f"TW50: {len(tw50)} stocks")
print(f"Sample: {tw50[:5]}")
print()

# Test 2: Second fetch (should use cache)
print("Test 2: Second fetch (should use cache)")
print("-" * 50)
tw50_cached = get_tw0050_stocks()
print(f"TW50: {len(tw50_cached)} stocks")
print(f"Same result: {tw50 == tw50_cached}")
print()

# Test 3: Different index
print("Test 3: Fetch SP500")
print("-" * 50)
sp500 = get_sp500_stocks()
print(f"SP500: {len(sp500)} stocks")
print(f"Sample: {sp500[:5]}")
print()

# Test 4: Check cache status
print("Test 4: Cache status")
print("-" * 50)
from stock_cache import get_cache_status
status = get_cache_status()
for index_name, info in status.items():
    print(f"{index_name}:")
    print(f"  Stocks: {info['stock_count']}")
    print(f"  Age: {info['age_days']} days")
    print(f"  Fresh: {'✅' if not info['expired'] else '❌'}")
print()

print("✅ Cache test complete!")
