#!/usr/bin/env python3
"""
scripts/check_loader_health.py - 隔離無快取上游金絲雀健康探測道 CLI

用法：
  python scripts/check_loader_health.py
  python scripts/check_loader_health.py --force
  python scripts/check_loader_health.py --json
  python scripts/check_loader_health.py --sources twse,tpex,yfinance --force
"""

import argparse
import json
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader_canary import run_canary_checks


def main():
    parser = argparse.ArgumentParser(description="上游金絲雀健康探測 (Upstream Canary Health Lane)")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式輸出")
    parser.add_argument("--force", action="store_true", help="強制重新探測，忽略 30 分鐘冷卻快照")
    parser.add_argument("--sources", type=str, default="", help="指定探測資料源 (以逗號分隔，如 twse,yfinance)")
    parser.add_argument("--timeout", type=float, default=8.0, help="單一來源逾時秒數 (預設 8.0)")

    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None

    result = run_canary_checks(
        sources=sources,
        force=args.force,
        timeout_sec=args.timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return

    # 終端機格式化輸出
    is_cached_str = f" [快取中 - {result.get('age_seconds', 0)}s 前]" if result.get("is_cached") else " [即時探測]"
    print(f"\n📡 上游金絲雀健康巡檢報告{is_cached_str}")
    print(f"整體狀態: {result.get('status', 'unknown').upper()} | 檢查時間: {result.get('checked_at')}")
    print("-" * 75)
    print(f"{'資料源':<18} | {'健康狀態':<34} | {'延遲 (ms)':<10} | {'備註'}")
    print("-" * 75)

    for src_name, info in sorted(result.get("sources", {}).items()):
        status = info.get("status", "unknown")
        latency = f"{info.get('latency_ms', 0):.1f}"
        reason = info.get("sanitized_reason") or ""
        if len(reason) > 25:
            reason = reason[:22] + "..."

        status_display = {
            "healthy": "🟢 HEALTHY",
            "upstream_unreachable_or_ratelimited": "🔴 UNREACHABLE / 429",
            "schema_drift_parse_error": "🟡 SCHEMA DRIFT",
        }.get(status, status)

        print(f"{src_name:<18} | {status_display:<34} | {latency:<10} | {reason}")

    print("-" * 75)


if __name__ == "__main__":
    main()
