#!/usr/bin/env python3
"""Local forecast verification and legacy evidence dry-run; never contacts Supabase."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.forecast_store import ForecastStore
from backtest.verification import session_close, verify_forecast


def fetch_confirmed_bars(ticker, target_date):
    """Fetch the exact target session using the same adjusted-close basis as the pipeline."""
    import yfinance as yf
    target = date.fromisoformat(str(target_date)[:10])
    frame = yf.Ticker(ticker).history(start=target.isoformat(),
        end=(target + timedelta(days=1)).isoformat(), auto_adjust=True)
    return [{"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"]), "complete": True}
            for idx, row in frame.iterrows()]


def verify_due_forecasts(store, provider=fetch_confirmed_bars, *, now=None, dry_run=False):
    """Verify each due forecast once; missing target data remains pending."""
    now = now or datetime.now(timezone.utc)
    results = []
    bars_cache = {}
    for forecast in store.list_forecasts():
        forecast_id = forecast["forecast_id"]
        if any(a.get("status") == "settled" for a in store.list_actuals(forecast_id, include_invalidated=False)):
            continue
        target = date.fromisoformat(str(forecast["target_date"])[:10])
        if now < session_close(target, forecast["market"]).astimezone(timezone.utc):
            continue
        key = (forecast["ticker"], target.isoformat())
        try:
            if key not in bars_cache:
                bars_cache[key] = provider(*key)
        except Exception:
            if not dry_run:
                store.mark_pending(forecast_id, "provider_unavailable")
            result = {"status": "pending", "reason": "provider_unavailable"}
        else:
            result = verify_forecast(store, forecast_id, bars_cache[key], now=now,
                source="yfinance_adjusted_close", price_basis="adjusted_close", dry_run=dry_run)
        results.append({"forecast_id": forecast_id, **result})
    return results


def classify_legacy(rows):
    classified = []
    counts = {"legacy_unverified": 0, "replay": 0, "prospective": 0}
    for row in rows:
        if row.get("checkpoint_revision") and row.get("data_cutoff") and row.get("issued_at"):
            label = row.get("evidence_type") if row.get("evidence_type") in {"prospective", "replay"} else "replay"
        else:
            label = "legacy_unverified"
        counts[label] += 1
        classified.append({"forecast_id": row.get("forecast_id"), "ticker": row.get("ticker"), "evidence_type": label,
                           "eligible_for_trust_ranking": label == "prospective"})
    return {"counts": counts, "excluded_from_trust_ranking": counts["legacy_unverified"], "records": classified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB path")
    parser.add_argument("--forecast-id")
    parser.add_argument("--bars", help="JSON file containing daily bars")
    parser.add_argument("--legacy-dry-run", metavar="JSON", help="classify legacy forecast rows without writing")
    parser.add_argument("--dry-run", action="store_true", help="calculate the outcome without writing")
    parser.add_argument("--all-due", action="store_true", help="verify all due forecasts using the market-data provider")
    args = parser.parse_args()
    if args.legacy_dry_run:
        rows = json.loads(Path(args.legacy_dry_run).read_text(encoding="utf-8"))
        print(json.dumps(classify_legacy(rows), ensure_ascii=False, sort_keys=True))
        return 0
    if args.all_due:
        store = ForecastStore(args.db, read_only=args.dry_run)
        results = verify_due_forecasts(store, dry_run=args.dry_run)
        print(json.dumps({"count": len(results), "results": results}, ensure_ascii=False, sort_keys=True))
        return 0 if all(r["status"] != "pending" for r in results) else 1
    if not args.forecast_id or not args.bars:
        parser.error("--forecast-id and --bars are required for verification")
    store = ForecastStore(args.db, read_only=args.dry_run)
    bars = json.loads(Path(args.bars).read_text(encoding="utf-8"))
    result = verify_forecast(store, args.forecast_id, bars, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
