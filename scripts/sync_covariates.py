#!/usr/bin/env python3
"""Retryable five-year historical backfill and daily incremental covariate sync."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from data.covariates import COVARIATE_REGISTRY, CovariateSynchronizer, LiveSourceProvider
from data.covariate_store import CovariateStore


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", help="comma-separated registry IDs (default: every fetchable registered series)")
    parser.add_argument("--start", type=date.fromisoformat, help="inclusive start date; defaults to five years ago")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(), help="inclusive end date; defaults to today")
    parser.add_argument("--daily", action="store_true", help="sync each series from its latest stored observation plus one day")
    parser.add_argument("--db", help="DuckDB path; defaults to configured local DuckDB")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--list-sources", action="store_true", help="print source registry and exit")
    parser.add_argument("--verbose", action="store_true")
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,format="%(levelname)s %(message)s")
    if args.list_sources:
        print(json.dumps({k:v.__dict__ for k,v in COVARIATE_REGISTRY.items()},ensure_ascii=False,indent=2))
        return 0
    if args.daily and args.start:
        parser.error("--daily chooses each series start; omit --start")
    start=args.start or args.end.replace(year=args.end.year-5)
    if start > args.end: parser.error("--start must be on or before --end")
    selected=args.series.split(",") if args.series else [k for k,v in COVARIATE_REGISTRY.items() if v.status!="blocked" and k not in ("industry","OHLCV","TW_INSTITUTIONAL")]
    for sid in selected:
        if sid not in COVARIATE_REGISTRY and not sid.startswith(("OHLCV:","TW_INSTITUTIONAL:")):
            parser.error(f"unknown series {sid}; use --list-sources")
    store=CovariateStore(args.db)
    sync=CovariateSynchronizer(store,LiveSourceProvider(),args.retries)
    if args.daily:
        reports=[]
        for sid in selected:
            latest=store.latest_observation_date(sid)
            series_start=(latest+timedelta(days=1)) if latest else start
            reports.append(sync.sync([sid],series_start,args.end))
        report={"runs":reports,"inserted":sum(r["inserted"] for r in reports),
                "fetched":{k:v for r in reports for k,v in r["fetched"].items()},
                "failed":{k:v for r in reports for k,v in r["failed"].items()}}
    else:
        report=sync.sync(selected,start,args.end)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
