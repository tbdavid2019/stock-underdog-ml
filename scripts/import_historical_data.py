#!/usr/bin/env python3
"""
scripts/import_historical_data.py - 開源歷史數據集初始化回填腳本 (Historical Data Ingestion CLI)

使用方式:
    python scripts/import_historical_data.py --flows      # 匯入三大法人歷史進出數據 (TWSE + TPEX)
    python scripts/import_historical_data.py --klines     # 匯入台股核心標的 (0050, 2330 等) 歷史日 K 棒
    python scripts/import_historical_data.py --all        # 執行全部歷史回填
"""

import sys
import os
import argparse
import logging
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.historical_importer import HistoricalImporter
from data.duckdb_manager import DuckDBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("historical_import")


def main():
    parser = argparse.ArgumentParser(description="台股與美股開源歷史數據庫回填工具")
    parser.add_argument("--flows", action="store_true", help="匯入三大法人歷史買賣超 (tw-institutional-stocker)")
    parser.add_argument("--klines", action="store_true", help="匯入核心權值股歷史日 K (tw_stocker)")
    parser.add_argument("--all", action="store_true", help="匯入全部歷史數據庫")
    parser.add_argument("--max-rows", type=int, default=None, help="限制匯入最大筆數 (測試用)")

    args = parser.parse_args()

    if not (args.flows or args.klines or args.all):
        args.all = True

    start_time = datetime.datetime.now()
    logger.info("🚀 開始執行歷史開源數據集回填作業...")

    db_mgr = DuckDBManager()
    if not db_mgr.enabled:
        logger.error("❌ DuckDB 未啟用")
        sys.exit(1)

    total_flows = 0
    total_bars = 0

    if args.flows or args.all:
        logger.info("📊 正在匯入三大法人歷史時序數據...")
        total_flows = HistoricalImporter.import_institutional_flows(db_mgr=db_mgr, max_rows=args.max_rows)

    if args.klines or args.all:
        logger.info("📈 正在匯入核心標的歷史日 K 棒數據...")
        total_bars = HistoricalImporter.import_benchmark_k_lines(db_mgr=db_mgr)

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(
        f"🎉 歷史數據初始化回填完成！"
        f"法人時序: {total_flows} 筆, 歷史日 K: {total_bars} 筆 (總耗時: {elapsed:.2f} 秒)"
    )


if __name__ == "__main__":
    main()
