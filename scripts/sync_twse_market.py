#!/usr/bin/env python3
"""
scripts/sync_twse_market.py - 每日台股全市場日 K 棒自動同步腳本 (Daily TWSE/TPEX Bulk Market Sync)

執行流程：
1. 透過 TWSE / TPEX OpenAPI 批量抓取當日上市與上櫃全部標的 OHLCV 與成交量數據。
2. 進行欄位清洗與資料型態標準化。
3. 寫入本地 DuckDB (tw_daily_bars)。
4. 輸出統計摘要報告。
"""

import sys
import os
import logging
import datetime

# 加入專案根目錄至 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.duckdb_manager import DuckDBManager
from data.market_universe import MarketUniverseSync
from data.universe_providers import TwseTpexProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tw_market_sync")


def main():
    logger.info("🚀 開始執行台股全市場日 K 棒排程同步作業...")
    start_time = datetime.datetime.now()

    db_mgr = DuckDBManager()
    if not db_mgr.enabled:
        logger.error("❌ DuckDB 未啟用，請檢查環境變數 ENABLE_DUCKDB")
        sys.exit(1)

    provider = TwseTpexProvider()
    sync = MarketUniverseSync(provider, db_manager=db_mgr)

    try:
        result = sync.refresh()
    except RuntimeError as exc:
        logger.error("❌ 台股清冊無法取得且沒有可用 cache: %s", exc)
        sys.exit(1)

    if result.stale:
        logger.warning(
            "⚠️ TWSE/TPEx 上游失敗，使用最後成功清冊 cache: %s 檔，snapshot=%s，原因=%s",
            result.count,
            result.snapshot_date,
            result.error,
        )
        return

    df = provider.last_snapshot_df
    if df.empty:
        logger.error("❌ 清冊成功但行情快照為空，拒絕寫入日 K")
        sys.exit(1)

    # 2. 寫入 DuckDB 日 K；清冊已由 MarketUniverseSync 寫入 market_universe
    records = df.to_dict(orient="records")
    inserted_count = db_mgr.save_daily_bars_batch(records)

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"✅ 全市場同步完成！共處理 {len(df)} 檔標的，成功寫入/更新 {inserted_count} 筆記錄 (耗時: {elapsed:.2f} 秒)")


if __name__ == "__main__":
    main()
