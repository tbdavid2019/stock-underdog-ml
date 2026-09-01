"""
scripts/export_supabase_to_duckdb.py - 從 Supabase 匯出全部歷史預測數據至本地 DuckDB

分頁拉取 Supabase 'predictions' 表格的所有數據，建立完整的本地時序資料庫備份。
"""

import os
import sys
import logging

# 加入專案根目錄至 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SupabaseManager
from data.duckdb_manager import DuckDBManager
from core.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stock_app.sync")


def export_supabase_to_duckdb(batch_size: int = 1000):
    logger.info("=" * 65)
    logger.info("🚀 開始從 Supabase 導出數據至本地 DuckDB...")
    logger.info(f"📁 本地目標資料庫: {config.db.DUCKDB_PATH}")
    logger.info("=" * 65)

    supabase_mgr = SupabaseManager()
    if not supabase_mgr.enabled:
        logger.error("❌ Supabase 未啟用或未設定連線憑證！")
        return

    duckdb_mgr = DuckDBManager()

    offset = 0
    total_synced = 0

    while True:
        logger.info(f"📥 正在從 Supabase 抓取記錄 (Offset: {offset}, Limit: {batch_size})...")
        try:
            resp = supabase_mgr.client.table("predictions").select("*").range(offset, offset + batch_size - 1).execute()
            rows = resp.data
            if not rows:
                logger.info("🏁 已讀取完畢所有 Supabase 記錄。")
                break

            inserted = duckdb_mgr.save_predictions_batch(rows)
            total_synced += inserted
            logger.info(f"   ✓ 本批成功寫入 {inserted} 筆 (累計: {total_synced} 筆)")

            if len(rows) < batch_size:
                break
            offset += batch_size
        except Exception as e:
            logger.error(f"❌ 讀取 Supabase 失敗 (Offset: {offset}): {e}")
            break

    # 驗證總數
    total_in_duckdb = duckdb_mgr.get_row_count("predictions")
    logger.info("=" * 65)
    logger.info(f"🎉 數據同步完成！")
    logger.info(f"📊 本次從 Supabase 同步: {total_synced} 筆")
    logger.info(f"🦆 DuckDB 現有總記錄數: {total_in_duckdb} 筆")
    logger.info("=" * 65)

    # 示範查詢
    if total_in_duckdb > 0:
        logger.info("🔍 DuckDB 最近 5 筆預測預覽:")
        preview_df = duckdb_mgr.query("SELECT timestamp, index_name, model_name, ticker, current_price, potential FROM predictions ORDER BY timestamp DESC LIMIT 5")
        print(preview_df.to_string(index=False))


if __name__ == "__main__":
    export_supabase_to_duckdb()
