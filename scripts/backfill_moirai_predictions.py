"""
Salesforce MOIRAI-2 Historical Predictions Backfill Script.
Fills missing moirai_predicted_price, moirai_potential, and quantiles into DuckDB predictions
for historical dates where TimesFM / Timestral ran, enabling complete PK scorecards and trajectories.
"""
import argparse
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.duckdb_manager import DuckDBManager
from data.fetcher import StockFetcher
from models.moirai_model import get_moirai_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_moirai")


def backfill_moirai(ticker_filter: Optional[str] = None, days: int = 14, dry_run: bool = False):
    db = DuckDBManager()
    if not db.enabled:
        logger.error("DuckDB is disabled or unavailable.")
        return

    logger.info(f"🚀 啟動 MOIRAI-2 歷史預測回填工具 (回溯天數: {days}, 標的過濾: {ticker_filter or '全部'})...")

    # 1. 查詢需要回填的 ticker 與日期 (依 ticker, dt 分組，聚合指數與記錄時間，防止跨指數重複運算)
    sql = """
        SELECT ticker,
               SUBSTRING(timestamp, 1, 10) as dt,
               MAX(index_name) as index_name,
               MAX(timestamp) as record_ts,
               AVG(current_price) as stored_curr_price
        FROM predictions
        WHERE moirai_predicted_price IS NULL
          AND TRY_CAST(SUBSTRING(timestamp, 1, 10) AS DATE) >= CURRENT_DATE - CAST(? AS INTEGER)
    """
    params = [days]
    if ticker_filter:
        sql += " AND ticker = ?"
        params.append(ticker_filter)

    sql += " GROUP BY ticker, dt ORDER BY ticker, dt DESC"

    with db._get_connection(read_only=True) as conn:
        df_pending = conn.execute(sql, params).df()

    if df_pending.empty:
        logger.info("✅ 沒有發現需要回填 MOIRAI-2 預測的資料記錄。")
        return

    logger.info(f"📊 共有 {len(df_pending)} 筆待回填記錄 (涵蓋 {df_pending['ticker'].nunique()} 檔標的)")

    # 2. 載入 MOIRAI 模型
    model = get_moirai_model()
    if not model.load_model():
        logger.error("❌ MOIRAI-2 模型載入失敗，終止回填作業。")
        return

    # 3. 按股票分組抓取日 K 線與推論
    by_ticker = df_pending.groupby("ticker")
    total_updated = 0
    total_inserted = 0

    for ticker, group in by_ticker:
        logger.info(f"▶ 正在處理標的: {ticker} (共 {len(group)} 個預測交易日)...")
        # 抓取 1 年歷史日 K 線
        hist_df = StockFetcher.get_stock_data(ticker, period="1y", use_cache=True)
        if hist_df is None or hist_df.empty or "Close" not in hist_df.columns:
            logger.warning(f"⚠️ 無法取得 {ticker} 歷史日 K 線，略過")
            continue

        # 統一去除時區以確保日期切片 (如 .loc[:'2026-11-03']) 穩定無誤
        if hist_df.index.tz is not None:
            hist_df.index = hist_df.index.tz_localize(None)

        close_series = hist_df["Close"].dropna()
        if len(close_series) < 10:
            logger.warning(f"⚠️ {ticker} 歷史日 K 線筆數過少 ({len(close_series)})，略過")
            continue

        for _, row in group.iterrows():
            dt_str = row["dt"]
            idx_name = row["index_name"]
            record_ts = row["record_ts"]
            stored_curr_price = float(row["stored_curr_price"] or 0.0)

            # 抓取該預測日之前的最後收盤價
            series_prior = close_series.loc[:dt_str]
            if len(series_prior) == 0:
                continue

            # 確定推論使用的截斷點：比對 stored_curr_price
            matched_slice = None
            # 優先比對 close_series 中收盤價等於 stored_curr_price 的日期
            for sub_len in range(len(series_prior), max(0, len(series_prior) - 5), -1):
                sub = series_prior.iloc[:sub_len]
                if abs(float(sub.iloc[-1]) - stored_curr_price) < 0.01:
                    matched_slice = sub
                    break

            if matched_slice is None:
                # 找不到完全符合現價的 bar，若 series_prior 最後一根日期就是 dt_str 且包含未收盤，往前回退一根
                if len(series_prior) > 1 and str(series_prior.index[-1])[:10] == dt_str:
                    matched_slice = series_prior.iloc[:-1]
                else:
                    matched_slice = series_prior

            if len(matched_slice) < 10:
                continue

            # 執行 MOIRAI-2 推論
            curr_p = float(matched_slice.iloc[-1])
            s_arr = matched_slice.to_numpy(dtype=float)
            try:
                forecast_res = model.forecast_batch({ticker: s_arr}, horizon=5)
                forecast = forecast_res.get(ticker)
                if not forecast:
                    continue

                pred_p = float(forecast["predicted_price"])
                pot = float(forecast["potential"])
                quantiles = forecast.get("quantiles", {})
                p10 = float(quantiles.get("p10")) if quantiles.get("p10") is not None else None
                p90 = float(quantiles.get("p90")) if quantiles.get("p90") is not None else None

                if dry_run:
                    logger.info(f"   [DRY-RUN] {ticker} ({dt_str}): 現價 {curr_p:.2f} -> 目標 {pred_p:.2f} ({pot:+.2f}%), P10: {p10}, P90: {p90}")
                    continue

                # 更新 DuckDB predictions
                with db._get_connection() as conn:
                    # 1. 更新既有 rows
                    conn.execute("""
                        UPDATE predictions
                        SET moirai_predicted_price = ?,
                            moirai_potential = ?,
                            moirai_p10 = ?,
                            moirai_p90 = ?
                        WHERE ticker = ? AND SUBSTRING(timestamp, 1, 10) = ?
                    """, [pred_p, pot, p10, p90, ticker, dt_str])
                    total_updated += 1

                    # 2. 檢查是否需要插入獨立的 MOIRAI-2 策略列
                    moi_row_count = conn.execute("""
                        SELECT COUNT(*) FROM predictions
                        WHERE ticker = ? AND SUBSTRING(timestamp, 1, 10) = ? AND model_name = 'MOIRAI-2'
                    """, [ticker, dt_str]).fetchone()[0]

                    if moi_row_count == 0:
                        moi_tag = "MOIRAI看漲 | MOIRAI強" if pot > 0 else "MOIRAI看跌"
                        # 插入 MOIRAI-2 獨立列
                        conn.execute("""
                            INSERT INTO predictions (
                                index_name, model_name, strategy_type, ticker, current_price,
                                predicted_price, potential, period, timestamp, tags,
                                moirai_predicted_price, moirai_potential, moirai_p10, moirai_p90,
                                dual_model_consensus, tri_model_consensus
                            ) VALUES (
                                ?, 'MOIRAI-2', 'MOIRAI預測', ?, ?,
                                ?, ?, '6mo', ?, ?,
                                ?, ?, ?, ?,
                                FALSE, FALSE
                            )
                        """, [
                            idx_name, ticker, curr_p,
                            pred_p, pot, record_ts, moi_tag,
                            pred_p, pot, p10, p90
                        ])
                        total_inserted += 1

                logger.info(f"   ✓ [回填成功] {ticker} ({dt_str}): 現價 ${curr_p:,.2f} -> 目標 ${pred_p:,.2f} ({pot:+.2f}%)")

            except Exception as e:
                logger.error(f"❌ 推論或寫入失敗 ({ticker} {dt_str}): {e}")

    logger.info("=" * 60)
    logger.info(f"🎉 MOIRAI-2 歷史回填完成！更新記錄: {total_updated} 筆，補入獨立模型列: {total_inserted} 筆。")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回填 MOIRAI-2 歷史預測與分位數數據")
    parser.add_argument("--ticker", "-t", help="指定單一股票代號 (例如 6770.T 或 2330.TW)")
    parser.add_argument("--days", "-d", type=int, default=14, help="回溯天數 (預設: 14 天)")
    parser.add_argument("--dry-run", action="store_true", help="乾跑模式 (不寫入資料庫)")
    args = parser.parse_args()

    backfill_moirai(ticker_filter=args.ticker, days=args.days, dry_run=args.dry_run)
