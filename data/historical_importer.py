"""
data/historical_importer.py - 歷史開源數據集匯入與回填器 (Historical Data Importer)

支援從以下開源數據倉庫匯入台股與美股深層歷史數據至 DuckDB：
1. voidful/tw-institutional-stocker: 台股上市/上櫃 三大法人每日淨買賣超與外資持股比率歷史表。
2. voidful/tw_stocker: 台股權值股與 ETF (0050, 0051, 2330 等) 歷史 OHLCV K 棒數據。
3. voidful/us_fddk: 美股 20 年 SPY, QQQ, VIX 宏觀指標。
"""

import os
import logging
import io
import datetime
from typing import Dict, List, Optional, Any
import requests
import pandas as pd

from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.importer")


class HistoricalImporter:
    """歷史開源數據集匯入器"""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # 1. 法人籌碼來源 URL
    TWSE_FLOWS_URL = "https://raw.githubusercontent.com/voidful/tw-institutional-stocker/main/data/twse_flows.csv"
    TPEX_FLOWS_URL = "https://raw.githubusercontent.com/voidful/tw-institutional-stocker/main/data/tpex_flows.csv"
    TWSE_FOREIGN_URL = "https://raw.githubusercontent.com/voidful/tw-institutional-stocker/main/data/twse_foreign.csv"
    TPEX_FOREIGN_URL = "https://raw.githubusercontent.com/voidful/tw-institutional-stocker/main/data/tpex_foreign.csv"

    # 2. 台股歷史 K 棒基底 URL
    TW_STOCKER_RAW_BASE = "https://raw.githubusercontent.com/voidful/tw_stocker/main/data"

    @classmethod
    def import_institutional_flows(
        cls, 
        db_mgr: Optional[DuckDBManager] = None, 
        max_rows: Optional[int] = None
    ) -> int:
        """
        匯入上市與上櫃三大法人歷史買賣超數據 (twse_flows.csv & tpex_flows.csv)
        """
        mgr = db_mgr or DuckDBManager()
        if not mgr.enabled:
            logger.warning("DuckDB 未啟用，略過匯入")
            return 0

        total_imported = 0
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for source_name, url, market in [
            ("TWSE 法人買賣超", cls.TWSE_FLOWS_URL, "TWSE"),
            ("TPEX 法人買賣超", cls.TPEX_FLOWS_URL, "TPEX"),
        ]:
            try:
                logger.info(f"📥 正在下載並解析 {source_name}...")
                resp = requests.get(url, headers=cls.HEADERS, timeout=60)
                if resp.status_code != 200:
                    logger.warning(f"⚠️ 下載 {source_name} 失敗 (HTTP {resp.status_code})")
                    continue

                df = pd.read_csv(io.StringIO(resp.text), nrows=max_rows)
                if df.empty:
                    continue

                # 欄位 mapping
                # 原 CSV 欄位: date,code,name,foreign_net,trust_net,dealer_net,market
                df["raw_code"] = df["code"].astype(str).str.strip()
                suffix = ".TW" if market == "TWSE" else ".TWO"
                df["ticker"] = df["raw_code"].apply(lambda c: f"{c}{suffix}" if not c.endswith(suffix) else c)
                df["total_net"] = df["foreign_net"].fillna(0) + df["trust_net"].fillna(0) + df["dealer_net"].fillna(0)
                df["foreign_ratio"] = 0.0
                df["total_shares"] = 0
                df["foreign_shares"] = 0
                df["market"] = market
                df["created_at"] = now_str

                # 分批寫入 DuckDB
                batch_size = 50000
                for i in range(0, len(df), batch_size):
                    chunk = df.iloc[i : i + batch_size].to_dict(orient="records")
                    count = mgr.save_institutional_flows_batch(chunk)
                    total_imported += count

                logger.info(f"✅ {source_name} 匯入完成: 共 {len(df)} 筆")

            except Exception as e:
                logger.error(f"❌ 匯入 {source_name} 時發生異常: {e}")

        return total_imported

    @classmethod
    def import_benchmark_k_lines(
        cls, 
        tickers: Optional[List[str]] = None, 
        db_mgr: Optional[DuckDBManager] = None
    ) -> int:
        """
        匯入特定核心標的 (如 0050, 0051, 2330 等) 之歷史 K 棒至 tw_daily_bars
        """
        mgr = db_mgr or DuckDBManager()
        if not mgr.enabled:
            return 0

        target_tickers = tickers or [
            "0050", "0051", "0056", "2330", "2317", "2454", "2382", "3231", "2308", "2412"
        ]

        total_bars = 0
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for code in target_tickers:
            url = f"{cls.TW_STOCKER_RAW_BASE}/{code}.csv"
            try:
                logger.info(f"📥 下載 {code} 歷史行情 ({url})...")
                resp = requests.get(url, headers=cls.HEADERS, timeout=30)
                if resp.status_code != 200:
                    logger.debug(f"跳過 {code}: HTTP {resp.status_code}")
                    continue

                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty:
                    continue

                # 處理日期欄位 (Date 或 Datetime)
                date_col = "Datetime" if "Datetime" in df.columns else "Date" if "Date" in df.columns else None
                if not date_col:
                    continue

                df["date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%d")
                
                # 若為日內 5分K，聚合為日 K 棒
                agg_dict = {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
                daily_df = df.groupby("date").agg(agg_dict).reset_index()

                ticker_full = f"{code}.TW"
                daily_df["ticker"] = ticker_full
                daily_df["raw_code"] = code
                daily_df["name"] = code
                daily_df["open"] = daily_df["Open"].astype(float)
                daily_df["high"] = daily_df["High"].astype(float)
                daily_df["low"] = daily_df["Low"].astype(float)
                daily_df["close"] = daily_df["Close"].astype(float)
                daily_df["volume"] = daily_df["Volume"].astype(int)
                daily_df["trade_value"] = daily_df["close"] * daily_df["volume"]
                daily_df["transaction_count"] = 0
                daily_df["market"] = "TWSE"
                daily_df["created_at"] = now_str

                records = daily_df[[
                    "date", "ticker", "raw_code", "name",
                    "open", "high", "low", "close", "volume",
                    "trade_value", "transaction_count", "market", "created_at"
                ]].to_dict(orient="records")

                inserted = mgr.save_daily_bars_batch(records)
                total_bars += inserted
                logger.info(f"✅ 成功匯入 {code} 歷史日 K 棒: {inserted} 筆")

            except Exception as e:
                logger.warning(f"⚠️ 匯入 {code} 歷史 K 棒失敗: {e}")

        return total_bars
