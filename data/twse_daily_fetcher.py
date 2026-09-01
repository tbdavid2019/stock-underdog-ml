"""
data/twse_daily_fetcher.py - 台股全市場日行情與法人批量抓取器 (TWSE/TPEX Daily Bulk Fetcher)

使用台灣證交所 (TWSE) 與櫃買中心 (TPEX) 官方 OpenAPI，
單次請求即可在 1~2 秒內抓取全市場 1,800+ 檔上市櫃股票之日 K 棒與成交量，
並自動清洗寫入本地 DuckDB / Parquet 時序資料庫，提供 yfinance 限流時的零延遲備援。
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any
import requests
import pandas as pd

from core.config import config
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.twse_fetcher")


class TWSEDailyFetcher:
    """TWSE / TPEX 全市場日行情批量抓取器"""

    TWSE_OPENAPI_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    TPEX_OPENAPI_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    @staticmethod
    def roc_date_to_iso(roc_date_str: str) -> str:
        """
        將民國年月日 (e.g. 1150831 or 115/08/31) 轉換為標準 ISO YYYY-MM-DD
        """
        s = str(roc_date_str).replace("/", "").strip()
        if len(s) >= 7:
            roc_year = int(s[:-4])
            month = s[-4:-2]
            day = s[-2:]
            year = roc_year + 1911
            return f"{year:04d}-{month}-{day}"
        elif len(s) == 6:
            roc_year = int(s[:-4])
            month = s[-4:-2]
            day = s[-2:]
            year = roc_year + 1911
            return f"{year:04d}-{month}-{day}"
        return datetime.date.today().strftime("%Y-%m-%d")

    @classmethod
    def fetch_twse_quotes(cls, timeout: int = 15) -> pd.DataFrame:
        """
        抓取 TWSE 上市股票全市場當日收盤行情
        """
        try:
            resp = requests.get(cls.TWSE_OPENAPI_URL, headers=cls.HEADERS, timeout=timeout)
            if resp.status_code != 200:
                logger.warning(f"⚠️ TWSE OpenAPI 回應異常: HTTP {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            if not isinstance(data, list) or not data:
                return pd.DataFrame()

            rows = []
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for item in data:
                try:
                    code = str(item.get("Code", "")).strip()
                    if not code:
                        continue
                    
                    # 規格化 ticker (上市股票加上 .TW)
                    ticker = f"{code}.TW" if not code.endswith(".TW") else code
                    name = str(item.get("Name", "")).strip()
                    raw_date = item.get("Date", "")
                    date_iso = cls.roc_date_to_iso(raw_date)

                    def safe_float(val: Any) -> float:
                        if val is None or val == "" or val == "--":
                            return 0.0
                        return float(str(val).replace(",", "").strip())

                    def safe_int(val: Any) -> int:
                        if val is None or val == "" or val == "--":
                            return 0
                        return int(float(str(val).replace(",", "").strip()))

                    open_p = safe_float(item.get("OpeningPrice"))
                    high_p = safe_float(item.get("HighestPrice"))
                    low_p = safe_float(item.get("LowestPrice"))
                    close_p = safe_float(item.get("ClosingPrice"))
                    vol = safe_int(item.get("TradeVolume"))
                    val = safe_float(item.get("TradeValue"))
                    tx_cnt = safe_int(item.get("Transaction"))

                    rows.append({
                        "date": date_iso,
                        "ticker": ticker,
                        "raw_code": code,
                        "name": name,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": vol,
                        "trade_value": val,
                        "transaction_count": tx_cnt,
                        "market": "TWSE",
                        "created_at": now_str,
                    })
                except Exception as e:
                    logger.debug(f"解析 TWSE 個股 {item.get('Code')} 出錯: {e}")
                    continue

            df = pd.DataFrame(rows)
            logger.info(f"✅ 成功獲取 TWSE 上市全市場行情: {len(df)} 檔")
            return df

        except Exception as e:
            logger.error(f"❌ 抓取 TWSE OpenAPI 失敗: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_tpex_quotes(cls, timeout: int = 15) -> pd.DataFrame:
        """
        抓取 TPEX 上櫃股票全市場當日收盤行情
        """
        try:
            resp = requests.get(cls.TPEX_OPENAPI_URL, headers=cls.HEADERS, timeout=timeout)
            if resp.status_code != 200:
                logger.warning(f"⚠️ TPEX OpenAPI 回應異常: HTTP {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            if not isinstance(data, list) or not data:
                return pd.DataFrame()

            rows = []
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for item in data:
                try:
                    code = str(item.get("SecuritiesCompanyCode", "")).strip()
                    if not code:
                        continue

                    # 規格化 ticker (上櫃股票加上 .TWO)
                    ticker = f"{code}.TWO" if not code.endswith(".TWO") else code
                    name = str(item.get("CompanyName", "")).strip()
                    raw_date = item.get("Date", "")
                    date_iso = cls.roc_date_to_iso(raw_date)

                    def safe_float(val: Any) -> float:
                        if val is None or val == "" or val == "--":
                            return 0.0
                        return float(str(val).replace(",", "").strip())

                    def safe_int(val: Any) -> int:
                        if val is None or val == "" or val == "--":
                            return 0
                        return int(float(str(val).replace(",", "").strip()))

                    open_p = safe_float(item.get("Open"))
                    high_p = safe_float(item.get("High"))
                    low_p = safe_float(item.get("Low"))
                    close_p = safe_float(item.get("Close"))
                    vol = safe_int(item.get("TradingShares"))
                    val = safe_float(item.get("TransactionAmount"))
                    tx_cnt = safe_int(item.get("TransactionNumber"))

                    rows.append({
                        "date": date_iso,
                        "ticker": ticker,
                        "raw_code": code,
                        "name": name,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": vol,
                        "trade_value": val,
                        "transaction_count": tx_cnt,
                        "market": "TPEX",
                        "created_at": now_str,
                    })
                except Exception as e:
                    logger.debug(f"解析 TPEX 個股 {item.get('SecuritiesCompanyCode')} 出錯: {e}")
                    continue

            df = pd.DataFrame(rows)
            logger.info(f"✅ 成功獲取 TPEX 上櫃全市場行情: {len(df)} 檔")
            return df

        except Exception as e:
            logger.error(f"❌ 抓取 TPEX OpenAPI 失敗: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_full_market_snapshot(cls) -> pd.DataFrame:
        """
        同時抓取 TWSE 與 TPEX，合成完整台股全市場當日行情 DataFrame
        """
        df_twse = cls.fetch_twse_quotes()
        df_tpex = cls.fetch_tpex_quotes()

        dfs = [d for d in [df_twse, df_tpex] if not d.empty]
        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"📊 台股全市場日行情彙總完成: 共 {len(full_df)} 檔標的 (TWSE: {len(df_twse)}, TPEX: {len(df_tpex)})")
        return full_df

    @classmethod
    def sync_daily_market_to_duckdb(cls, db_manager: Optional[DuckDBManager] = None) -> int:
        """
        抓取全市場行情並寫入 DuckDB
        """
        df = cls.fetch_full_market_snapshot()
        if df.empty:
            logger.warning("⚠️ 全市場行情數據為空，略過寫入")
            return 0

        mgr = db_manager or DuckDBManager()
        inserted = mgr.save_daily_bars_batch(df.to_dict(orient="records"))
        logger.info(f"💾 成功同步 {inserted} 筆台股全市場日 K 棒至 DuckDB")
        return inserted
