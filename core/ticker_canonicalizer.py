"""
core/ticker_canonicalizer.py - 標的代號正規化與 Universe 白名單校驗器

統一處理台股與美股代號格式：
- 台股：切除市場後綴 (.TW, .TWO)，保留純數字代號 (例: "2330.TW" -> "2330", "6442.TWO" -> "6442")
- 美股：轉為大寫並切除空白 (例: "nvda" -> "NVDA", "googl" -> "GOOGL")
- 市場識別：純數字為台股 (TW)，英文字母為美股 (US)
- 白名單檢驗：與 DuckDB 中的股票池 (Universe) 進行嚴格白名單校驗
"""

import logging
import os
import re
from typing import Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("stock_app.ticker_canonicalizer")


def canonicalize_ticker(raw_ticker: str) -> Optional[str]:
    """
    將傳入的股票代號規格化：
    - 去除前後空白
    - 切除 .TW, .TWO, .ROCS, .TWO 等台股後綴
    - 若為純數字字串 (台股)，保留純數字
    - 若為純英文字母 (美股)，轉為全大寫
    - 其餘非法字元返回 None
    """
    if not raw_ticker or not isinstance(raw_ticker, str):
        return None

    cleaned = raw_ticker.strip()
    if not cleaned:
        return None

    # 切除常見市場後綴
    cleaned_upper = cleaned.upper()
    for suffix in [".TW", ".TWO", ".ROCS", ".TT", ".TP"]:
        if cleaned_upper.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break

    # 1. 判斷是否為台股代碼 (通常為 4 到 6 碼數字)
    if re.fullmatch(r"\d{4,6}", cleaned):
        return cleaned

    # 2. 判斷是否為美股代碼 (1 到 5 碼英文字母，可含點號如 BRK.B)
    cleaned_us = cleaned.upper()
    if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", cleaned_us):
        return cleaned_us

    # 不符合標準代號規範
    return None


def detect_market(ticker: str) -> str:
    """
    識別市場：
    - 純數字 (或正規化為台股代碼) -> 'TW' (台股)
    - 英文字母 (或正規化為美股代碼) -> 'US' (美股)
    - 其他 -> 'UNKNOWN'
    """
    if not ticker:
        return "UNKNOWN"
    canon = canonicalize_ticker(ticker)
    if not canon:
        return "UNKNOWN"
    if re.fullmatch(r"\d{4,6}", canon):
        return "TW"
    if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", canon):
        return "US"
    return "UNKNOWN"


class TickerUniverseValidator:
    """
    股票池白名單檢驗器，支援自 DuckDB、快取或系統核心配置載入 Universe 白名單。
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._cached_whitelist: Optional[Set[str]] = None

    def get_whitelist(self, force_refresh: bool = False) -> Set[str]:
        """
        取得系統當前可用的 Universe 白名單集合 (皆為 canonical 代號)。
        包含 DuckDB 既有宇宙與系統核心配置。
        """
        if self._cached_whitelist is not None and not force_refresh:
            return self._cached_whitelist

        whitelist: Set[str] = set()

        # 1. 嘗試由 DuckDB market_universe 及 predictions 載入
        try:
            import duckdb
            target_db = self.db_path or "stocks.duckdb"
            if os.path.exists(target_db):
                con = duckdb.connect(target_db, read_only=True)
                try:
                    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                    if "market_universe" in tables:
                        rows = con.execute("SELECT DISTINCT local_symbol, normalized_symbol FROM market_universe").fetchall()
                        for r in rows:
                            if r[0]:
                                c = canonicalize_ticker(r[0])
                                if c:
                                    whitelist.add(c)
                            if r[1]:
                                c = canonicalize_ticker(r[1])
                                if c:
                                    whitelist.add(c)

                    if "predictions" in tables:
                        pred_rows = con.execute("SELECT DISTINCT ticker FROM predictions").fetchall()
                        for r in pred_rows:
                            if r[0]:
                                c = canonicalize_ticker(r[0])
                                if c:
                                    whitelist.add(c)
                finally:
                    con.close()
        except Exception as e:
            logger.debug("自 DuckDB 讀取 Universe 異常: %s", e)

        # 2. 合併系統核心配置中的 TICKER_SECTOR_MAP
        try:
            from core.config import config
            for t in config.TICKER_SECTOR_MAP.keys():
                c = canonicalize_ticker(t)
                if c:
                    whitelist.add(c)
        except Exception:
            pass

        # 3. 合併系統 TW50 / 核心監控快取
        try:
            import json
            cache_file = "cache/stock_lists.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for group_data in data.values():
                        if isinstance(group_data, dict) and "stocks" in group_data:
                            stocks = group_data["stocks"]
                            if isinstance(stocks, dict):
                                for k in stocks.keys():
                                    c = canonicalize_ticker(k)
                                    if c:
                                        whitelist.add(c)
                            elif isinstance(stocks, list):
                                for s in stocks:
                                    c = canonicalize_ticker(s)
                                    if c:
                                        whitelist.add(c)
        except Exception:
            pass

        self._cached_whitelist = whitelist
        return self._cached_whitelist

    def is_valid(self, ticker: str, whitelist: Optional[Set[str]] = None) -> bool:
        """
        判斷代號是否合法且存在於白名單中。
        - 若提供 whitelist 參數，則強制檢驗 ticker 是否在該 set 中。
        - 若未提供 whitelist 參數，則使用系統內建 Universe 白名單。
        """
        canon = canonicalize_ticker(ticker)
        if not canon:
            return False

        if whitelist is not None:
            return canon in whitelist

        active_wl = self.get_whitelist()
        if active_wl:
            return canon in active_wl

        # 若環境完全無任何白名單配置，則只檢驗代號格式合法性
        return True

    def filter_and_canonicalize(
        self, tickers: Iterable[str], whitelist: Optional[Set[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        批次處理一組股票代號，分離為合格 (valid) 與拒絕 (rejected) 清單。
        """
        valid: List[str] = []
        rejected: List[str] = []
        seen = set()

        for t in tickers:
            canon = canonicalize_ticker(t)
            if not canon or not self.is_valid(canon, whitelist=whitelist):
                rejected.append(t)
            else:
                if canon not in seen:
                    seen.add(canon)
                    valid.append(canon)

        return valid, rejected


# 全域單例
default_validator = TickerUniverseValidator()
