"""
data/company_profile.py - 公司簡介與即時新聞提取服務 (支援 2md.aiurl.tw 與多級 Fallback)

整合 2md URL to Markdown API (https://2md.aiurl.tw / https://2md.glsoft.ai / https://create360.ai)
提供股票代碼 (如 9945.TW, 2330.TW, AAPL) 之繁體中文公司簡介、核心營運業務、董事長/股本與最新新聞摘要。
內建記憶體 LRU 快取與 DuckDB 股票名稱解析，提供毫秒級 Hover Tooltip 回應速度。
"""

import re
import time
import logging
import urllib.parse
from typing import Dict, List, Optional, Any
import requests

from data.ticker_resolver import normalize_ticker, COMPANY_ALIASES
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.company_profile")


class CompanyProfileService:
    """公司簡介與即時新聞服務"""

    FALLBACK_2MD_HOSTS = [
        "https://2md.aiurl.tw",
        "https://2md.glsoft.ai",
        "https://create360.ai"
    ]

    # In-memory cache: ticker -> (timestamp, data_dict)
    _CACHE: Dict[str, tuple] = {}
    CACHE_TTL = 3600 * 6  # 6 小時快取

    @classmethod
    def get_company_profile(cls, ticker_raw: str, db: Optional[DuckDBManager] = None) -> Dict[str, Any]:
        """
        取得公司名稱、簡介、主要業務與最新新聞
        """
        ticker = normalize_ticker(ticker_raw).upper()
        now = time.time()

        # 1. 檢查快取
        if ticker in cls._CACHE:
            ts, cached_data = cls._CACHE[ticker]
            if now - ts < cls.CACHE_TTL:
                return cached_data

        # 2. 解析公司基本名稱
        stock_name = cls._resolve_stock_name(ticker, db)
        is_tw = ticker.endswith(".TW") or ticker.endswith(".TWO")
        raw_code = ticker.split(".")[0] if is_tw else ticker

        profile_data = {
            "ticker": ticker,
            "raw_code": raw_code,
            "name": stock_name or raw_code,
            "industry": "綜合產業",
            "business_summary": "",
            "chairman": "",
            "market_cap": "",
            "established": "",
            "recent_news": [],
            "links": cls._build_stock_links(ticker, raw_code, is_tw),
            "source": "2md"
        }

        # 3. 嘗試透過 2md Web Reader 抓取 Yahoo 股市基本資料
        try:
            cls._fetch_profile_from_2md(profile_data, is_tw)
        except Exception as e:
            logger.warning(f"⚠️ [2md Profile] 抓取 {ticker} 基本資料異常 ({e})，嘗試搜尋備援...")

        # 4. 嘗試透過 2md 搜尋抓取最新即時新聞
        try:
            cls._fetch_news_from_2md(profile_data, is_tw)
        except Exception as e:
            logger.warning(f"⚠️ [2md News] 搜尋 {ticker} 即時新聞異常 ({e})")

        # 5. 若 2md 未取得足夠簡介，使用 yfinance 終極備援
        if not profile_data.get("business_summary") or profile_data["business_summary"] == "":
            cls._fallback_yfinance(profile_data, ticker)

        # 6. 寫入快取
        cls._CACHE[ticker] = (now, profile_data)
        return profile_data

    @classmethod
    def _resolve_stock_name(cls, ticker: str, db: Optional[DuckDBManager] = None) -> Optional[str]:
        """從 DuckDB 或別名表解析繁體中文股票名稱"""
        # 1. 別名表反查
        for alias, t in COMPANY_ALIASES.items():
            if t.upper() == ticker.upper():
                return alias

        # 2. DuckDB 查詢
        manager = db or DuckDBManager()
        try:
            with manager._get_connection() as con:
                row = con.execute("SELECT name FROM tw_daily_bars WHERE ticker = ? AND name IS NOT NULL AND name != '' LIMIT 1", [ticker]).fetchone()
                if row and row[0]:
                    return str(row[0]).strip()
                row_inst = con.execute("SELECT name FROM tw_institutional_daily WHERE ticker = ? AND name IS NOT NULL AND name != '' LIMIT 1", [ticker]).fetchone()
                if row_inst and row_inst[0]:
                    return str(row_inst[0]).strip()
        except Exception as e:
            logger.debug(f"DuckDB stock name lookup failed: {e}")

        return None

    @classmethod
    def _fetch_profile_from_2md(cls, profile: Dict[str, Any], is_tw: bool):
        """利用 2md URL Reader 抓取 Yahoo 股市 Profile 頁面"""
        ticker = profile["ticker"]
        if is_tw:
            target_url = f"https://tw.stock.yahoo.com/quote/{ticker}/profile"
        else:
            target_url = f"https://finance.yahoo.com/quote/{ticker}/profile"

        text = ""
        for host in cls.FALLBACK_2MD_HOSTS:
            try:
                reader_url = f"{host.rstrip('/')}/{target_url}"
                resp = requests.get(reader_url, timeout=6)
                if resp.status_code == 200:
                    resp.encoding = "utf-8"
                    if len(resp.text) > 20:
                        text = resp.text
                        break
            except Exception:
                continue

        if not text:
            return

        # 解析台股 Yahoo 基本資料
        if is_tw:
            # 公司名稱
            name_m = re.search(r"公司名稱\s*\n\s*([^\n]+)", text)
            if name_m and name_m.group(1).strip() and (not profile.get("name") or profile["name"] == profile["raw_code"]):
                profile["name"] = name_m.group(1).strip()

            # 產業類別
            ind_m = re.search(r"產業類別\s*\n\s*([^\n]+)", text)
            if ind_m and ind_m.group(1).strip():
                cat = ind_m.group(1).strip()
                if cat != "其他":
                    profile["industry"] = cat

            # 董事長
            chair_m = re.search(r"董事長\s*\n\s*([^\n]+)", text)
            if chair_m:
                profile["chairman"] = chair_m.group(1).strip()

            # 市值
            cap_m = re.search(r"市值\s*\(百萬\)\s*\n\s*([\d,\.]+)", text)
            if cap_m:
                try:
                    val = float(cap_m.group(1).replace(",", ""))
                    profile["market_cap"] = f"{val/100:.1f} 億元" if val > 100 else f"{val:.0f} 百萬元"
                except Exception:
                    pass

            # 成立時間
            est_m = re.search(r"成立時間\s*\n\s*([^\n]+)", text)
            if est_m:
                profile["established"] = est_m.group(1).strip()

            # 主要經營業務
            biz_m = re.search(r"主要經營業務\s*\n\s*([^\n#]+)", text)
            if biz_m:
                biz = biz_m.group(1).strip()
                biz = re.sub(r"\s+", " ", biz)
                profile["business_summary"] = biz[:300]

    @classmethod
    def _fetch_news_from_2md(cls, profile: Dict[str, Any], is_tw: bool):
        """利用 2md Live Search 抓取即時新聞與業務概述"""
        ticker = profile["ticker"]
        name = profile.get("name") or ticker
        query = f"{ticker} {name} 公司簡介 營運 最新新聞"

        text = ""
        for host in cls.FALLBACK_2MD_HOSTS:
            try:
                search_url = f"{host.rstrip('/')}/s/{urllib.parse.quote(query)}"
                resp = requests.get(search_url, timeout=5)
                if resp.status_code == 200:
                    resp.encoding = "utf-8"
                    if len(resp.text) > 20:
                        text = resp.text
                        break
            except Exception:
                continue

        if not text:
            return

        # 解析 Markdown 搜尋結果
        pattern = re.compile(r"\[(\d+)\] Title:\s*(.+?)\n\[\1\] URL Source:\s*(.+?)\n\[\1\] Description:\s*(.+?)(?=\n\[\d+\] Title|\Z)", re.DOTALL)
        matches = pattern.findall(text)
        news_items = []
        for m in matches:
            title = m[1].strip()
            url = m[2].strip()
            desc = m[3].strip()
            # 過濾無關結果
            if not title or len(title) < 4:
                continue
            # 若無 business_summary 且該項目包含傳產/科技營運描述，可用作補充
            if not profile.get("business_summary") and ("業務" in desc or "營運" in desc or "從事" in desc or "提供" in desc):
                profile["business_summary"] = desc[:200]

            news_items.append({
                "title": title,
                "url": url,
                "snippet": desc[:120] + ("..." if len(desc) > 120 else "")
            })
            if len(news_items) >= 4:
                break

        profile["recent_news"] = news_items

    @classmethod
    def _fallback_yfinance(cls, profile: Dict[str, Any], ticker: str):
        """使用 yfinance 作為最後備援"""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            if not profile.get("name") or profile["name"] == profile["raw_code"]:
                profile["name"] = info.get("shortName") or info.get("longName") or profile["raw_code"]
            if profile.get("industry") == "綜合產業" and info.get("industry"):
                profile["industry"] = info.get("industry")
            if not profile.get("business_summary") and info.get("longBusinessSummary"):
                profile["business_summary"] = info.get("longBusinessSummary")[:300]
        except Exception as e:
            logger.debug(f"yfinance profile fallback failed: {e}")

    @staticmethod
    def _build_stock_links(ticker: str, raw_code: str, is_tw: bool) -> Dict[str, str]:
        """建立常用外部行情鏈結"""
        if is_tw:
            return {
                "yahoo": f"https://tw.stock.yahoo.com/quote/{ticker}",
                "goodinfo": f"https://goodinfo.tw/tw/BasicInfo.asp?STOCK_ID={raw_code}",
                "cnyes": f"https://www.cnyes.com/twstock/{raw_code}",
                "tradingview": f"https://www.tradingview.com/symbols/TWSE-{raw_code}/"
            }
        else:
            return {
                "yahoo": f"https://finance.yahoo.com/quote/{ticker}",
                "finviz": f"https://finviz.com/quote.ashx?t={ticker}",
                "tradingview": f"https://www.tradingview.com/symbols/{ticker}/"
            }
