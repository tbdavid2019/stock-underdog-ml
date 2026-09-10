"""
data/polymarket_service.py - Polymarket Prediction Market Real-Money Macro Sentiment Service

透過 2MD API (https://2md.aiurl.tw / https://2md.glsoft.ai / https://create360.ai)
以及 DoH (Google 8.8.8.8 / Cloudflare 1.1.1.1) 備援線路獲取 Polymarket 預測市場數據：
1. 聯準會利率決策 (FOMC Rate Cut / Hike / Pause Real-Money Probabilities)
2. 地緣政治與關稅風險 (Geopolitical Ceasefires, Tariffs, Sanctions)
3. 科技七巨頭與 AI 突破 (Big Tech AI Breakthroughs, Nvidia, Apple, Tesla)
4. 宏觀經濟衰退與美股大盤走勢 (US Recession, S&P 500, Inflation)
"""

import datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("stock_app.polymarket")


class PolymarketService:
    """
    Polymarket 真金白銀預測市場宏觀情緒與重大催化劑抓取服務
    """

    # 1. 2MD 代理節點輪詢清單
    FALLBACK_2MD_HOSTS = [
        "https://2md.aiurl.tw",
        "https://2md.glsoft.ai",
        "https://create360.ai",
    ]

    # 2. DoH 備援 DNS 節點 (8.8.8.8 & 1.1.1.1)
    DOH_ENDPOINTS = [
        "https://dns.google/resolve?name=gamma-api.polymarket.com&type=A",
        "https://1.1.1.1/dns-query?name=gamma-api.polymarket.com&type=A",
    ]

    TARGET_API = "https://gamma-api.polymarket.com/markets?limit=100&active=true&closed=false&order=volume24hr&ascending=false"
    TIMEOUT = 12
    TTL_MACRO_SENTIMENT = 900  # 15 分鐘快取

    # 快取結構
    _MEM_CACHE: Dict[str, Any] = {}
    _CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
    _CRAWLER_SEMAPHORE = threading.Semaphore(3)
    _RESOLVED_IP: Optional[str] = None
    _IP_RESOLVED_AT: float = 0

    # 分類規則
    PATTERNS = {
        "fed_rates": re.compile(
            r"\b(fed|federal reserve|fomc|interest rate|rates|rate cut|rate hike|bps)\b",
            re.IGNORECASE,
        ),
        "geopolitics": re.compile(
            r"\b(ceasefire|war|sanction|tariff|tariffs|taiwan|china|russia|ukraine|iran|israel|nato)\b",
            re.IGNORECASE,
        ),
        "tech_giants": re.compile(
            r"\b(nvidia|apple|iphone|tesla|openai|chatgpt|google|microsoft|semiconductor|tsmc)\b",
            re.IGNORECASE,
        ),
        "macro_recession": re.compile(
            r"\b(recession|gdp|inflation|cpi|unemployment|s&p|sp500|nasdaq|debt ceiling)\b",
            re.IGNORECASE,
        ),
    }

    EXCLUDE_PATTERNS = re.compile(
        r"\b(nba|nfl|mlb|nhl|atp|wta|premier league|champions league|super bowl|superbowl|oscars|vs\.?|against|score|points|over/under|matchup|lap top|laptop fdv)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _get_cache_filepath(cls, key: str) -> str:
        os.makedirs(cls._CACHE_DIR, exist_ok=True)
        return os.path.join(cls._CACHE_DIR, f"polymarket_{key}.json")

    @classmethod
    def _read_cache(cls, key: str, ttl: int) -> Optional[Any]:
        now = time.time()
        # 1. 檢查記憶體快取
        if key in cls._MEM_CACHE:
            cached_at, data = cls._MEM_CACHE[key]
            if now - cached_at < ttl:
                return data

        # 2. 檢查磁碟快取
        filepath = cls._get_cache_filepath(key)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cached_at = payload.get("cached_at", 0)
                data = payload.get("data")
                if data is not None and (now - cached_at < ttl):
                    cls._MEM_CACHE[key] = (cached_at, data)
                    return data
            except Exception as e:
                logger.debug(f"讀取 Polymarket 快取失敗 {key}: {e}")

        return None

    @classmethod
    def _write_cache(cls, key: str, data: Any):
        now = time.time()
        cls._MEM_CACHE[key] = (now, data)
        try:
            filepath = cls._get_cache_filepath(key)
            tmp_path = f"{filepath}.tmp.{threading.get_ident()}"
            payload = {
                "key": key,
                "cached_at": now,
                "cached_date": datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                "data": data,
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
        except Exception as e:
            logger.warning(f"寫入 Polymarket 磁碟快取失敗 {key}: {e}")

    @classmethod
    def _resolve_ip_via_doh(cls) -> Optional[str]:
        """透過 Google (8.8.8.8) 與 Cloudflare (1.1.1.1) DoH 解析 Polymarket IP 解決 ISP 污染"""
        now = time.time()
        if cls._RESOLVED_IP and (now - cls._IP_RESOLVED_AT < 3600):
            return cls._RESOLVED_IP

        for endpoint in cls.DOH_ENDPOINTS:
            try:
                headers = {"accept": "application/dns-json"}
                res = requests.get(endpoint, headers=headers, timeout=5).json()
                for ans in res.get("Answer", []):
                    if ans.get("type") == 1:  # Type A record
                        cls._RESOLVED_IP = ans.get("data")
                        cls._IP_RESOLVED_AT = now
                        logger.info(f"🌐 [DoH Fallback] gamma-api.polymarket.com 解析至: {cls._RESOLVED_IP}")
                        return cls._RESOLVED_IP
            except Exception as e:
                logger.debug(f"DoH 查詢失敗 {endpoint}: {e}")

        return None

    @classmethod
    def _fetch_raw_markets(cls) -> List[Dict[str, Any]]:
        """
        雙軌獲取 Polymarket 原始市場清單：
        1. 優先透過 2MD 代理輪詢
        2. 2MD 全數失敗時自動啟動 DoH (8.8.8.8 / 1.1.1.1) 直連
        """
        with cls._CRAWLER_SEMAPHORE:
            # 軌道 1: 2MD 輪詢
            for host in cls.FALLBACK_2MD_HOSTS:
                url = f"{host.rstrip('/')}/{cls.TARGET_API}"
                try:
                    resp = requests.get(url, timeout=cls.TIMEOUT)
                    if resp.status_code == 200:
                        text = resp.text
                        match = re.search(r"(\[|\{).*", text, re.DOTALL)
                        if match:
                            data = json.loads(match.group(0))
                            if isinstance(data, list) and len(data) > 0:
                                return data
                except requests.exceptions.Timeout:
                    logger.debug(f"2MD timeout on {host} for Polymarket, trying next...")
                except Exception as e:
                    logger.debug(f"2MD fetch error on {host} for Polymarket: {e}")

            # 軌道 2: DoH (8.8.8.8 / 1.1.1.1) 繞過本機/ISP 污染直連
            resolved_ip = cls._resolve_ip_via_doh()
            if resolved_ip:
                try:
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.connection import create_connection

                    class HostHeaderSSLAdapter(HTTPAdapter):
                        def init_poolmanager(self, *args, **kwargs):
                            def custom_create_connection(address, *args_conn, **kwargs_conn):
                                host, port = address
                                if host == "gamma-api.polymarket.com":
                                    address = (resolved_ip, port)
                                return create_connection(address, *args_conn, **kwargs_conn)

                            import urllib3.util.connection
                            urllib3.util.connection.create_connection = custom_create_connection
                            super().init_poolmanager(*args, **kwargs)

                    session = requests.Session()
                    session.mount("https://gamma-api.polymarket.com", HostHeaderSSLAdapter())
                    resp = session.get(cls.TARGET_API, timeout=cls.TIMEOUT)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list):
                            logger.info(f"✅ Polymarket DoH 直連成功，獲取 {len(data)} 檔市場")
                            return data
                except Exception as e:
                    logger.warning(f"Polymarket DoH 直連失敗: {e}")

        return []

    @classmethod
    def get_macro_sentiment(
        cls, force_refresh: bool = False, category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        獲取 Polymarket 真金白銀宏觀情緒與重大預測市場數據
        """
        cache_key = "macro_sentiment"
        if not force_refresh:
            cached = cls._read_cache(cache_key, cls.TTL_MACRO_SENTIMENT)
            if cached:
                if category:
                    filtered_markets = [
                        m for m in cached.get("markets", []) if m.get("category") == category
                    ]
                    return {**cached, "markets": filtered_markets, "count": len(filtered_markets)}
                return cached

        raw_markets = cls._fetch_raw_markets()
        processed_markets: List[Dict[str, Any]] = []

        # 指標統計
        fed_prob_summary: Dict[str, float] = {}

        for m in raw_markets:
            q = m.get("question", "").strip()
            if not q:
                continue

            # 排除純體育投注等雜訊
            if cls.EXCLUDE_PATTERNS.search(q):
                continue

            matched_cats = [cat for cat, pat in cls.PATTERNS.items() if pat.search(q)]
            if not matched_cats:
                continue

            primary_cat = matched_cats[0]

            # 解析機率與選項
            try:
                outcomes = json.loads(m.get("outcomes", "[]"))
                prices = json.loads(m.get("outcomePrices", "[]"))
                prices = [float(p) for p in prices]
            except Exception:
                outcomes = ["Yes", "No"]
                prices = [0.5, 0.5]

            odds_map: Dict[str, float] = {}
            for o, p in zip(outcomes, prices):
                odds_map[o] = round(p * 100, 1)

            vol_24h = float(m.get("volume24hr", 0) or 0)
            total_vol = float(m.get("volume", 0) or 0)
            liquidity = float(m.get("liquidity", 0) or 0)
            slug = m.get("slug", "")

            market_item = {
                "question": q,
                "category": primary_cat,
                "slug": slug,
                "url": f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com",
                "outcomes": outcomes,
                "odds_percent": odds_map,
                "yes_prob": odds_map.get("Yes", 0.0),
                "no_prob": odds_map.get("No", 0.0),
                "volume_24h": round(vol_24h, 2),
                "total_volume": round(total_vol, 2),
                "liquidity": round(liquidity, 2),
                "end_date": m.get("endDateIso", m.get("endDate", "")),
            }
            processed_markets.append(market_item)

            # 提取 Fed 核心預期
            if "fed" in q.lower() and "interest rates" in q.lower():
                yes_val = odds_map.get("Yes", 0.0)
                if "no change" in q.lower():
                    fed_prob_summary["pause"] = yes_val
                elif "decrease" in q.lower() and "25 bps" in q.lower():
                    fed_prob_summary["cut_25bps"] = yes_val
                elif "decrease" in q.lower() and "50" in q.lower():
                    fed_prob_summary["cut_50bps"] = yes_val
                elif "increase" in q.lower() and "25 bps" in q.lower():
                    fed_prob_summary["hike_25bps"] = yes_val

        # 依 24 小時成交量降序排序
        processed_markets.sort(key=lambda x: x["volume_24h"], reverse=True)

        # 整理輸出
        result = {
            "success": True,
            "source": "Polymarket via 2MD / DoH (8.8.8.8)",
            "timestamp": datetime.datetime.now().isoformat(),
            "count": len(processed_markets),
            "fed_real_money_odds": fed_prob_summary,
            "markets": processed_markets[:25],
        }

        # 寫入快取
        cls._write_cache(cache_key, result)

        if category:
            filtered = [m for m in result["markets"] if m.get("category") == category]
            return {**result, "markets": filtered, "count": len(filtered)}

        return result
