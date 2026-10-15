"""
data/investing_service.py - Investing.com 宏觀數據與美股行事曆整合服務

透過 2MD API (https://2md.aiurl.tw / https://2md.glsoft.ai / https://create360.ai) 獲取 Investing.com 之：
1. 聯準會利率決策監控 (Fed Rate Monitor Tool - CME FedWatch)
2. 重磅總體經濟行事曆 (Economic Calendar - CPI, PCE, 非農, FOMC)
3. 美股重量級財報行事曆 (US Earnings Calendar)
4. 關鍵大宗商品與原物料 (Commodities - 銅 Copper, WTI 原油, 黃金)

支援 3 級容災 Fallback、信號量節流、以及 6~12 小時原子磁碟持久化快取。
"""

import os
import re
import json
import time
import datetime
import logging
import threading
import concurrent.futures
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger("stock_app.investing")


class InvestingService:
    """Investing.com 宏觀與行事曆資訊服務"""

    FALLBACK_2MD_HOSTS = [
        "https://2md.glsoft.ai",
        "https://create360.ai",
        "https://2md.aiurl.tw"
    ]

    # 全域節流信號量 (擴展至 4 並發)
    _CRAWLER_SEMAPHORE = threading.Semaphore(4)
    TIMEOUT = 7.0

    # 磁碟快取目錄
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(BASE_DIR, "cache", "investing")

    # 快取效期配置 (秒)
    TTL_FED_RATE = 3600 * 12          # 聯準會降息機率 (12 小時)
    TTL_ECONOMIC_CALENDAR = 3600 * 6  # 總經行事曆 (6 小時)
    TTL_EARNINGS_CALENDAR = 3600 * 12 # 財報行事曆 (12 小時)
    TTL_COMMODITIES = 3600 * 6        # 大宗商品行情 (6 小時)
    TTL_SUMMARY = 3600 * 6            # 綜合總經前瞻 (6 小時)

    # 記憶體快取 L1
    _MEM_CACHE: Dict[str, tuple] = {}

    @classmethod
    def _get_cache_filepath(cls, key: str) -> str:
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        return os.path.join(cls.CACHE_DIR, f"{key}.json")

    @classmethod
    def _read_cache(cls, key: str, ttl: int, allow_stale: bool = False) -> Optional[Any]:
        now = time.time()
        # 1. 檢查 L1 記憶體
        if key in cls._MEM_CACHE:
            ts, data = cls._MEM_CACHE[key]
            if (now - ts < ttl) or allow_stale:
                return data

        # 2. 檢查 L2 DuckDB
        try:
            from data.duckdb_manager import DuckDBManager
            db = DuckDBManager()
            max_age = None if allow_stale else ttl
            db_data = db.get_macro_snapshot(key, max_age_seconds=max_age)
            if db_data is not None:
                cls._MEM_CACHE[key] = (now, db_data)
                logger.debug(f"📦 Investing DuckDB 快取命中: {key}")
                return db_data
        except Exception as e:
            logger.debug(f"讀取 DuckDB 宏觀快照失敗 {key}: {e}")

        # 3. 檢查 L3 磁碟
        filepath = cls._get_cache_filepath(key)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cached_at = payload.get("cached_at", 0)
                data = payload.get("data")
                if data is not None and ((now - cached_at < ttl) or allow_stale):
                    cls._MEM_CACHE[key] = (cached_at, data)
                    # 同步回寫 DuckDB 保持一致性
                    try:
                        from data.duckdb_manager import DuckDBManager
                        DuckDBManager().save_macro_snapshot(key, data, source="DiskCache")
                    except Exception:
                        pass
                    logger.debug(f"📦 Investing 磁碟快取命中: {key}")
                    return data
            except Exception as e:
                logger.debug(f"讀取 Investing 磁碟快取失敗 {key}: {e}")

        return None

    @classmethod
    def _write_cache(cls, key: str, data: Any):
        now = time.time()
        cls._MEM_CACHE[key] = (now, data)

        # 1. 寫入 L2 DuckDB
        try:
            from data.duckdb_manager import DuckDBManager
            DuckDBManager().save_macro_snapshot(key, data, source="Investing.com")
        except Exception as e:
            logger.debug(f"寫入 DuckDB 宏觀快照失敗 {key}: {e}")

        # 2. 寫入 L3 磁碟
        try:
            filepath = cls._get_cache_filepath(key)
            tmp_path = f"{filepath}.tmp.{threading.get_ident()}"
            payload = {
                "key": key,
                "cached_at": now,
                "cached_date": datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                "data": data
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
            logger.debug(f"💾 Investing 快取已持久化至磁碟與 DuckDB: {key}")
        except Exception as e:
            logger.warning(f"寫入 Investing 磁碟快取失敗 {key}: {e}")

    @classmethod
    def _fetch_url_markdown(cls, target_url: str) -> str:
        """透過 2MD 輪詢擷取目標網址之 Markdown 內容"""
        with cls._CRAWLER_SEMAPHORE:
            for host in cls.FALLBACK_2MD_HOSTS:
                reader_url = f"{host.rstrip('/')}/{target_url}"
                try:
                    resp = requests.get(reader_url, timeout=cls.TIMEOUT)
                    if resp.status_code == 200:
                        resp.encoding = "utf-8"
                        text = resp.text
                        if len(text) > 100 and "Performing security verification" not in text and "Just a moment..." not in text:
                            return text
                        else:
                            logger.debug(f"2md host {host} response for {target_url} was blocked or too short, falling back...")
                            continue
                    elif resp.status_code in (429, 502, 503, 504):
                        logger.warning(f"2md host {host} returned {resp.status_code}, falling back...")
                        continue
                except requests.exceptions.Timeout:
                    logger.warning(f"2md timeout on {host} for {target_url}, falling back to next host...")
                    continue
                except Exception as e:
                    logger.debug(f"2md fetch error on {host} for {target_url}: {e}")
                    continue

        return ""

    # =========================================================================
    # 1. 聯準會利率監控 (Fed Rate Monitor Tool)
    # =========================================================================

    @classmethod
    def get_fed_rate_monitor(cls, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        取得 CME 聯準會下次 FOMC 利率決策機率
        來源: https://www.investing.com/central-banks/fed-rate-monitor
        """
        cache_key = "fed_rate_monitor"
        if not force_refresh:
            cached = cls._read_cache(cache_key, cls.TTL_FED_RATE)
            if cached:
                return cached

        target_url = "https://www.investing.com/central-banks/fed-rate-monitor"
        text = cls._fetch_url_markdown(target_url)
        result: Dict[str, Any] = {
            "source": "Investing.com (CME Group)",
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "meeting_date": "近期會議",
            "future_price": None,
            "target_rates": [],
            "highest_prob_rate": None,
            "highest_probability": None,
            "summary": "數據更新中"
        }

        if text:
            # 解析會議時間
            m_date = re.search(r"Meeting Time:_([^_\n]+)_", text) or re.search(r"Fed Interest Rate Decision ([A-Za-z]+\s+\d+,\s+\d{4})", text)
            if m_date:
                result["meeting_date"] = m_date.group(1).strip()

            # 解析期貨定價
            m_price = re.search(r"Future Price:_([\d\.]+)_", text)
            if m_price:
                try:
                    result["future_price"] = float(m_price.group(1))
                except ValueError:
                    pass

            # 解析機率表格 (Target Rate | Current | Previous Day | Previous Week)
            # 例如: | 3.50 - 3.75 | 49.6% | 39.9% | 65.6% |
            rows = re.findall(r"\|\s*([\d\.]+\s*-\s*[\d\.]+)\s*\|\s*([\d\.]+%?)\s*\|\s*([\d\.]+%?)\s*\|\s*([\d\.]+%?)\s*\|", text)
            parsed_rates = []
            highest_prob = -1.0
            highest_rate = ""

            for r in rows:
                rate_range = r[0].strip()
                curr_prob_str = r[1].strip().replace("%", "")
                prev_day_str = r[2].strip()
                prev_week_str = r[3].strip()

                try:
                    curr_prob = float(curr_prob_str)
                except ValueError:
                    curr_prob = 0.0

                parsed_rates.append({
                    "rate_range": rate_range,
                    "current_probability": f"{curr_prob:.1f}%",
                    "probability_value": curr_prob,
                    "previous_day": prev_day_str,
                    "previous_week": prev_week_str
                })

                if curr_prob > highest_prob:
                    highest_prob = curr_prob
                    highest_rate = rate_range

            if parsed_rates:
                result["target_rates"] = parsed_rates
                result["highest_prob_rate"] = highest_rate
                result["highest_probability"] = f"{highest_prob:.1f}%"
                result["summary"] = f"市場預期下次 FOMC ({result['meeting_date']}) 目標利率以 {highest_rate} 機率最高 ({highest_prob:.1f}%)"
                cls._write_cache(cache_key, result)
                return result

        if not force_refresh:
            stale = cls._read_cache(cache_key, ttl=3600 * 24 * 30, allow_stale=True)
            if stale:
                return stale

        logger.warning("⚠️ FedWatch 上游沒有可驗證資料，略過利率預期顯示")
        return None

    # =========================================================================
    # 2. 美股重量級財報行事曆 (US Earnings Calendar)
    # =========================================================================

    @classmethod
    def get_earnings_calendar(cls, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        取得近期重要美股財報公佈行事曆
        來源: https://www.investing.com/earnings-calendar/
        """
        cache_key = "earnings_calendar"
        if not force_refresh:
            cached = cls._read_cache(cache_key, cls.TTL_EARNINGS_CALENDAR)
            if cached:
                return cached

        target_url = "https://www.investing.com/earnings-calendar/"
        text = cls._fetch_url_markdown(target_url)
        items: List[Dict[str, Any]] = []

        if text:
            current_date = "近期"
            for line in text.splitlines():
                # 匹配日期標題列 (例如: Friday, September 4, 2026)
                date_match = re.search(r"\|\s*([A-Za-z]+day,\s*[A-Za-z]+\s+\d+,\s+\d{4})", line)
                if date_match:
                    current_date = date_match.group(1).strip()
                    continue

                # 匹配公司資料列
                # 例如: #### KNOT Offshore Partners LP ( [KNOP](...) ) KNOT ... | - / | 0.1526 | - / | 91.7M | 388.03M |
                row_match = re.search(
                    r"####\s*(.+?)\s*\(\s*\[([A-Z0-9\.\-]+)\]\([^\)]+\)\s*\)\s*(?:.+?)\|\s*([^\/\|]+)\s*/\s*\|\s*([^\/\|]+)\s*\|\s*([^\/\|]+)\s*/\s*\|\s*([^\/\|]+)\s*\|\s*([^\/\|]+)",
                    line
                )
                if row_match:
                    name = row_match.group(1).strip()
                    ticker = row_match.group(2).strip()
                    eps_forecast = row_match.group(4).strip()
                    rev_forecast = row_match.group(6).strip()
                    market_cap = row_match.group(7).strip()

                    items.append({
                        "date": current_date,
                        "ticker": ticker,
                        "name": name,
                        "eps_forecast": eps_forecast if eps_forecast != "-" else "未公佈",
                        "revenue_forecast": rev_forecast if rev_forecast != "-" else "未公佈",
                        "market_cap": market_cap if market_cap != "-" else "—",
                        "yahoo_link": f"https://finance.yahoo.com/quote/{ticker}"
                    })

            if items:
                # 按市值或順序保留前 30 家
                cls._write_cache(cache_key, items[:30])
                return items[:30]

        if not force_refresh:
            stale = cls._read_cache(cache_key, ttl=3600 * 24 * 30, allow_stale=True)
            if stale:
                return stale

        return []

    # =========================================================================
    # 3. 關鍵大宗商品與原物料 (Commodities)
    # =========================================================================

    @classmethod
    def get_commodities_summary(cls, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        取得黃金、銅博士、WTI 原油即時行情與漲跌
        來源: https://www.investing.com/commodities/
        """
        cache_key = "commodities_summary"
        if not force_refresh:
            cached = cls._read_cache(cache_key, cls.TTL_COMMODITIES)
            if cached:
                return cached

        target_url = "https://www.investing.com/commodities/"
        text = cls._fetch_url_markdown(target_url)
        commodities: Dict[str, Any] = {}

        if text:
            for line in text.splitlines():
                # 匹配大宗商品報價行
                # | - [x] | #### [Gold](...) | -0.06% | -0.06% | -0.46% | ...
                m = re.search(r"####\s*\[([^\]]+)\]\([^\)]+\)[^\|]*\|\s*([+\-]?\d+\.?\d*%?)\s*\|\s*([+\-]?\d+\.?\d*%?)\s*\|\s*([+\-]?\d+\.?\d*%?)", line)
                if m:
                    raw_name = m.group(1).strip()
                    chg_d = m.group(2).strip()
                    chg_w = m.group(4).strip()

                    if "Gold" in raw_name:
                        commodities["gold"] = {"name": "黃金 (Gold)", "daily_change": chg_d, "weekly_change": chg_w, "indicator": "避險與流動性"}
                    elif "Copper" in raw_name:
                        commodities["copper"] = {"name": "銅 (Copper)", "daily_change": chg_d, "weekly_change": chg_w, "indicator": "製造業與AI硬體景氣"}
                    elif "Crude Oil" in raw_name:
                        commodities["crude_oil"] = {"name": "WTI 原油 (Crude Oil)", "daily_change": chg_d, "weekly_change": chg_w, "indicator": "通膨預期與能源成本"}

            if commodities:
                cls._write_cache(cache_key, commodities)
                return commodities

        if not force_refresh:
            stale = cls._read_cache(cache_key, ttl=3600 * 24 * 30, allow_stale=True)
            if stale:
                return stale

        logger.warning("⚠️ 商品行情上游沒有可驗證資料，略過商品行情顯示")
        return None

    @staticmethod
    def _clean_and_dedup_economic_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗並去重總體經濟數據行事曆：
        1. 解決同一機構同時間發布 s.a. (季調後) 與 n.s.a. (未季調) 之重複顯示問題，優先保留最具市場參考性之 s.a. 季調數據。
        2. 自動為 s.a. 與 n.s.a. 標註中文解說，如「季調後 (s.a.)」、「未季調 (n.s.a.)」，消除語意混淆。
        """
        if not raw_events:
            return []

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for ev in raw_events:
            t = ev.get("event", "")
            # 去除 s.a. / n.s.a. 等縮寫標記，提取基準指標名
            base_name = re.sub(r'[\(\[\{]?(?:n\.s\.a\.|s\.a\.|nsa|sa)[\)\]\}]?', '', t, flags=re.I)
            base_name = re.sub(r'\s+', ' ', base_name).strip()
            group_key = f"{ev.get('date')}_{ev.get('time')}_{ev.get('country')}_{base_name.lower()}"
            groups.setdefault(group_key, []).append(ev)

        deduped: List[Dict[str, Any]] = []
        for g_key, ev_list in groups.items():
            if len(ev_list) == 1:
                chosen = ev_list[0]
            else:
                # 若同時包含 s.a. (季調後) 與 n.s.a. (未季調)，市場標準優先採用 s.a.
                sa_ev = next((e for e in ev_list if re.search(r'\bs\.a\.?\b', e.get("event", ""), re.I)), None)
                chosen = sa_ev if sa_ev else ev_list[0]

            # 增強標題中文可讀性
            ev_title = chosen.get("event", "")
            if re.search(r'\bn\.s\.a\.?\b', ev_title, re.I):
                ev_title = re.sub(r'\bn\.s\.a\.?\b', '未季調 (n.s.a.)', ev_title, flags=re.I)
            elif re.search(r'\bs\.a\.?\b', ev_title, re.I):
                ev_title = re.sub(r'\bs\.a\.?\b', '季調後 (s.a.)', ev_title, flags=re.I)

            chosen_copy = dict(chosen)
            chosen_copy["event"] = ev_title
            deduped.append(chosen_copy)

        return deduped

    # =========================================================================
    # 4. 重磅總體經濟行事曆 (Economic Calendar)
    # =========================================================================

    @classmethod
    def get_economic_calendar(cls, filter_countries: Optional[List[str]] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        取得即將到來的總體經濟數據行事曆 (預設過濾 US / TW 重要事件)
        來源: https://www.investing.com/economic-calendar/
        """
        cache_key = "economic_calendar"
        if not force_refresh:
            cached = cls._read_cache(cache_key, cls.TTL_ECONOMIC_CALENDAR)
            if cached:
                return cached

        target_url = "https://www.investing.com/economic-calendar/"
        text = cls._fetch_url_markdown(target_url)
        events: List[Dict[str, Any]] = []

        if text:
            # 支援格式: | 08:30 US | [Nonfarm Payrolls (Aug)](...) Act: - Cons: 160K Prev.: 114K |
            allowed = [c.upper() for c in (filter_countries or ["US", "USD", "TW", "TWD", "GLOBAL"])]
            current_date = "近期"

            for line in text.splitlines():
                d_match = re.search(r"\|\s*([A-Za-z]+day,\s*[A-Za-z]+\s+\d+,\s+\d{4})", line)
                if d_match:
                    current_date = d_match.group(1).strip()
                    continue

                # 匹配事件行
                # | 05:00 JP | 05:00 | JP | [Leading Index ...](...) Act: - Cons: - Prev.: 116.5 |
                row_m = re.search(r"\|\s*([0-9:]+)\s*([A-Z]{2,3})?\s*\|\s*[0-9:]*\s*\|\s*([A-Z]{2,3})\s*\|\s*\[([^\]]+)\]\([^\)]+\)\s*(?:Act:\s*([^\s]+))?\s*(?:Cons:\s*([^\s]+))?\s*(?:Prev.:\s*([^\s\|]+))?", line)
                if row_m:
                    time_str = row_m.group(1).strip()
                    country = (row_m.group(3) or row_m.group(2) or "US").strip().upper()
                    event_title = row_m.group(4).strip()
                    actual = row_m.group(5) or "—"
                    consensus = row_m.group(6) or "—"
                    previous = row_m.group(7) or "—"

                    # 篩選感興趣的國家或全球核心核彈級事件 (FED/FOMC/CPI/PCE/PAYROLL/GDP)
                    is_core_global = any(k in event_title.upper() for k in ("CPI", "FOMC", "PAYROLLS", "FED", "GDP", "PCE", "INTEREST RATE DECISION"))
                    if country in allowed or is_core_global:
                        # 標註是否為市場核彈級指標
                        is_high_impact = any(h in event_title.upper() for h in ("CPI", "PCE", "PAYROLL", "FOMC", "INTEREST RATE", "GDP", "UNEMPLOYMENT"))

                        events.append({
                            "date": current_date,
                            "time": time_str,
                            "country": country,
                            "event": event_title,
                            "actual": actual,
                            "forecast": consensus,
                            "previous": previous,
                            "is_high_impact": is_high_impact
                        })

            if events:
                events = cls._clean_and_dedup_economic_events(events)
                cls._write_cache(cache_key, events[:25])
                return events[:25]

        if not force_refresh:
            stale = cls._read_cache(cache_key, ttl=3600 * 24 * 30, allow_stale=True)
            if stale:
                return stale

        return []

    # =========================================================================
    # 5. 綜合宏觀前瞻摘要 (Full Macro & Calendar Summary - DuckDB Fast Parallel)
    # =========================================================================

    @classmethod
    def get_investing_macro_summary(cls, force_refresh: bool = False) -> Dict[str, Any]:
        """彙整聯準會利率預期、重要財報與大宗原物料之一站式宏觀數據 (DuckDB 優先並行加速)"""
        cache_key = "investing_summary"

        if not force_refresh:
            cached = cls._read_cache(cache_key, cls.TTL_SUMMARY)
            if cached:
                return cached

        # 並行抓取 4 大宏觀數據，大幅降低延遲至秒級
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            fut_fed = executor.submit(cls.get_fed_rate_monitor, force_refresh)
            fut_earn = executor.submit(cls.get_earnings_calendar, force_refresh)
            fut_comm = executor.submit(cls.get_commodities_summary, force_refresh)
            fut_econ = executor.submit(cls.get_economic_calendar, None, force_refresh)

            fed_rate = fut_fed.result()
            earnings = fut_earn.result()
            commodities = fut_comm.result()
            economic_calendar = fut_econ.result()

        # 提煉今日與明日重點警示 (去重且多元)
        clean_econ = cls._clean_and_dedup_economic_events(economic_calendar or [])
        high_impact_events = [e for e in clean_econ if e.get("is_high_impact")]
        catalyst_alerts = []
        seen_alert_keys = set()
        for e in high_impact_events:
            alert_k = f"{e.get('country', '')}_{e.get('event', '')}"
            if alert_k not in seen_alert_keys:
                seen_alert_keys.add(alert_k)
                catalyst_alerts.append(f"⚠️ {e.get('date', '')} {e.get('time', '')} [{e.get('country', '')}] {e.get('event')}")
            if len(catalyst_alerts) >= 3:
                break

        result = {
            "fed_rate": fed_rate,
            "earnings_calendar": (earnings or [])[:15],
            "commodities": commodities,
            "economic_calendar": (clean_econ or [])[:10],
            "catalyst_alerts": catalyst_alerts,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # 只要有抓到任一有效區塊即快取寫入 DuckDB 與磁碟
        if fed_rate or commodities or earnings or economic_calendar:
            cls._write_cache(cache_key, result)
        elif not force_refresh:
            # 若全部抓取失敗，優先使用 DuckDB 中的歷史快照容災，杜絕前端空窗
            stale = cls._read_cache(cache_key, ttl=3600 * 24 * 30, allow_stale=True)
            if stale:
                return stale

        return result
