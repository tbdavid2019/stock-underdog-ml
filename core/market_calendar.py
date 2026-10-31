"""
core/market_calendar.py - 跨市場交易行事曆與開休市守衛模組 (Market Trading Calendar & Holiday Guard)

功能特點：
1. 支援台股 (TWSE/TPEx) 與美股 (NYSE/NASDAQ) 市場辨識與精準交易日/休市日判斷。
2. 台股串接 TWSE 官方 OpenAPI (https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule)，
   支援 30 天本機檔案快取 (data/cache/market_calendar/twse_holidays_{year}.json) 與內嵌 2024~2027 備援字典。
3. 嚴謹解析民國年 (如 1150101 -> 2026-01-01) 及「最後交易/開始交易/市場無交易/放假補假」語意分類。
4. 美股採用 NYSE/NASDAQ 官方標準 10 大節日規則，以 Meeus/Butcher 演算法精算復活節好禮拜五 (Good Friday)，
   並嚴格遵循週六提前至週五、週日延後至週一之補假 (Observation Rule) 規則。
5. 提供前後交易日計算 (T+1 / T-1)、最近交易日回溯 (含台股 15:30 籌碼出表防護)，為全市場同步腳本與 API 提供開休市守衛。
"""

import os
import re
import json
import time
import logging
import datetime
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger("stock_app.market_calendar")


class MarketCalendar:
    """跨市場 (台股 TWSE / 美股 NYSE) 交易行事曆與休市守衛"""

    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "cache",
        "market_calendar",
    )
    CACHE_EXPIRATION_SECONDS = 30 * 86400  # 30 天

    # 記憶體快取 (減少磁碟與網路頻繁讀取)
    _TW_HOLIDAY_CACHE: Dict[int, Dict[str, Dict[str, str]]] = {}
    _TW_SPECIAL_TRADING_CACHE: Dict[int, set] = {}
    _US_HOLIDAY_CACHE: Dict[int, Dict[str, Dict[str, str]]] = {}
    _JP_HOLIDAY_CACHE: Dict[int, Dict[str, Dict[str, str]]] = {}

    # 備援離線資料庫 (2024 ~ 2027 台股官方公告重要休市與封關交割日)
    FALLBACK_TW_HOLIDAYS: Dict[str, Dict[str, str]] = {
        # 2024
        "2024-01-01": {"name": "中華民國開國紀念日", "description": "依規定放假1日。"},
        "2024-02-06": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前市場無交易"},
        "2024-02-07": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前市場無交易"},
        "2024-02-08": {"name": "農曆除夕及春節", "description": "農曆春節連假"},
        "2024-02-09": {"name": "農曆除夕及春節", "description": "農曆除夕"},
        "2024-02-12": {"name": "農曆除夕及春節", "description": "春節初三補假"},
        "2024-02-13": {"name": "農曆除夕及春節", "description": "春節初四補假"},
        "2024-02-14": {"name": "農曆除夕及春節", "description": "春節初五補假"},
        "2024-02-28": {"name": "和平紀念日", "description": "依規定放假1日。"},
        "2024-04-04": {"name": "兒童節及民族掃墓節", "description": "兒童節"},
        "2024-04-05": {"name": "兒童節及民族掃墓節", "description": "民族掃墓節"},
        "2024-05-01": {"name": "勞動節", "description": "依規定放假1日。"},
        "2024-06-10": {"name": "端午節", "description": "依規定放假1日。"},
        "2024-09-17": {"name": "中秋節", "description": "依規定放假1日。"},
        "2024-10-10": {"name": "國慶日", "description": "依規定放假1日。"},

        # 2025
        "2025-01-01": {"name": "中華民國開國紀念日", "description": "依規定放假1日。"},
        "2025-01-22": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前無交易，僅辦理結算交割作業"},
        "2025-01-23": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前無交易，僅辦理結算交割作業"},
        "2025-01-24": {"name": "農曆除夕及春節", "description": "春節小年夜放假"},
        "2025-01-27": {"name": "農曆除夕及春節", "description": "春節調整放假"},
        "2025-01-28": {"name": "農曆除夕及春節", "description": "除夕"},
        "2025-01-29": {"name": "農曆除夕及春節", "description": "初一"},
        "2025-01-30": {"name": "農曆除夕及春節", "description": "初二"},
        "2025-01-31": {"name": "農曆除夕及春節", "description": "初三"},
        "2025-02-28": {"name": "和平紀念日", "description": "依規定放假1日。"},
        "2025-04-03": {"name": "兒童節及民族掃墓節", "description": "兒童節補假"},
        "2025-04-04": {"name": "兒童節及民族掃墓節", "description": "民族掃墓節"},
        "2025-05-01": {"name": "勞動節", "description": "依規定放假1日。"},
        "2025-05-30": {"name": "端午節", "description": "端午節補假"},
        "2025-10-06": {"name": "中秋節", "description": "中秋節補假"},
        "2025-10-10": {"name": "國慶日", "description": "依規定放假1日。"},
        "2025-12-25": {"name": "行憲紀念日", "description": "依規定放假1日。"},

        # 2026
        "2026-01-01": {"name": "中華民國開國紀念日", "description": "依規定放假1日。"},
        "2026-02-12": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前無交易，僅辦理結算交割作業"},
        "2026-02-13": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前無交易，僅辦理結算交割作業"},
        "2026-02-16": {"name": "農曆除夕及春節", "description": "依規定放假。"},
        "2026-02-17": {"name": "農曆除夕及春節", "description": "依規定放假。"},
        "2026-02-18": {"name": "農曆除夕及春節", "description": "依規定放假。"},
        "2026-02-19": {"name": "農曆除夕及春節", "description": "依規定放假。"},
        "2026-02-20": {"name": "農曆除夕及春節", "description": "農曆春節補假。"},
        "2026-02-27": {"name": "和平紀念日", "description": "和平紀念日補假。"},
        "2026-02-28": {"name": "和平紀念日", "description": "依規定放假1日。"},
        "2026-04-03": {"name": "兒童節及民族掃墓節", "description": "兒童節補假。"},
        "2026-04-04": {"name": "兒童節及民族掃墓節", "description": "依規定放假1日。"},
        "2026-04-05": {"name": "兒童節及民族掃墓節", "description": "依規定放假1日。"},
        "2026-04-06": {"name": "兒童節及民族掃墓節", "description": "民族掃墓節補假。"},
        "2026-05-01": {"name": "勞動節", "description": "依規定放假1日。"},
        "2026-06-19": {"name": "端午節", "description": "依規定放假1日。"},
        "2026-09-25": {"name": "中秋節", "description": "依規定放假1日。"},
        "2026-09-28": {"name": "孔子誕辰紀念日/ 教師節", "description": "依規定放假1日。"},
        "2026-10-09": {"name": "國慶日", "description": "國慶日補假。"},
        "2026-10-10": {"name": "國慶日", "description": "依規定放假1日。"},
        "2026-10-25": {"name": "臺灣光復暨金門古寧頭大捷紀念日", "description": "依規定放假1日。"},
        "2026-10-26": {"name": "臺灣光復暨金門古寧頭大捷紀念日", "description": "補假1日。"},
        "2026-12-25": {"name": "行憲紀念日", "description": "依規定放假1日。"},

        # 2027
        "2027-01-01": {"name": "中華民國開國紀念日", "description": "依規定放假1日。"},
        "2027-02-04": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前無交易，僅辦理結算交割作業"},
        "2027-02-05": {"name": "市場無交易，僅辦理結算交割作業", "description": "農曆春節前無交易，僅辦理結算交割作業"},
        "2027-02-08": {"name": "農曆除夕及春節", "description": "農曆春節連假"},
        "2027-02-09": {"name": "農曆除夕及春節", "description": "農曆春節連假"},
        "2027-02-10": {"name": "農曆除夕及春節", "description": "農曆春節連假"},
        "2027-02-11": {"name": "農曆除夕及春節", "description": "農曆春節連假"},
        "2027-02-12": {"name": "農曆除夕及春節", "description": "農曆春節補假"},
        "2027-03-01": {"name": "和平紀念日", "description": "和平紀念日補假"},
        "2027-04-02": {"name": "兒童節及民族掃墓節", "description": "兒童節補假"},
        "2027-04-05": {"name": "兒童節及民族掃墓節", "description": "民族掃墓節"},
        "2027-04-30": {"name": "勞動節", "description": "勞動節補假"},
        "2027-06-09": {"name": "端午節", "description": "依規定放假1日。"},
        "2027-09-15": {"name": "中秋節", "description": "依規定放假1日。"},
        "2027-10-11": {"name": "國慶日", "description": "國慶日補假"},
    }

    @classmethod
    def resolve_market(cls, ticker_or_market: Optional[str] = "TW") -> str:
        """
        解析股票代號或市場字串所屬市場：
        - 包含 .TW / .TWO 或符合 ^\\d{4,6}(\\.TW|\\.TWO)?$ 或屬於 TWSE/TPEX/TW -> "TW"
        - 其餘代號 (如 AAPL, CAT, MSFT) 或 US -> "US"
        """
        if not ticker_or_market:
            return "TW"
        val = str(ticker_or_market).strip().upper()
        if val in ("TW", "TWSE", "TPEX", "TAIWAN") or any(k in val for k in ("台股", "台灣")):
            return "TW"
        if val in ("JP", "JPX", "TSE", "NIKKEI", "NIKKEI225", "N225", "JP225", "JAPAN") or any(k in val for k in ("日經", "日股")):
            return "JP"
        if val.endswith(".T") or re.match(r"^\d{4,5}\.T$", val):
            return "JP"
        if ".TW" in val or ".TWO" in val:
            return "TW"
        if re.match(r"^\d{4,6}(\.TW|\.TWO)?$", val):
            return "TW"
        return "US"

    @classmethod
    def _ensure_cache_dir(cls):
        """確保本機快取目錄存在"""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def _parse_minguo_or_iso_date(cls, date_str: str) -> Optional[datetime.date]:
        """
        解析民國年 (如 1150101 -> 2026-01-01) 或標準 ISO / YYYYMMDD 日期
        """
        s = str(date_str).strip()
        if not s:
            return None
        # 處理標準 ISO 格式
        if "-" in s or "/" in s:
            s_iso = s.replace("/", "-")
            try:
                return datetime.date.fromisoformat(s_iso)
            except ValueError:
                pass
        # 處理 8 碼 YYYYMMDD
        if len(s) == 8 and s.isdigit():
            try:
                return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))
            except ValueError:
                pass
        # 處理民國年：7 碼 (如 1150101) 或 6 碼 (如 990101)
        if s.isdigit():
            if len(s) == 7:
                y = int(s[:3]) + 1911
                m = int(s[3:5])
                d = int(s[5:])
                try:
                    return datetime.date(y, m, d)
                except ValueError:
                    pass
            elif len(s) == 6:
                y = int(s[:2]) + 1911
                m = int(s[2:4])
                d = int(s[4:])
                try:
                    return datetime.date(y, m, d)
                except ValueError:
                    pass
        return None

    @classmethod
    def _fetch_twse_openapi(cls) -> List[Dict[str, Any]]:
        """從證交所官方 OpenAPI 抓取行事曆"""
        url = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        import requests
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def _load_tw_holidays(cls, year: int) -> Dict[str, Dict[str, str]]:
        """
        載入指定年份的台股休市行事曆字典：
        { "YYYY-MM-DD": {"date": "YYYY-MM-DD", "name": str, "description": str, "market": "TW"} }
        三層防禦：
        1. 記憶體快取
        2. 30 天本機磁碟快取 (data/cache/market_calendar/twse_holidays_{year}.json)
        3. TWSE OpenAPI 即時抓取 -> 若失敗則降級使用過期快取或內嵌 2024~2027 備援資料
        """
        if year in cls._TW_HOLIDAY_CACHE:
            return cls._TW_HOLIDAY_CACHE[year]

        cls._ensure_cache_dir()
        cache_path = os.path.join(cls.CACHE_DIR, f"twse_holidays_{year}.json")

        # 檢查磁碟快取有效性 (30 天以內)
        if os.path.exists(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                if time.time() - mtime < cls.CACHE_EXPIRATION_SECONDS:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_doc = json.load(f)
                    holidays_list = cached_doc.get("holidays", [])
                    holidays_map = {item["date"]: item for item in holidays_list}
                    cls._TW_HOLIDAY_CACHE[year] = holidays_map
                    return holidays_map
            except Exception as e:
                logger.warning(f"⚠️ 讀取台股行事曆本機快取 ({cache_path}) 失敗: {e}")

        # 快取不存在或已逾期，嘗試從 TWSE OpenAPI 抓取
        api_data = None
        try:
            api_data = cls._fetch_twse_openapi()
        except Exception as e:
            logger.warning(f"⚠️ 請求 TWSE OpenAPI 行事曆失敗: {e}，準備降級處理")

        if api_data:
            # 依年份歸納 OpenAPI 結果
            grouped_holidays: Dict[int, Dict[str, Dict[str, str]]] = {}
            grouped_trading: Dict[int, set] = {}

            for item in api_data:
                name = str(item.get("Name", "")).strip()
                desc = str(item.get("Description", "")).strip()
                raw_d = str(item.get("Date", "")).strip()
                parsed_d = cls._parse_minguo_or_iso_date(raw_d)
                if not parsed_d:
                    continue

                d_iso = parsed_d.strftime("%Y-%m-%d")
                y = parsed_d.year
                if y not in grouped_holidays:
                    grouped_holidays[y] = {}
                    grouped_trading[y] = set()

                # 規則分類：
                # 1. 若 Name 包含「最後交易」或「開始交易」-> 標記為交易日
                is_trading = "最後交易" in name or "開始交易" in name
                # 2. 若 Description 包含「放假」或「補假」或「無交易」，或 Name 包含「市場無交易」-> 判定為休市日
                is_holiday = (
                    any(kw in desc for kw in ("放假", "補假", "無交易"))
                    or "市場無交易" in name
                    or "放假" in name
                ) and not is_trading

                if is_holiday:
                    clean_desc = desc.replace("<br>", " ").strip()
                    grouped_holidays[y][d_iso] = {
                        "date": d_iso,
                        "name": name,
                        "description": clean_desc,
                        "market": "TW",
                    }
                elif is_trading:
                    grouped_trading[y].add(d_iso)

            # 將各年份存入磁碟快取與記憶體快取
            for y, h_map in grouped_holidays.items():
                cls._TW_HOLIDAY_CACHE[y] = h_map
                cls._TW_SPECIAL_TRADING_CACHE[y] = grouped_trading.get(y, set())
                y_cache_file = os.path.join(cls.CACHE_DIR, f"twse_holidays_{y}.json")
                try:
                    payload = {
                        "year": y,
                        "cached_at": time.time(),
                        "holidays": list(h_map.values()),
                    }
                    with open(y_cache_file, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as ex:
                    logger.warning(f"⚠️ 寫入台股行事曆快取 ({y_cache_file}) 失敗: {ex}")

            if year in cls._TW_HOLIDAY_CACHE:
                return cls._TW_HOLIDAY_CACHE[year]

        # 若 OpenAPI 未包含該年，但磁碟有過期快取，則寬容使用
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_doc = json.load(f)
                holidays_list = cached_doc.get("holidays", [])
                holidays_map = {item["date"]: item for item in holidays_list}
                cls._TW_HOLIDAY_CACHE[year] = holidays_map
                logger.info(f"ℹ️ 使用已過期但有效之台股磁碟行事曆快取 ({year})")
                return holidays_map
            except Exception:
                pass

        # 降級使用內嵌備援字典
        fallback_map: Dict[str, Dict[str, str]] = {}
        for d_str, meta in cls.FALLBACK_TW_HOLIDAYS.items():
            if d_str.startswith(f"{year}-"):
                fallback_map[d_str] = {
                    "date": d_str,
                    "name": meta["name"],
                    "description": meta["description"],
                    "market": "TW",
                }
        cls._TW_HOLIDAY_CACHE[year] = fallback_map
        return fallback_map

    @classmethod
    def _easter_sunday(cls, year: int) -> datetime.date:
        """
        以標準 Meeus/Jones/Butcher 匿名公曆演算法精確推算指定年份之復活節星期日 (Easter Sunday)
        """
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return datetime.date(year, month, day)

    @classmethod
    def _observe_us_holiday(cls, dt: datetime.date) -> datetime.date:
        """
        美股官方法定放假補假規則 (Observation Rule)：
        - 週六 (5) 提前至週五 (4) 補假
        - 週日 (6) 延後至週一 (0) 補假
        """
        if dt.weekday() == 5:
            return dt - datetime.timedelta(days=1)
        if dt.weekday() == 6:
            return dt + datetime.timedelta(days=1)
        return dt

    @classmethod
    def _get_us_holidays(cls, year: int) -> Dict[str, Dict[str, str]]:
        """
        推算美股 (NYSE/NASDAQ) 全年度 10 大法定休市行事曆
        """
        if year in cls._US_HOLIDAY_CACHE:
            return cls._US_HOLIDAY_CACHE[year]

        holidays: Dict[str, Dict[str, str]] = {}

        def _add(d_obj: datetime.date, name: str, desc: str = "US Market / Federal Holiday"):
            d_str = d_obj.strftime("%Y-%m-%d")
            holidays[d_str] = {
                "date": d_str,
                "name": name,
                "description": desc,
                "market": "US",
            }

        # 1. New Year's Day (Jan 1)
        _add(cls._observe_us_holiday(datetime.date(year, 1, 1)), "New Year's Day")

        # 若下一年元旦逢週六，則當年前一年 12 月 31 日週五提前補假休市
        next_ny = datetime.date(year + 1, 1, 1)
        if next_ny.weekday() == 5:
            _add(datetime.date(year, 12, 31), "New Year's Day (Observed)")

        # 2. Martin Luther King Jr. Day (1 月第三個星期一)
        d_jan1 = datetime.date(year, 1, 1)
        m_mlk = d_jan1 + datetime.timedelta(days=(0 - d_jan1.weekday()) % 7) + datetime.timedelta(days=14)
        _add(m_mlk, "Martin Luther King Jr. Day")

        # 3. Washington's Birthday / Presidents' Day (2 月第三個星期一)
        d_feb1 = datetime.date(year, 2, 1)
        m_pres = d_feb1 + datetime.timedelta(days=(0 - d_feb1.weekday()) % 7) + datetime.timedelta(days=14)
        _add(m_pres, "Washington's Birthday")

        # 4. Good Friday (復活節前兩天週五)
        gf = cls._easter_sunday(year) - datetime.timedelta(days=2)
        _add(gf, "Good Friday")

        # 5. Memorial Day (5 月最後一個星期一)
        d_may31 = datetime.date(year, 5, 31)
        m_mem = d_may31 - datetime.timedelta(days=(d_may31.weekday() - 0) % 7)
        _add(m_mem, "Memorial Day")

        # 6. Juneteenth National Independence Day (6 月 19 日，自 2021 起)
        if year >= 2021:
            _add(cls._observe_us_holiday(datetime.date(year, 6, 19)), "Juneteenth National Independence Day")

        # 7. Independence Day (7 月 4 日)
        _add(cls._observe_us_holiday(datetime.date(year, 7, 4)), "Independence Day")

        # 8. Labor Day (9 月第一個星期一)
        d_sep1 = datetime.date(year, 9, 1)
        m_labor = d_sep1 + datetime.timedelta(days=(0 - d_sep1.weekday()) % 7)
        _add(m_labor, "Labor Day")

        # 9. Thanksgiving Day (11 月第四個星期四)
        d_nov1 = datetime.date(year, 11, 1)
        th_thanks = d_nov1 + datetime.timedelta(days=(3 - d_nov1.weekday()) % 7) + datetime.timedelta(days=21)
        _add(th_thanks, "Thanksgiving Day")

        # 10. Christmas Day (12 月 25 日)
        _add(cls._observe_us_holiday(datetime.date(year, 12, 25)), "Christmas Day")

        cls._US_HOLIDAY_CACHE[year] = holidays
        return holidays

    @classmethod
    def _observe_jp_holiday(cls, dt: datetime.date) -> datetime.date:
        """日本國定假日遇週日則週一補假 (振替休日)"""
        if dt.weekday() == 6:  # 週日
            return dt + datetime.timedelta(days=1)
        return dt

    @classmethod
    def _get_jp_holidays(cls, year: int) -> Dict[str, Dict[str, str]]:
        """東京證券交易所 (JPX/TSE) 國定節日與休市清冊"""
        if year in cls._JP_HOLIDAY_CACHE:
            return cls._JP_HOLIDAY_CACHE[year]

        holidays: Dict[str, Dict[str, str]] = {}

        def _add(dt: datetime.date, name: str):
            iso = dt.strftime("%Y-%m-%d")
            holidays[iso] = {"date": iso, "name": name, "description": f"東京證交所國定節日: {name}"}

        # 1. 元旦與正月休み (1月1日 ~ 1月3日)
        _add(datetime.date(year, 1, 1), "元日")
        _add(datetime.date(year, 1, 2), "正月休市")
        _add(datetime.date(year, 1, 3), "正月休市")

        # 2. 成人の日 (1 月第二個星期一)
        d_jan1 = datetime.date(year, 1, 1)
        m_adult = d_jan1 + datetime.timedelta(days=(0 - d_jan1.weekday()) % 7) + datetime.timedelta(days=7)
        _add(m_adult, "成人の日")

        # 3. 建國記念之日 (2月11日)
        _add(cls._observe_jp_holiday(datetime.date(year, 2, 11)), "建國記念之日")

        # 4. 天皇誕生日 (2月23日)
        _add(cls._observe_jp_holiday(datetime.date(year, 2, 23)), "天皇誕生日")

        # 5. 春分之日 (~3月20或21日，官方天文推算)
        spring_day = int(20.8431 + 0.242194 * (year - 1980) - int((year - 1980) / 4))
        _add(cls._observe_jp_holiday(datetime.date(year, 3, spring_day)), "春分之日")

        # 6. 昭和之日 (4月29日)
        _add(cls._observe_jp_holiday(datetime.date(year, 4, 29)), "昭和之日")

        # 7. 黃金週連休 (5月3日 憲法記念日、5月4日 綠之日、5月5日 兒童之日)
        _add(cls._observe_jp_holiday(datetime.date(year, 5, 3)), "憲法記念日")
        _add(cls._observe_jp_holiday(datetime.date(year, 5, 4)), "綠之日")
        _add(cls._observe_jp_holiday(datetime.date(year, 5, 5)), "兒童之日")
        if datetime.date(year, 5, 3).weekday() == 6 or datetime.date(year, 5, 4).weekday() == 6:
            _add(datetime.date(year, 5, 6), "黃金週補假 (振替休日)")

        # 8. 海之日 (7 月第三個星期一)
        d_jul1 = datetime.date(year, 7, 1)
        m_marine = d_jul1 + datetime.timedelta(days=(0 - d_jul1.weekday()) % 7) + datetime.timedelta(days=14)
        _add(m_marine, "海之日")

        # 9. 山之日 (8月11日)
        _add(cls._observe_jp_holiday(datetime.date(year, 8, 11)), "山之日")

        # 10. 敬老之日 (9 月第三個星期一)
        d_sep1 = datetime.date(year, 9, 1)
        m_respect = d_sep1 + datetime.timedelta(days=(0 - d_sep1.weekday()) % 7) + datetime.timedelta(days=14)
        _add(m_respect, "敬老之日")

        # 11. 秋分之日 (~9月22或23日，官方天文推算)
        autumn_day = int(23.2488 + 0.242194 * (year - 1980) - int((year - 1980) / 4))
        _add(cls._observe_jp_holiday(datetime.date(year, 9, autumn_day)), "秋分之日")

        # 祝日法第3條第3項：若敬老之日 (週一) 與秋分之日 (週三) 間夾週二，則週二自動成為「國民之休日」
        if m_respect.weekday() == 0 and autumn_day == 23 and datetime.date(year, 9, 23).weekday() == 2:
            _add(datetime.date(year, 9, 22), "國民之休日")

        # 12. 運動之日 (10 月第二個星期一)
        d_oct1 = datetime.date(year, 10, 1)
        m_sports = d_oct1 + datetime.timedelta(days=(0 - d_oct1.weekday()) % 7) + datetime.timedelta(days=7)
        _add(m_sports, "運動之日")

        # 13. 文化之日 (11月3日)
        _add(cls._observe_jp_holiday(datetime.date(year, 11, 3)), "文化之日")

        # 14. 勤勞感謝之日 (11月23日)
        _add(cls._observe_jp_holiday(datetime.date(year, 11, 23)), "勤勞感謝之日")

        # 15. 年末休市 (12月31日)
        _add(datetime.date(year, 12, 31), "大晦日 (年末休市)")

        cls._JP_HOLIDAY_CACHE[year] = holidays
        return holidays

    @classmethod
    def _normalize_date(
        cls,
        date_val: Union[datetime.date, datetime.datetime, str, None] = None,
        market: str = "TW",
    ) -> datetime.date:
        """
        標準化各類輸入日期為 datetime.date 物件。
        若為 None，預設依市場時區取今日日期 (TW: Asia/Taipei, JP: Asia/Tokyo, US: America/New_York)
        """
        if date_val is None:
            try:
                import zoneinfo
                if market == "TW":
                    tz_name = "Asia/Taipei"
                elif market == "JP":
                    tz_name = "Asia/Tokyo"
                else:
                    tz_name = "America/New_York"
                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception:
                offset = 8 if market == "TW" else (9 if market == "JP" else -4)
                tz = datetime.timezone(datetime.timedelta(hours=offset))
            return datetime.datetime.now(tz).date()

        if isinstance(date_val, datetime.datetime):
            return date_val.date()
        if isinstance(date_val, datetime.date):
            return date_val
        if isinstance(date_val, str):
            parsed = cls._parse_minguo_or_iso_date(date_val)
            if parsed:
                return parsed
        raise ValueError(f"無法解析有效日期格式: {date_val!r}")

    @classmethod
    def is_trading_day(
        cls,
        date_val: Union[datetime.date, datetime.datetime, str, None] = None,
        market_or_ticker: str = "TW",
    ) -> bool:
        """
        判斷指定日期是否為開盤交易日 (排除週末與市場休市日)
        """
        market = cls.resolve_market(market_or_ticker)
        cur = cls._normalize_date(date_val, market)
        date_iso = cur.strftime("%Y-%m-%d")

        if market == "TW":
            holidays = cls._load_tw_holidays(cur.year)
            # 若為公告休市日
            if date_iso in holidays:
                return False
            # 若為特殊開市補班日 (例週六開市)
            special_trading = cls._TW_SPECIAL_TRADING_CACHE.get(cur.year, set())
            if date_iso in special_trading:
                return True
            # 一般週末週六 (5) 與週日 (6) 休市
            if cur.weekday() >= 5:
                return False
            return True
        elif market == "JP":
            # 東證日股
            if cur.weekday() >= 5:
                return False
            jp_holidays = cls._get_jp_holidays(cur.year)
            if date_iso in jp_holidays:
                return False
            return True
        else:
            # 美股
            if cur.weekday() >= 5:
                return False
            us_holidays = cls._get_us_holidays(cur.year)
            if date_iso in us_holidays:
                return False
            return True

    @classmethod
    def get_trading_day_status(
        cls,
        date_val: Union[datetime.date, datetime.datetime, str, None] = None,
        market_or_ticker: str = "TW",
    ) -> Dict[str, Any]:
        """
        取得指定日期之交易日完整狀態報告：
        回傳:
        {
            "is_trading_day": bool,
            "market": str,
            "date": "YYYY-MM-DD",
            "reason": str,
            "next_trading_day": "YYYY-MM-DD",
            "previous_trading_day": "YYYY-MM-DD"
        }
        """
        market = cls.resolve_market(market_or_ticker)
        cur = cls._normalize_date(date_val, market)
        date_iso = cur.strftime("%Y-%m-%d")
        is_trading = cls.is_trading_day(cur, market_or_ticker=market)

        reason = "開盤交易日 (Trading Day)"
        if not is_trading:
            if cur.weekday() == 5:
                reason = "週末休市 (週六)"
            elif cur.weekday() == 6:
                reason = "週末休市 (週日)"
            else:
                if market == "TW":
                    holidays = cls._load_tw_holidays(cur.year)
                    item = holidays.get(date_iso)
                    if item:
                        reason = item["name"]
                        if item.get("description") and item["description"] not in item["name"]:
                            reason = f"{item['name']} ({item['description']})"
                    else:
                        reason = "市場休市日 (Holiday)"
                elif market == "JP":
                    jp_holidays = cls._get_jp_holidays(cur.year)
                    item = jp_holidays.get(date_iso)
                    reason = item["name"] if item else "東證市場休市日"
                else:
                    us_holidays = cls._get_us_holidays(cur.year)
                    item = us_holidays.get(date_iso)
                    reason = item["name"] if item else "美股市場休市日"

        next_day = cls.get_next_trading_day(cur, market_or_ticker=market).strftime("%Y-%m-%d")
        prev_day = cls.get_previous_trading_day(cur, market_or_ticker=market).strftime("%Y-%m-%d")

        return {
            "is_trading_day": is_trading,
            "market": market,
            "date": date_iso,
            "reason": reason,
            "next_trading_day": next_day,
            "previous_trading_day": prev_day,
        }

    @classmethod
    def get_next_trading_day(
        cls,
        date_val: Union[datetime.date, datetime.datetime, str, None] = None,
        market_or_ticker: str = "TW",
    ) -> datetime.date:
        """取得指定日期的下一個開盤交易日 (T+1)"""
        market = cls.resolve_market(market_or_ticker)
        cur = cls._normalize_date(date_val, market)
        step = 0
        while step < 120:
            cur += datetime.timedelta(days=1)
            step += 1
            if cls.is_trading_day(cur, market_or_ticker=market):
                return cur
        return cur

    @classmethod
    def get_previous_trading_day(
        cls,
        date_val: Union[datetime.date, datetime.datetime, str, None] = None,
        market_or_ticker: str = "TW",
    ) -> datetime.date:
        """取得指定日期的前一個開盤交易日 (T-1)"""
        market = cls.resolve_market(market_or_ticker)
        cur = cls._normalize_date(date_val, market)
        step = 0
        while step < 120:
            cur -= datetime.timedelta(days=1)
            step += 1
            if cls.is_trading_day(cur, market_or_ticker=market):
                return cur
        return cur

    @classmethod
    def get_recent_trading_days(
        cls,
        count: int = 20,
        reference_dt: Optional[datetime.datetime] = None,
        market_or_ticker: str = "TW",
    ) -> List[str]:
        """
        推算最近 count 個開盤交易日 (嚴格排除週末與國定假日) 之 YYYYMMDD 日期字串清單 (由近至遠排列)。
        遵守盤後出表截止時間門檻：
        - 台股 (TWSE): 15:30 前籌碼與收盤數據尚未齊備，起算基準往前推算一日。
        - 日股 (TSE): 東京時間 15:30 前不包含當日。
        - 美股 (US): 美東時間 16:00 前不包含當日。
        """
        market = cls.resolve_market(market_or_ticker)
        if reference_dt is None:
            try:
                import zoneinfo
                if market == "TW":
                    tz_name = "Asia/Taipei"
                elif market == "JP":
                    tz_name = "Asia/Tokyo"
                else:
                    tz_name = "America/New_York"
                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception:
                tz_offset = 8 if market == "TW" else (9 if market == "JP" else -4)
                tz = datetime.timezone(datetime.timedelta(hours=tz_offset))
            ref_now = datetime.datetime.now(tz)
        else:
            ref_now = reference_dt

        cur = ref_now.date()
        if market == "TW":
            # 證交所 T86 約於 15:00~15:30 始出表，在 15:30 前（含午夜換日與 08:00 盤前）不包含當日
            if ref_now.hour < 15 or (ref_now.hour == 15 and ref_now.minute < 30):
                cur -= datetime.timedelta(days=1)
        elif market == "JP":
            # 東證 TSE 於東京時間 15:30 收盤，在 15:30 前不包含當日
            if ref_now.hour < 15 or (ref_now.hour == 15 and ref_now.minute < 30):
                cur -= datetime.timedelta(days=1)
        else:
            # 美股於美東時間 16:00 收盤
            if ref_now.hour < 16:
                cur -= datetime.timedelta(days=1)

        days: List[str] = []
        max_lookback = count * 7 + 100
        step = 0
        while len(days) < count and step < max_lookback:
            if cls.is_trading_day(cur, market_or_ticker=market):
                days.append(cur.strftime("%Y%m%d"))
            cur -= datetime.timedelta(days=1)
            step += 1
        return days

    @classmethod
    def get_holiday_list(cls, year: int, market: str = "TW") -> List[Dict[str, str]]:
        """
        取得指定市場與年份之全年度休市清冊 (依日期遞增排序)
        """
        resolved_market = cls.resolve_market(market)
        if resolved_market == "TW":
            holidays = cls._load_tw_holidays(year)
        elif resolved_market == "JP":
            holidays = cls._get_jp_holidays(year)
        else:
            holidays = cls._get_us_holidays(year)

        items = list(holidays.values())
        items.sort(key=lambda x: x["date"])
        return items
