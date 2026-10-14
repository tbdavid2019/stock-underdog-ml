"""
data/earnings_fetcher.py - 跨市場法說會資料擷取與快取管道 (Earnings Call Fetcher)

支援以 BigGo 財經與公開資訊觀測站 (MOPS) 作為主要聚合源，提供完整台股 (TWSE/TPEx)
與美股法說會時程、官方簡報 PDF、Webcast 直播/回放與重大訊息紀錄。
具備多視窗自動翻頁、官方簡報直連、以及本地快取降級備援機制，遵守「優先抓取最新、失敗時自動降級本地快取」之鐵律。
"""

import os
import re
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple
import urllib.request
import urllib.parse

from core.config import config

logger = logging.getLogger("stock_app.earnings")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "earnings")

# 台股前 50 大與美股科技巨頭權值股清單
TW_WEIGHT_STOCKS = {
    "2330.TW", "2317.TW", "2454.TW", "2382.TW", "2308.TW", "2412.TW", "2881.TW",
    "2882.TW", "2303.TW", "3711.TW", "3231.TW", "2357.TW", "2891.TW", "2886.TW",
    "2884.TW", "1216.TW", "2603.TW", "2892.TW", "2002.TW", "2885.TW", "3008.TW"
}

US_WEIGHT_STOCKS = {
    "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO",
    "AMD", "TSM", "ASML", "ORCL", "NFLX", "CRM", "INTC", "QCOM", "TXN", "MU"
}

# 美股科技巨頭官方 IR 入口
US_IR_WEBCAST_MAP = {
    "NVDA": "https://investor.nvidia.com/events-and-presentations/",
    "AAPL": "https://www.apple.com/investor/earnings-call/",
    "MSFT": "https://www.microsoft.com/investor/events/",
    "GOOGL": "https://abc.xyz/investor/earnings/",
    "AMZN": "https://ir.aboutamazon.com/events-and-presentations/",
    "META": "https://investor.atmeta.com/investor-events/",
    "TSLA": "https://ir.tesla.com/events-and-presentations",
    "AMD": "https://ir.amd.com/news-events/ir-calendar",
    "TSM": "https://investor.tsmc.com/chinese/quarterly-results"
}

BIGGO_API_BASE = "https://api.biggo.com/api/v1/finance"
BIGGO_HEADERS = {
    "Accept": "application/json",
    "Site": "finance.biggo.com.tw",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def generate_date_windows(
    as_of: Optional[datetime.date] = None,
    past_windows: int = 2,
    future_windows: int = 4,
    window_days: int = 6
) -> List[Tuple[str, str]]:
    """
    產生符合 BigGo API 單次查詢區間限制 (<= 7-8天) 的滾動日期視窗。
    預設包含過去 12 天至未來 24 天，確保歷史回放與近期/即將召開法說會完整覆蓋。
    """
    base = as_of or datetime.date.today()
    windows = []
    for i in range(-past_windows, future_windows):
        s = base + datetime.timedelta(days=i * window_days)
        e = base + datetime.timedelta(days=(i + 1) * window_days)
        windows.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return windows


def parse_roc_date(date_str: str) -> str:
    """轉換民國年日期 (如 113/10/17 或 1131017) 為西元年 YYYY-MM-DD"""
    if not date_str:
        return ""
    clean = re.sub(r"[^\d]", "", str(date_str).strip())
    if len(clean) == 7:  # 1131017
        roc_year = int(clean[:3])
        ad_year = roc_year + 1911
        return f"{ad_year:04d}-{clean[3:5]}-{clean[5:7]}"
    elif len(clean) == 8:  # 20261017
        return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"
    elif "/" in date_str or "-" in date_str:
        parts = re.split(r"[/ -]", date_str.strip())
        if len(parts) >= 3:
            y = int(parts[0])
            if y < 1900:
                y += 1911
            return f"{y:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    return date_str.strip()


def normalize_tw_ticker(raw_code: str) -> str:
    """正規化台股代碼 (如 2330 -> 2330.TW)"""
    code = str(raw_code).strip().upper()
    if not code:
        return ""
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    if "." in code:
        return code
    # 若為 4 位數字或標準數字代碼，預設標為 .TW
    if re.match(r"^\d{4,5}$", code):
        return f"{code}.TW"
    return code


def is_tw_weight_stock(ticker: str) -> bool:
    """判斷台股是否為重量級權值股"""
    clean = ticker.strip().upper()
    bare = clean.split(".")[0]
    return clean in TW_WEIGHT_STOCKS or bare in {s.split(".")[0] for s in TW_WEIGHT_STOCKS}


def is_us_weight_stock(ticker: str) -> bool:
    """判斷美股是否為重量級標的"""
    return ticker.strip().upper() in US_WEIGHT_STOCKS


def fetch_biggo_slide_url(call_id: str, timeout: int = 3) -> Optional[str]:
    """向 BigGo File API 查詢法說會簡報 PDF 連結"""
    try:
        url = f"{BIGGO_API_BASE}/stock/earnings-calls/file?type=slides&call_id={urllib.parse.quote(call_id)}&region=tw"
        req = urllib.request.Request(url, headers=BIGGO_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            slides = data.get("data", {}).get("list", [])
            if slides and isinstance(slides, list):
                return slides[0].get("url")
            return data.get("data", {}).get("url")
    except Exception as e:
        logger.debug(f"查詢簡報 PDF 失敗 ({call_id}): {e}")
        return None


class EarningsCallFetcher:
    """跨市場法人說明會資料獲取器"""

    def __init__(self, db_manager: Optional[Any] = None):
        if db_manager is not None:
            self.db = db_manager
        else:
            try:
                from data.duckdb_manager import DuckDBManager
                self.db = DuckDBManager()
            except Exception as e:
                logger.warning(f"DuckDBManager 初始化跳過: {e}")
                self.db = None
        _ensure_cache_dir()

    def fetch_biggo_calls(
        self,
        market: str = "tw",
        past_windows: int = 2,
        future_windows: int = 4,
        enrich_slides_limit: int = 8
    ) -> List[Dict[str, Any]]:
        """
        透過 BigGo 財經公開 API 多視窗聚合法人說明會清單 (包含台灣 TWSE/TPEx 與美股)
        """
        m_lower = market.lower()
        windows = generate_date_windows(past_windows=past_windows, future_windows=future_windows)
        cache_file = os.path.join(CACHE_DIR, f"biggo_{m_lower}_calls.json")
        raw_items_map: Dict[str, Dict[str, Any]] = {}
        network_success = False

        for w_start, w_end in windows:
            url = f"{BIGGO_API_BASE}/stock/earnings-calls/list?start_date={w_start}&end_date={w_end}&market={m_lower}&region=tw&stock_filter=all"
            try:
                req = urllib.request.Request(url, headers=BIGGO_HEADERS)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("result"):
                        network_success = True
                        items = payload.get("data", {}).get("list", [])
                        for it in items:
                            cid = it.get("call_id")
                            if cid and cid not in raw_items_map:
                                raw_items_map[cid] = it
            except Exception as e:
                logger.debug(f"BigGo 視窗抓取失敗 ({w_start} ~ {w_end}): {e}")

        # 若網路成功且有資料，寫入本地快取
        if network_success and raw_items_map:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(list(raw_items_map.values()), f, ensure_ascii=False, indent=2)
                logger.info(f"✅ BigGo {market.upper()} 抓取成功: {len(raw_items_map)} 筆法說會紀錄")
            except Exception as e:
                logger.warning(f"寫入 BigGo {market.upper()} 快取失敗: {e}")
        elif not raw_items_map:
            # 嘗試讀取本地快取
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_list = json.load(f)
                    for it in cached_list:
                        cid = it.get("call_id")
                        if cid:
                            raw_items_map[cid] = it
                    logger.info(f"📁 已自本地快取載入 {len(raw_items_map)} 筆 BigGo {market.upper()} 法說會")
                except Exception as cf_err:
                    logger.error(f"❌ 讀取 BigGo 快取失敗: {cf_err}")

        if not raw_items_map:
            return []

        # 解析為標準法說會結構
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        parsed_records: List[Dict[str, Any]] = []
        enriched_count = 0

        for it in raw_items_map.values():
            call_id = it.get("call_id", "")
            stock_id = (it.get("stock_id") or "").strip().upper()
            if not stock_id:
                continue

            name = (it.get("name") or "").strip()
            title = (it.get("title") or "").strip()
            call_date = it.get("date", "")

            # 市場與代碼判斷
            if m_lower == "tw":
                mkt = "TW"
                ticker = normalize_tw_ticker(stock_id)
                tz = "Asia/Taipei"
                is_weight = is_tw_weight_stock(ticker)
                ir_url = f"https://finance.biggo.com.tw/earnings-call?q={urllib.parse.quote(ticker)}"
                source = "BigGo / MOPS"
            else:
                mkt = "US"
                ticker = stock_id
                tz = "America/New_York"
                is_weight = is_us_weight_stock(ticker)
                ir_url = US_IR_WEBCAST_MAP.get(ticker) or f"https://finance.biggo.com.tw/earnings-call?q={urllib.parse.quote(ticker)}"
                source = "BigGo / US IR"

            meeting_type = "online" if (it.get("has_video") or it.get("has_audio")) else "physical"
            loc = "線上視訊/電話法說會" if meeting_type == "online" else "法人說明會 (實體/線上同步)"

            # 簡報 PDF 連結解析 (對近期/當天且有簡報者進行連網簽名解析)
            pres_url = ""
            if it.get("has_slides"):
                # 對近期或今天場次優先抓取 signed S3 PDF
                if call_date >= today_str and enriched_count < enrich_slides_limit:
                    slide_link = fetch_biggo_slide_url(call_id)
                    if slide_link:
                        pres_url = slide_link
                        enriched_count += 1
                if not pres_url:
                    pres_url = ir_url

            parsed_records.append({
                "id": call_id,
                "market": mkt,
                "ticker": ticker,
                "name": name or title,
                "call_date": call_date,
                "call_time": "14:00" if mkt == "TW" else "17:00",
                "timezone": tz,
                "meeting_type": meeting_type,
                "location": loc,
                "webcast_url": ir_url,
                "presentation_url": pres_url,
                "transcript_url": ir_url if it.get("has_transcript") else "",
                "language": "zh-TW" if mkt == "TW" else "en",
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": is_weight,
                "remarks": title,
                "source": source
            })

        # 依日期由近到遠排序
        parsed_records.sort(key=lambda x: (x["call_date"], x["call_time"]))
        return parsed_records

    def fetch_twse_calls(self) -> List[Dict[str, Any]]:
        """
        獲取台灣證交所 (TWSE) 上市公司法說會一覽表
        作為 BigGo 的次級備援管道 (若 TWSE OpenAPI 恢復)
        """
        url = "https://openapi.twse.com.tw/v1/opendata/t100sb02_1"
        cache_file = os.path.join(CACHE_DIR, "twse_calls.json")
        data = None

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; 888StockQuant/2.4)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data, list):
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ TWSE OpenAPI 抓取成功: {len(data)} 筆法說會")
        except Exception as e:
            logger.debug(f"⚠️ TWSE OpenAPI 暫不可用: {e}")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as cf_err:
                    logger.error(f"❌ 讀取 TWSE 快取失敗: {cf_err}")

        if not data or not isinstance(data, list):
            return []

        parsed_records = []
        for item in data:
            raw_code = str(item.get("公司代號", "")).strip()
            if not raw_code:
                continue
            ticker = f"{raw_code}.TW"
            name = item.get("公司名稱", "").strip()
            call_date = parse_roc_date(item.get("召開日期", ""))
            call_time = str(item.get("召開時間", "")).strip()
            if len(call_time) == 4 and ":" not in call_time:
                call_time = f"{call_time[:2]}:{call_time[2:]}"

            pres_url = item.get("中文簡報", "") or item.get("英文簡報", "")
            video_url = item.get("影音連結", "") or item.get("影音回放", "")
            call_id = f"TW_{ticker}_{call_date.replace('-', '')}"
            is_weight = is_tw_weight_stock(ticker)

            parsed_records.append({
                "id": call_id,
                "market": "TW",
                "ticker": ticker,
                "name": name,
                "call_date": call_date,
                "call_time": call_time or "14:00",
                "timezone": "Asia/Taipei",
                "meeting_type": "online" if "線上" in str(item.get("地點", "")) else "physical",
                "location": item.get("地點", "").strip(),
                "webcast_url": video_url.strip(),
                "presentation_url": pres_url.strip(),
                "transcript_url": "",
                "language": "zh-TW",
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": is_weight,
                "remarks": item.get("法人說明會擇要", "").strip(),
                "source": "TWSE"
            })
        return parsed_records

    def fetch_tpex_calls(self) -> List[Dict[str, Any]]:
        """
        獲取證券櫃檯買賣中心 (TPEx) 上櫃公司法說會一覽表
        作為 BigGo 的次級備援管道
        """
        url = "https://www.tpex.org.tw/openapi/v1/mopsfin_t100sb02_1"
        cache_file = os.path.join(CACHE_DIR, "tpex_calls.json")
        data = None

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; 888StockQuant/2.4)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data, list):
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ TPEx OpenAPI 抓取成功: {len(data)} 筆法說會")
        except Exception as e:
            logger.debug(f"⚠️ TPEx OpenAPI 暫不可用: {e}")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as cf_err:
                    logger.error(f"❌ 讀取 TPEx 快取失敗: {cf_err}")

        if not data or not isinstance(data, list):
            return []

        parsed_records = []
        for item in data:
            raw_code = str(item.get("公司代號", "")).strip()
            if not raw_code:
                continue
            ticker = f"{raw_code}.TWO"
            name = item.get("公司名稱", "").strip()
            call_date = parse_roc_date(item.get("召開日期", ""))
            call_time = str(item.get("召開時間", "")).strip()
            if len(call_time) == 4 and ":" not in call_time:
                call_time = f"{call_time[:2]}:{call_time[2:]}"

            pres_url = item.get("中文簡報", "") or item.get("英文簡報", "")
            video_url = item.get("影音連結", "") or item.get("影音回放", "")
            call_id = f"TW_{ticker}_{call_date.replace('-', '')}"

            parsed_records.append({
                "id": call_id,
                "market": "TW",
                "ticker": ticker,
                "name": name,
                "call_date": call_date,
                "call_time": call_time or "14:00",
                "timezone": "Asia/Taipei",
                "meeting_type": "online" if "線上" in str(item.get("地點", "")) else "physical",
                "location": item.get("地點", "").strip(),
                "webcast_url": video_url.strip(),
                "presentation_url": pres_url.strip(),
                "transcript_url": "",
                "language": "zh-TW",
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": is_tw_weight_stock(ticker),
                "remarks": item.get("法人說明會擇要", "").strip(),
                "source": "TPEX"
            })
        return parsed_records

    def fetch_tw_calls(self) -> List[Dict[str, Any]]:
        """
        獲取台灣上市與上櫃公司法人說明會 (整合 BigGo MOPS 與官方備援)
        """
        records = self.fetch_biggo_calls(market="tw")
        if records:
            return records

        # 備援：TWSE 與 TPEx 原生 OpenAPI
        twse_records = self.fetch_twse_calls()
        tpex_records = self.fetch_tpex_calls()
        fallback_records = twse_records + tpex_records
        if fallback_records:
            return fallback_records

        return []

    def fetch_us_calls(self) -> List[Dict[str, Any]]:
        """
        獲取美股主要科技巨頭與 S&P 500 指標股之法說會與財報排程
        """
        records = self.fetch_biggo_calls(market="us")
        if records:
            return records

        cache_file = os.path.join(CACHE_DIR, "us_calls.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if cached and isinstance(cached, list):
                        return cached
            except Exception as e:
                logger.warning(f"讀取美股法說會快取失敗: {e}")

        # 無可驗證的快取時保持空集合，避免把推測日期當成正式法說會資料。
        logger.warning("⚠️ 無美股法說會快取，略過未驗證的預置排程")
        return []

    def fetch_ticker_calls(self, ticker: str) -> List[Dict[str, Any]]:
        """
        依單一股票代碼 (如 2330.TW 或 NVDA) 線上查詢該標的之歷史與預約法說會
        並寫入 DuckDB 供後續快速快取檢索。
        """
        clean = ticker.strip().upper()
        clean_query = clean.split(".")[0]
        results: List[Dict[str, Any]] = []

        url = f"{BIGGO_API_BASE}/stock/earnings-calls/list?q={urllib.parse.quote(clean_query)}&page=1&size=50"
        try:
            req = urllib.request.Request(url, headers=BIGGO_HEADERS)
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("result"):
                    items = data.get("data", {}).get("list", [])
                    is_tw = ".TW" in clean or re.match(r"^\d{4,5}", clean_query)
                    for it in items:
                        cid = it.get("call_id")
                        stock_id = it.get("stock_id", "")
                        # 簡單過濾符合代碼
                        if clean_query not in stock_id.upper():
                            continue
                        parsed_t = normalize_tw_ticker(stock_id) if is_tw else stock_id
                        results.append({
                            "id": cid,
                            "market": "TW" if is_tw else "US",
                            "ticker": parsed_t,
                            "name": it.get("name") or it.get("title", ""),
                            "call_date": it.get("date", ""),
                            "call_time": "14:00" if is_tw else "17:00",
                            "timezone": "Asia/Taipei" if is_tw else "America/New_York",
                            "meeting_type": "online" if (it.get("has_video") or it.get("has_audio")) else "physical",
                            "location": "線上視訊/電話法說會",
                            "webcast_url": f"https://finance.biggo.com.tw/earnings-call?q={urllib.parse.quote(stock_id)}",
                            "presentation_url": f"https://finance.biggo.com.tw/earnings-call?q={urllib.parse.quote(stock_id)}",
                            "transcript_url": "",
                            "language": "zh-TW" if is_tw else "en",
                            "eps_estimate": None,
                            "eps_actual": None,
                            "revenue_estimate": None,
                            "revenue_actual": None,
                            "is_weight_stock": is_tw_weight_stock(parsed_t) if is_tw else is_us_weight_stock(parsed_t),
                            "remarks": it.get("title", ""),
                            "source": "BigGo / MOPS" if is_tw else "BigGo / US IR"
                        })
        except Exception as e:
            logger.debug(f"查詢標的法說會失敗 ({ticker}): {e}")

        if results and self.db and hasattr(self.db, "save_earnings_calls_batch"):
            try:
                self.db.save_earnings_calls_batch(results)
            except Exception as e:
                logger.debug(f"快取個股法說會至 DuckDB 失敗: {e}")

        return results

    def fetch_all_and_sync(self) -> int:
        """執行跨市場法說會抓取並存入 DuckDB"""
        tw_records = self.fetch_tw_calls()
        us_records = self.fetch_us_calls()

        all_records = tw_records + us_records
        if not all_records:
            logger.warning("⚠️ 未抓取到任何跨市場法說會紀錄")
            return 0

        if not self.db or not hasattr(self.db, "save_earnings_calls_batch"):
            logger.warning("⚠️ 無資料庫管理器可儲存法說會資料")
            return len(all_records)

        saved_count = self.db.save_earnings_calls_batch(all_records)
        logger.info(f"🎉 跨市場法說會資料同步完成: 共寫入 {saved_count} 筆 (台股 {len(tw_records)} / 美股 {len(us_records)})")
        return saved_count


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    fetcher = EarningsCallFetcher()
    count = fetcher.fetch_all_and_sync()
    print(f"Sync complete: {count} earnings calls written.")
