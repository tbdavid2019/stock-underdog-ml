"""
data/earnings_fetcher.py - 跨市場法說會資料擷取與快取管道 (Earnings Call Fetcher)

整合 BigGo 財經 (https://finance.biggo.com.tw/earnings-call) 官方後端 API，
提供完整跨市場（台灣 TWSE/TPEx、美股 US、日股 JP、韓股 KR 與其他市場）法人說明會時程、
管理層營收/毛利率/資本支出官方指引 (Guidance)、簡報 PDF 原始下載、Webcast 直播回放與逐字稿。
具備多視窗並行抓取、多市場智慧識別、官方簡報直連、以及本地快取/原生 OpenAPI 降級備援機制，
遵守「優先抓取最新、失敗時自動降級本地快取」之鐵律。
"""

import os
import re
import json
import logging
import datetime
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
import urllib.request
import urllib.parse

logger = logging.getLogger("stock_app.earnings")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "earnings")
VERIFIED_EARNINGS_FILE = os.path.join(os.path.dirname(__file__), "verified_earnings.json")


def load_verified_corporate_earnings() -> List[Dict[str, Any]]:
    """載入官方已確認之全球指標企業 (如台積電 2330.TW、聯發科 2454.TW、鴻海 2317.TW、美股巨頭等) 法說會基準庫"""
    if os.path.exists(VERIFIED_EARNINGS_FILE):
        try:
            with open(VERIFIED_EARNINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"讀取官方基準法說會檔案失敗: {e}")
    return []


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

JP_WEIGHT_STOCKS = {
    "7203.T", "9984.T", "6758.T", "8035.T", "6861.T", "6501.T", "4063.T", "6902.T", "8306.T"
}

KR_WEIGHT_STOCKS = {
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS", "068270.KS"
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
    past_windows: int = 3,
    future_windows: int = 3,
    window_days: int = 6
) -> List[Tuple[str, str]]:
    """
    產生符合 BigGo API 單次查詢區間限制 (<= 7-8天) 的滾動日期視窗。
    預設包含過去 18 天至未來 18 天（共 6 個 6 日視窗），對齊 stockbot 基準。
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
    if re.match(r"^\d{4,5}$", code):
        return f"{code}.TW"
    return code


def parse_biggo_stock_metadata(item: Dict[str, Any]) -> Dict[str, str]:
    """
    精確解析 BigGo 股票之市場類別與時區
    支援：台股 (TWSE/TPEx)、美股 (US)、日股 (JP)、韓股 (KR)、其他 (OTHER)
    """
    stock_id = str(item.get("stock_id") or "").strip().upper()
    region = str(item.get("region") or "").strip().lower()

    # 日本股市 (e.g. 7203.T, 9984.T, region == 'jp')
    if region == "jp" or stock_id.endswith(".T"):
        symbol = stock_id if stock_id.endswith(".T") else f"{stock_id}.T"
        return {
            "symbol": symbol,
            "market": "JP",
            "timezone": "Asia/Tokyo",
            "default_time": "15:00"
        }

    # 韓國股市 (e.g. 005930.KS, 000660.KS, region == 'kr')
    if region == "kr" or stock_id.endswith(".KS") or stock_id.endswith(".KQ"):
        symbol = stock_id
        if not (symbol.endswith(".KS") or symbol.endswith(".KQ")):
            symbol = f"{symbol}.KS"
        return {
            "symbol": symbol,
            "market": "KR",
            "timezone": "Asia/Seoul",
            "default_time": "14:00"
        }

    # 台灣上櫃 (TPEx)
    if stock_id.endswith(".TWO"):
        return {
            "symbol": stock_id,
            "market": "TW",
            "timezone": "Asia/Taipei",
            "default_time": "14:00"
        }

    # 台灣上市 (TWSE)
    if stock_id.endswith(".TW"):
        return {
            "symbol": stock_id,
            "market": "TW",
            "timezone": "Asia/Taipei",
            "default_time": "14:00"
        }

    # 台灣 4 碼純數字或 region tw
    if region == "tw" or re.match(r"^\d{4,5}$", stock_id):
        return {
            "symbol": normalize_tw_ticker(stock_id),
            "market": "TW",
            "timezone": "Asia/Taipei",
            "default_time": "14:00"
        }

    # 美股標的 (NYSE / NASDAQ / S&P 500)
    if region == "us" or re.match(r"^[A-Z]{1,5}$", stock_id):
        return {
            "symbol": stock_id or "US",
            "market": "US",
            "timezone": "America/New_York",
            "default_time": "17:00"
        }

    return {
        "symbol": stock_id or "OTHER",
        "market": "OTHER",
        "timezone": "Asia/Taipei",
        "default_time": "14:00"
    }


def is_weight_stock(ticker: str, market: str) -> bool:
    """判斷指定市場之標的是否為重量級指標權值股"""
    clean = ticker.strip().upper()
    bare = clean.split(".")[0]
    if market == "TW":
        return clean in TW_WEIGHT_STOCKS or bare in {s.split(".")[0] for s in TW_WEIGHT_STOCKS}
    elif market == "US":
        return clean in US_WEIGHT_STOCKS
    elif market == "JP":
        return clean in JP_WEIGHT_STOCKS or bare in {s.split(".")[0] for s in JP_WEIGHT_STOCKS}
    elif market == "KR":
        return clean in KR_WEIGHT_STOCKS or bare in {s.split(".")[0] for s in KR_WEIGHT_STOCKS}
    return False


def is_tw_weight_stock(ticker: str) -> bool:
    """向後相容：判斷台股是否為重量級權值股"""
    return is_weight_stock(ticker, "TW")


def is_us_weight_stock(ticker: str) -> bool:
    """向後相容：判斷美股是否為重量級權值股"""
    return is_weight_stock(ticker, "US")


def extract_financial_period(title: str, date: str) -> Optional[str]:
    """從法說會標題或日期中提取財務季度 (例如 FY2026 Q3、2026 Q3)"""
    if not title:
        return None

    # 中文季度格式：例如 2026年第三季 / FY2026 第四季 / 2026會計年度第三季
    fy_zh = re.search(r"(?:FY)?(202\d)\s*(?:會計年度|財年|年度|年)?\s*第([一二三四1-4])季", title)
    if fy_zh:
        q_map = {"一": "Q1", "二": "Q2", "三": "Q3", "四": "Q4", "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
        return f"FY{fy_zh.group(1)} {q_map.get(fy_zh.group(2), fy_zh.group(2))}"

    # 英文季度格式：例如 Q3 2026 / 2026 Q3 / Q3 FY2027
    q_en = re.search(r"(?:Q([1-4])\s*(FY)?(202\d)|(FY)?(202\d)\s*Q([1-4]))", title, re.IGNORECASE)
    if q_en:
        if q_en.group(1) and q_en.group(3):
            prefix = "FY" if q_en.group(2) else ""
            return f"{prefix}{q_en.group(3)} Q{q_en.group(1)}"
        if q_en.group(5) and q_en.group(6):
            prefix = "FY" if q_en.group(4) else ""
            return f"{prefix}{q_en.group(5)} Q{q_en.group(6)}"

    # 日期回退季度推算
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        try:
            parts = date.split("-")
            y = int(parts[0])
            m = int(parts[1])
            q = (m - 1) // 3 + 1
            return f"{y} Q{q}"
        except Exception:
            pass

    return None


def parse_biggo_guidance(detail: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """
    從 BigGo /detail 內容深度萃取高管財測展望、營收指引、毛利率目標與資本支出
    """
    if not detail or not isinstance(detail, dict):
        return {
            "summary": None,
            "outlook": None,
            "margin": None,
            "capex": None,
            "qa": None
        }

    outlook_text = str(detail.get("outlook") or "")
    takeaways_text = str(detail.get("takeaways") or detail.get("summary") or "")
    qa_text = str(detail.get("qa_highlights") or "")

    margin_match = re.search(r"(?:毛利率|Gross Margin)[：:]\s*([^\n]+)", outlook_text, re.IGNORECASE)
    capex_match = re.search(r"(?:資本支出|CapEx|Capital Expenditures)[：:]\s*([^\n]+)", outlook_text, re.IGNORECASE)
    rev_match = re.search(r"(?:營收|Revenue)[：:]\s*([^\n]+)", outlook_text, re.IGNORECASE)

    rev_outlook = f"營收展望：{rev_match.group(1).strip()}" if rev_match else (outlook_text.split("\n")[0].strip() if outlook_text else None)
    margin_outlook = f"毛利率目標：{margin_match.group(1).strip()}" if margin_match else None
    capex_outlook = f"資本支出規劃：{capex_match.group(1).strip()}" if capex_match else None

    return {
        "summary": takeaways_text.strip() if takeaways_text else None,
        "outlook": rev_outlook,
        "margin": margin_outlook,
        "capex": capex_outlook,
        "qa": qa_text.strip() if qa_text else None
    }


def _extract_file_url(payload: Any) -> Optional[str]:
    """相容 BigGo /file 回傳可能是 list 或 dict 之多型結構"""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("url") or first.get("file_url")
        elif isinstance(first, str):
            return first
    elif isinstance(data, dict):
        if data.get("list") and isinstance(data["list"], list) and data["list"]:
            first = data["list"][0]
            if isinstance(first, dict):
                return first.get("url") or first.get("file_url")
        return data.get("url") or data.get("file_url")
    return None


def fetch_biggo_call_enrichment(
    call_id: str,
    stock_id: str = "",
    region: str = "tw",
    has_slides: bool = True,
    has_video: bool = True,
    timeout: int = 3
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    並行向 BigGo 調用 /detail, /file?type=slides, /file?type=video
    提取法說會官方簡報 PDF、視訊回放直連與指引摘要
    """
    detail = None
    slides_url = None
    video_url = None

    def _fetch_detail():
        nonlocal detail
        try:
            url = f"{BIGGO_API_BASE}/stock/earnings-calls/detail?call_id={urllib.parse.quote(call_id)}&stock_id={urllib.parse.quote(stock_id)}&region={region}"
            req = urllib.request.Request(url, headers=BIGGO_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
                if d.get("result"):
                    detail = d.get("data")
        except Exception as e:
            logger.debug(f"BigGo detail 失敗 ({call_id}): {e}")

    def _fetch_slides():
        nonlocal slides_url
        if not has_slides:
            return
        try:
            url = f"{BIGGO_API_BASE}/stock/earnings-calls/file?type=slides&call_id={urllib.parse.quote(call_id)}&region={region}"
            req = urllib.request.Request(url, headers=BIGGO_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
                if d.get("result"):
                    slides_url = _extract_file_url(d)
        except Exception as e:
            logger.debug(f"BigGo slides 失敗 ({call_id}): {e}")

    def _fetch_video():
        nonlocal video_url
        if not has_video:
            return
        try:
            url = f"{BIGGO_API_BASE}/stock/earnings-calls/file?type=video&call_id={urllib.parse.quote(call_id)}&region={region}"
            req = urllib.request.Request(url, headers=BIGGO_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
                if d.get("result"):
                    video_url = _extract_file_url(d)
        except Exception as e:
            logger.debug(f"BigGo video 失敗 ({call_id}): {e}")

    # 使用線程池並行執行 3 項富化任務
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_detail = executor.submit(_fetch_detail)
        f_slides = executor.submit(_fetch_slides)
        f_video = executor.submit(_fetch_video)
        concurrent.futures.wait([f_detail, f_slides, f_video], timeout=timeout + 1)

    return detail, slides_url, video_url


class EarningsCallFetcher:
    """跨市場法人說明會資料獲取器 (支援 TW, US, JP, KR 及全球)"""

    def __init__(self, db_manager: Optional[Any] = None, db: Optional[Any] = None):
        target_db = db if db is not None else db_manager
        if target_db is not None:
            self.db = target_db
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
        market: str = "ALL",
        past_windows: int = 3,
        future_windows: int = 3,
        enrich_details_limit: int = 6,
        enrich_slides_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        透過 BigGo 財經官方 API 多視窗並行聚合法人說明會清單。
        當 market="ALL" 時，不帶 market 參數，撈取台灣、美股、日股、韓股等全球所有法說會；
        當 market="TW"|"US"|"JP"|"KR" 時，精確過濾單一市場。
        """
        effective_enrich_limit = enrich_slides_limit if enrich_slides_limit is not None else enrich_details_limit
        m_str = (market or "ALL").strip().upper()
        cache_key = m_str.lower()
        cache_file = os.path.join(CACHE_DIR, f"biggo_{cache_key}_calls.json")
        windows = generate_date_windows(past_windows=past_windows, future_windows=future_windows)
        
        # 決定市場 query 參數
        market_param = ""
        if m_str in ("TW", "US", "JP", "KR", "OTHER"):
            market_param = f"&market={m_str.lower()}"

        raw_items_map: Dict[str, Dict[str, Any]] = {}
        # 1. 分頁榜單任務 (對齊 stockbot: 抓取最新跨市場榜單與日韓分頁)
        fetch_urls: List[str] = []
        if m_str == "ALL":
            fetch_urls.extend([
                f"{BIGGO_API_BASE}/stock/earnings-calls/list?page=1&size=100",
                f"{BIGGO_API_BASE}/stock/earnings-calls/list?page=1&size=60&market=tw",
                f"{BIGGO_API_BASE}/stock/earnings-calls/list?page=1&size=60&market=us",
                f"{BIGGO_API_BASE}/stock/earnings-calls/list?page=1&size=60&market=jp",
                f"{BIGGO_API_BASE}/stock/earnings-calls/list?page=1&size=40&market=kr"
            ])
        else:
            fetch_urls.append(f"{BIGGO_API_BASE}/stock/earnings-calls/list?page=1&size=100{market_param}")

        # 2. 6 滾動日期視窗任務
        for w_start, w_end in windows:
            fetch_urls.append(f"{BIGGO_API_BASE}/stock/earnings-calls/list?start_date={w_start}&end_date={w_end}{market_param}")

        def _fetch_url(u: str) -> List[Dict[str, Any]]:
            try:
                req = urllib.request.Request(u, headers=BIGGO_HEADERS)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("result"):
                        return payload.get("data", {}).get("list", [])
            except Exception as e:
                logger.debug(f"BigGo 抓取失敗 ({u}): {e}")
            return []

        # 並行抓取所有分頁與視窗，秒級完成
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(fetch_urls), 10)) as executor:
            future_to_url = {executor.submit(_fetch_url, u): u for u in fetch_urls}
            for fut in concurrent.futures.as_completed(future_to_url):
                items = fut.result()
                if items:
                    network_success = True
                    for it in items:
                        cid = it.get("call_id")
                        if cid and cid not in raw_items_map:
                            raw_items_map[cid] = it

        # 若網路成功且有資料，寫入本地快取備援
        if network_success and raw_items_map:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(list(raw_items_map.values()), f, ensure_ascii=False, indent=2)
                logger.info(f"✅ BigGo {m_str} 抓取成功: {len(raw_items_map)} 筆法說會紀錄")
            except Exception as e:
                logger.warning(f"寫入 BigGo {m_str} 快取失敗: {e}")
        elif not raw_items_map:
            # 網路全數逾時或斷線，平滑退回本地快取
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_list = json.load(f)
                    for it in cached_list:
                        cid = it.get("call_id")
                        if cid:
                            raw_items_map[cid] = it
                    logger.info(f"📁 已自本地快取載入 {len(raw_items_map)} 筆 BigGo {m_str} 法說會")
                except Exception as cf_err:
                    logger.error(f"❌ 讀取 BigGo 快取失敗: {cf_err}")

        if not raw_items_map:
            return []

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        raw_list = list(raw_items_map.values())

        # 依日期排序，找出即將召開或今天的場次進行優先富化 (簡報 PDF、視訊、指引)
        raw_list.sort(key=lambda x: (x.get("date", ""), x.get("stock_id", "")))
        urgent_candidates = [
            it for it in raw_list
            if it.get("date", "") >= today_str
        ]
        enrich_target_ids = set()
        if effective_enrich_limit > 0:
            enrich_target_ids = {it.get("call_id") for it in (urgent_candidates[:effective_enrich_limit] if urgent_candidates else raw_list[:effective_enrich_limit])}

        # 並行富化重要場次
        enrichment_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]] = {}
        if enrich_target_ids:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(enrich_target_ids), 4)) as executor:
                def _do_enrich(it):
                    cid = it.get("call_id", "")
                    stock_id = it.get("stock_id", "")
                    region = it.get("region", "tw")
                    return cid, fetch_biggo_call_enrichment(
                        cid, stock_id=stock_id, region=region,
                        has_slides=bool(it.get("has_slides")),
                        has_video=bool(it.get("has_video"))
                    )

                futs = [executor.submit(_do_enrich, it) for it in raw_list if it.get("call_id") in enrich_target_ids]
                for fut in concurrent.futures.as_completed(futs):
                    cid, res = fut.result()
                    enrichment_cache[cid] = res

        parsed_records: List[Dict[str, Any]] = []
        for it in raw_list:
            call_id = it.get("call_id", "")
            stock_id = (it.get("stock_id") or "").strip().upper()
            if not stock_id:
                continue

            meta = parse_biggo_stock_metadata(it)
            mkt = meta["market"]
            ticker = meta["symbol"]
            tz = meta["timezone"]
            def_time = meta["default_time"]

            # 若使用者指定單一市場，且解析後不相符則跳過 (雙重保護)
            if m_str in ("TW", "US", "JP", "KR", "OTHER") and mkt != m_str:
                continue

            name = (it.get("name") or "").strip()
            title = (it.get("title") or "").strip()
            call_date = it.get("date", "")
            period = extract_financial_period(title, call_date)
            weight = is_weight_stock(ticker, mkt)

            ir_url = US_IR_WEBCAST_MAP.get(ticker) if mkt == "US" else f"https://finance.biggo.com.tw/earnings-call?q={urllib.parse.quote(stock_id)}"
            meeting_type = "online" if (it.get("has_video") or it.get("has_audio")) else "physical"
            loc = "線上視訊/電話法說會" if meeting_type == "online" else "法人說明會 (實體/線上同步)"

            # 取用富化資訊
            detail_data = None
            slides_url = None
            video_url = None
            if call_id in enrichment_cache:
                detail_data, slides_url, video_url = enrichment_cache[call_id]

            guidance = parse_biggo_guidance(detail_data)

            pres_url = slides_url or (ir_url if it.get("has_slides") else "")
            webcast_url = video_url or (ir_url if (it.get("has_video") or it.get("has_audio")) else "")

            parsed_records.append({
                "id": call_id,
                "market": mkt,
                "ticker": ticker,
                "name": name or title,
                "call_date": call_date,
                "call_time": def_time,
                "timezone": tz,
                "meeting_type": meeting_type,
                "location": loc,
                "webcast_url": webcast_url,
                "presentation_url": pres_url,
                "transcript_url": ir_url if it.get("has_transcript") else "",
                "language": "zh-TW" if mkt == "TW" else ("ja" if mkt == "JP" else ("ko" if mkt == "KR" else "en")),
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": weight,
                "remarks": title,
                "financial_period": period,
                "guidance_summary": guidance.get("summary"),
                "guidance_outlook": guidance.get("outlook"),
                "guidance_margin": guidance.get("margin"),
                "guidance_capex": guidance.get("capex"),
                "guidance_qa": guidance.get("qa"),
                "source": "BigGo / MOPS" if mkt == "TW" else f"BigGo / {mkt} IR"
            })

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
            weight = is_weight_stock(ticker, "TW")

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
                "is_weight_stock": weight,
                "remarks": item.get("法人說明會擇要", "").strip(),
                "financial_period": extract_financial_period(item.get("法人說明會擇要", ""), call_date),
                "guidance_summary": None,
                "guidance_outlook": None,
                "guidance_margin": None,
                "guidance_capex": None,
                "guidance_qa": None,
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
                "is_weight_stock": is_weight_stock(ticker, "TW"),
                "remarks": item.get("法人說明會擇要", "").strip(),
                "financial_period": extract_financial_period(item.get("法人說明會擇要", ""), call_date),
                "guidance_summary": None,
                "guidance_outlook": None,
                "guidance_margin": None,
                "guidance_capex": None,
                "guidance_qa": None,
                "source": "TPEX"
            })
        return parsed_records

    def fetch_tw_calls(self) -> List[Dict[str, Any]]:
        """獲取台灣上市與上櫃公司法人說明會"""
        records = self.fetch_biggo_calls(market="TW")
        if records:
            return records
        return self.fetch_twse_calls() + self.fetch_tpex_calls()

    def fetch_us_calls(self) -> List[Dict[str, Any]]:
        """獲取美股主要公司法說會與財報日程"""
        return self.fetch_biggo_calls(market="US")

    def fetch_jp_calls(self) -> List[Dict[str, Any]]:
        """獲取日本 (TSE) 主要公司法說會日程"""
        return self.fetch_biggo_calls(market="JP")

    def fetch_kr_calls(self) -> List[Dict[str, Any]]:
        """獲取南韓 (KRX) 主要公司法說會日程"""
        return self.fetch_biggo_calls(market="KR")

    def fetch_ticker_calls(self, ticker: str, enrich_top: int = 3) -> List[Dict[str, Any]]:
        """
        依單一股票代碼 (如 2330.TW, NVDA, 7203.T) 線上查詢該標的之歷史與預約法說會
        並自動富化 top 場次之高管指引、簡報 PDF 與影音回放，最後寫入 DuckDB 快取。
        """
        clean = ticker.strip().upper()
        clean_query = clean.split(".")[0]
        results_map: Dict[str, Dict[str, Any]] = {}

        tasks = [
            f"{BIGGO_API_BASE}/stock/earnings-calls/list?q={urllib.parse.quote(clean_query)}&page=1&size=50"
        ]
        # 若為台股 4 碼純代號，額外查詢專用個股端點
        if re.match(r"^\d{4}$", clean_query):
            tasks.append(f"{BIGGO_API_BASE}/stock/{clean_query}.TW/earnings-calls/list?region=tw")

        for u in tasks:
            try:
                req = urllib.request.Request(u, headers=BIGGO_HEADERS)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("result"):
                        for it in payload.get("data", {}).get("list", []):
                            cid = it.get("call_id")
                            if cid and cid not in results_map:
                                results_map[cid] = it
            except Exception as e:
                logger.debug(f"查詢標的法說會失敗 ({u}): {e}")

        raw_items = list(results_map.values()) if results_map else []
        parsed_records: List[Dict[str, Any]] = []

        # 富化前 N 場次
        for idx, it in enumerate(raw_items):
            cid = it.get("call_id", "")
            meta = parse_biggo_stock_metadata(it)
            stock_id = it.get("stock_id", "")
            region = it.get("region", "tw")
            mkt = meta["market"]
            parsed_t = meta["symbol"]
            tz = meta["timezone"]
            title = it.get("title", "")
            call_date = it.get("date", "")

            detail_data = None
            slides_url = None
            video_url = None
            if idx < enrich_top:
                detail_data, slides_url, video_url = fetch_biggo_call_enrichment(
                    cid, stock_id=stock_id, region=region,
                    has_slides=bool(it.get("has_slides")),
                    has_video=bool(it.get("has_video"))
                )

            guidance = parse_biggo_guidance(detail_data)
            ir_url = f"https://finance.biggo.com.tw/earnings-call?q={urllib.parse.quote(stock_id or parsed_t)}"

            parsed_records.append({
                "id": cid,
                "market": mkt,
                "ticker": parsed_t,
                "name": it.get("name") or title,
                "call_date": call_date,
                "call_time": meta["default_time"],
                "timezone": tz,
                "meeting_type": "online" if (it.get("has_video") or it.get("has_audio")) else "physical",
                "location": "線上視訊/電話法說會",
                "webcast_url": video_url or ir_url,
                "presentation_url": slides_url or (ir_url if it.get("has_slides") else ""),
                "transcript_url": ir_url if it.get("has_transcript") else "",
                "language": "zh-TW" if mkt == "TW" else ("ja" if mkt == "JP" else ("ko" if mkt == "KR" else "en")),
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": is_weight_stock(parsed_t, mkt),
                "remarks": title,
                "financial_period": extract_financial_period(title, call_date),
                "guidance_summary": guidance.get("summary"),
                "guidance_outlook": guidance.get("outlook"),
                "guidance_margin": guidance.get("margin"),
                "guidance_capex": guidance.get("capex"),
                "guidance_qa": guidance.get("qa"),
                "source": f"BigGo / {mkt} IR"
            })

        # 融合官方基準庫 (如台積電 2330.TW 等)，即使上游 BigGo 無資料亦能精準回傳
        verified_items = load_verified_corporate_earnings()
        verified_map = {f"{r.get('ticker')}_{r.get('call_date')}": r for r in parsed_records}
        for v in verified_items:
            v_t = (v.get("ticker") or "").upper()
            if v_t == clean or v_t.startswith(clean_query) or (v.get("name") and clean_query in v["name"].upper()):
                k = f"{v.get('ticker')}_{v.get('call_date')}"
                if k not in verified_map:
                    parsed_records.append(v)
                    verified_map[k] = v
                else:
                    existing = verified_map[k]
                    for fld in ("webcast_url", "presentation_url", "transcript_url", "guidance_summary", "guidance_outlook", "guidance_margin", "guidance_capex", "guidance_qa"):
                        if not existing.get(fld) and v.get(fld):
                            existing[fld] = v[fld]

        if not parsed_records:
            return []

        parsed_records.sort(key=lambda x: (x["call_date"], x["call_time"]))

        if parsed_records and self.db and hasattr(self.db, "save_earnings_calls_batch"):
            try:
                self.db.save_earnings_calls_batch(parsed_records)
            except Exception as e:
                logger.debug(f"快取個股法說會至 DuckDB 失敗: {e}")

        return parsed_records

    def fetch_all_and_sync(self) -> int:
        """
        執行全維度跨市場法說會抓取 (台灣、美股、日股、韓股) 並全量 Upsert 存入 DuckDB
        """
        # 一次性調用全域多視窗抓取
        all_records = self.fetch_biggo_calls(market="ALL")

        # 若台股完全為空，嘗試 TWSE/TPEx 次級備援
        has_tw = any(r.get("market") == "TW" for r in all_records)
        if not has_tw:
            logger.info("⚠️ BigGo 未包含台股資料，啟動 TWSE/TPEx 次級備援...")
            tw_backup = self.fetch_twse_calls() + self.fetch_tpex_calls()
            all_records.extend(tw_backup)

        # 融合官方指標企業基準庫 (如台積電 2330.TW、聯發科、鴻海等)，確保重大法說會永不遺漏
        verified_items = load_verified_corporate_earnings()
        merged_map: Dict[str, Dict[str, Any]] = {}
        for r in all_records:
            t = (r.get("ticker") or "").strip().upper()
            d = (r.get("call_date") or "").strip()
            merged_map[f"{t}_{d}"] = r

        for v in verified_items:
            t = (v.get("ticker") or "").strip().upper()
            d = (v.get("call_date") or "").strip()
            key = f"{t}_{d}"
            if key not in merged_map:
                merged_map[key] = v
            else:
                existing = merged_map[key]
                for fld in ("webcast_url", "presentation_url", "transcript_url", "guidance_summary", "guidance_outlook", "guidance_margin", "guidance_capex", "guidance_qa"):
                    if not existing.get(fld) and v.get(fld):
                        existing[fld] = v[fld]

        all_records = list(merged_map.values())

        if not all_records:
            logger.warning("⚠️ 未抓取到任何跨市場法說會紀錄")
            return 0

        if not self.db or not hasattr(self.db, "save_earnings_calls_batch"):
            logger.warning("⚠️ 無資料庫管理器可儲存法說會資料")
            return len(all_records)

        saved_count = self.db.save_earnings_calls_batch(all_records)
        
        # 統計各市場筆數
        counts = {}
        for r in all_records:
            m = r.get("market", "UNKNOWN")
            counts[m] = counts.get(m, 0) + 1

        logger.info(f"🎉 跨市場法說會資料同步完成: 共寫入 {saved_count} 筆 (分佈: {counts})")
        return saved_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = EarningsCallFetcher()
    count = fetcher.fetch_all_and_sync()
    print(f"Sync complete: {count} earnings calls written.")
