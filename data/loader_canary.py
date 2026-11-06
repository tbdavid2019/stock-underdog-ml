"""
data/loader_canary.py - 隔離無快取上游金絲雀健康探測道 (Upstream Canary Health Lane)

提供與生產資料庫完全隔離、具 30 分鐘冷卻保護且強制繞過本地快取之多資料源金絲雀健康探測：
1. 8 大核心資料源探測 (TWSE, TPEx, yfinance, answerbook, Investing, Polymarket, BigGo, arXiv)
2. ThreadPoolExecutor 平行執行與單源 8 秒超時
3. 嚴謹三態健康碼 (healthy, upstream_unreachable_or_ratelimited, schema_drift_parse_error)
4. 白名單欄位原子寫入 (data/cache/canary_health_latest.json)
5. 敏感資訊 100% 遮罩 (Token / Cookie / 內網 IP / 帶參數 URL)
6. 零生產 DuckDB 連線保證
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger("stock_app.canary")

CANARY_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "canary_health_latest.json")
DEFAULT_COOLDOWN_SECONDS = 1800  # 30 分鐘
DEFAULT_PROBE_TIMEOUT = 8.0     # 8 秒逾時

IP_REGEX = re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|127\.0\.0\.1)\b")
TOKEN_REGEX = re.compile(r"(?i)(?:key|token|secret|password|auth|bearer)[\s=:'\"]+([a-zA-Z0-9_\-\.]{6,})")
URL_QUERY_REGEX = re.compile(r"(\bhttps?://[^\s?#]+)\?[^\s#]*")


def sanitize_message(msg: Optional[str]) -> Optional[str]:
    """100% 遮罩錯誤訊息中的金鑰、密碼、內網 IP 與 URL Query 參數"""
    if not msg:
        return None
    s = str(msg)
    # 遮罩 URL query 參數
    s = URL_QUERY_REGEX.sub(r"\1?[REDACTED]", s)
    # 遮罩金鑰 / Token
    s = TOKEN_REGEX.sub(r"[REDACTED_SECRET]", s)
    # 遮罩內網 IP
    s = IP_REGEX.sub("[INTERNAL_IP]", s)
    return s[:300]


# ==================== 8 大資料源專屬金絲雀探針 ====================

def _probe_twse(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """TWSE 臺灣證券交易所金絲雀探針"""
    # 探測 TWSE OpenAPI 或 MIS 最小行情
    url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_2330.tw&json=1&delay=0"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return resp.status_code, [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
        missing = []
        if "msgArray" not in data or not isinstance(data["msgArray"], list) or len(data["msgArray"]) == 0:
            missing.append("msgArray")
        else:
            item = data["msgArray"][0]
            for req_field in ["c", "z", "n"]:  # 代號、成交價、名稱
                if req_field not in item:
                    missing.append(f"msgArray[0].{req_field}")
        return resp.status_code, missing, None
    except Exception as e:
        return resp.status_code, ["valid_json_payload"], str(e)


def _probe_tpex(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """TPEx 櫃買中心金絲雀探針"""
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return resp.status_code, [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
        missing = []
        if not isinstance(data, list) or len(data) == 0:
            missing.append("list_items")
        else:
            item = data[0]
            for req_field in ["SecuritiesCompanyCode", "Close"]:
                if req_field not in item:
                    missing.append(req_field)
        return resp.status_code, missing, None
    except Exception as e:
        return resp.status_code, ["valid_json_payload"], str(e)


def _probe_yfinance(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """yfinance Yahoo 財經行情金絲雀探針"""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return resp.status_code, [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
        missing = []
        chart = data.get("chart", {})
        result = chart.get("result")
        if not result or not isinstance(result, list) or len(result) == 0:
            missing.append("chart.result")
        else:
            meta = result[0].get("meta", {})
            if "regularMarketPrice" not in meta and "chartPreviousClose" not in meta:
                missing.append("chart.result[0].meta.regularMarketPrice")
        return resp.status_code, missing, None
    except Exception as e:
        return resp.status_code, ["valid_json_payload"], str(e)


def _probe_answerbook(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """answerbook 榜單上游金絲雀探針"""
    from core.config import config
    url = getattr(config.api, "TW0050_URL", "https://answerbook.david888.com/TW0050")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return resp.status_code, [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
        missing = []
        if "TW0050" not in data and "data" not in data and "rankings" not in data and not isinstance(data, list):
            missing.append("TW0050")
        return resp.status_code, missing, None
    except Exception as e:
        return resp.status_code, ["valid_json_payload"], str(e)


def _probe_investing(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """Investing.com 總經與財經日曆 (2MD Proxy 通道) 金絲雀探針"""
    target = "https://www.investing.com/central-banks/fed-rate-monitor"
    hosts = ["https://2md.aiurl.tw", "https://create360.ai"]
    key_raw = os.getenv("INVESTING_2MD_KEYS") or os.getenv("CREATE360_API_KEY") or ""
    first_key = [k.strip() for k in key_raw.split(",") if k.strip()]
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    if first_key:
        headers["Authorization"] = f"Bearer {first_key[0]}"

    last_code = 503
    last_err = "2MD proxy unreachable"
    per_host_timeout = max(2.5, timeout / len(hosts))
    for host in hosts:
        try:
            resp = requests.get(f"{host}/{target}", headers=headers, timeout=per_host_timeout)
            last_code = resp.status_code
            if resp.status_code == 200:
                text = resp.text or ""
                if len(text) >= 120 and "Just a moment..." not in text and "403 Forbidden" not in text:
                    return 200, [], None
                last_err = "Blocked or short markdown response"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
    return last_code, [], last_err


def _probe_polymarket(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """Polymarket 預測市場真實資金情緒金絲雀探針"""
    url = "https://gamma-api.polymarket.com/events?limit=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return resp.status_code, [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
        missing = []
        if not isinstance(data, list) or len(data) == 0:
            missing.append("events_list")
        else:
            item = data[0]
            if "id" not in item and "title" not in item:
                missing.append("event_id_or_title")
        return resp.status_code, missing, None
    except Exception as e:
        return resp.status_code, ["valid_json_payload"], str(e)


def _probe_biggo_earnings(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """BigGo 法說會資訊金絲雀探針"""
    url = "https://api.biggo.com/api/v1/finance/stock/earnings-calls/list?page=1&size=1"
    headers = {
        "Accept": "application/json",
        "Site": "finance.biggo.com.tw",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return resp.status_code, [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
        missing = []
        if "data" not in data and "list" not in data and not isinstance(data, list):
            missing.append("earnings_call_data")
        return resp.status_code, missing, None
    except Exception as e:
        return resp.status_code, ["valid_json_payload"], str(e)


def _probe_arxiv(timeout: float = DEFAULT_PROBE_TIMEOUT) -> Tuple[int, List[str], Optional[str]]:
    """arXiv 量化與 AI 論文金絲雀探針"""
    headers = {"User-Agent": "StockQuantCanary/2.6 (+https://stockdata.david888.com)"}
    urls = [
        "https://rss.arxiv.org/rss/cs.AI",
        "https://export.arxiv.org/api/query?search_query=cat:q-fin.ST&max_results=1",
    ]
    per_timeout = max(3.5, timeout / len(urls))
    last_code = 503
    last_err: Optional[str] = None
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=per_timeout)
            last_code = resp.status_code
            if resp.status_code == 200:
                text = resp.text or ""
                if "<rss" in text or "<feed" in text or "<rdf:RDF" in text:
                    return 200, [], None
                return 200, ["xml_feed_tag"], None
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
    return last_code, [], last_err or "arXiv probe failed"


PROBE_REGISTRY: Dict[str, Callable[[float], Tuple[int, List[str], Optional[str]]]] = {
    "twse": _probe_twse,
    "tpex": _probe_tpex,
    "yfinance": _probe_yfinance,
    "answerbook": _probe_answerbook,
    "investing": _probe_investing,
    "polymarket": _probe_polymarket,
    "biggo_earnings": _probe_biggo_earnings,
    "arxiv": _probe_arxiv,
}


def _execute_single_canary_probe(source_name: str, probe_fn: Callable[[float], Tuple[int, List[str], Optional[str]]], timeout: float) -> Dict[str, Any]:
    """執行單一資料源探針並歸類三態健康碼"""
    start_t = time.perf_counter()
    checked_at = datetime.now(timezone.utc).isoformat()
    http_status: Optional[int] = None
    missing_fields: List[str] = []
    sanitized_reason: Optional[str] = None
    status = "healthy"

    try:
        http_code, missing, err_msg = probe_fn(timeout)
        http_status = http_code
        missing_fields = missing
        if err_msg:
            sanitized_reason = sanitize_message(err_msg)

        if http_code in (403, 429, 500, 502, 503, 504) or http_code != 200:
            status = "upstream_unreachable_or_ratelimited"
            if not sanitized_reason:
                sanitized_reason = f"HTTP {http_code}"
        elif missing_fields:
            status = "schema_drift_parse_error"
            if not sanitized_reason:
                sanitized_reason = f"Missing required fields: {', '.join(missing_fields)}"
        else:
            status = "healthy"

    except requests.exceptions.Timeout as e:
        status = "upstream_unreachable_or_ratelimited"
        sanitized_reason = sanitize_message(f"Connection timeout after {timeout}s: {e}")
    except (requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
        status = "upstream_unreachable_or_ratelimited"
        sanitized_reason = sanitize_message(f"Connection or SSL error: {e}")
    except Exception as e:
        status = "schema_drift_parse_error"
        sanitized_reason = sanitize_message(f"Unhandled exception during probe: {e}")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 1)

    return {
        "source": source_name,
        "status": status,
        "latency_ms": elapsed_ms,
        "checked_at": checked_at,
        "http_status": http_status,
        "missing_fields": missing_fields,
        "sanitized_reason": sanitized_reason,
    }


def run_canary_checks(
    sources: Optional[Sequence[str]] = None,
    force: bool = False,
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    timeout_sec: float = DEFAULT_PROBE_TIMEOUT,
    cache_path: str = CANARY_CACHE_PATH,
) -> Dict[str, Any]:
    """執行隔離金絲雀探測道 (支援冷卻重用、平行逾時、白名單原子寫入與 0 DuckDB 連線)"""

    # 1. 冷卻保護檢查：若未帶 --force 且現有快照未滿冷卻時間，直接回傳快照
    if not force and os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            age_sec = time.time() - mtime
            if age_sec < cooldown_seconds:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_data["is_cached"] = True
                cached_data["age_seconds"] = round(age_sec, 1)
                return cached_data
        except Exception as e:
            logger.debug(f"讀取金絲雀快照失敗，重新探測: {e}")

    target_sources = list(sources) if sources else list(PROBE_REGISTRY.keys())
    probes_to_run = {src: PROBE_REGISTRY[src] for src in target_sources if src in PROBE_REGISTRY}

    results: Dict[str, Any] = {}
    # 2. ThreadPoolExecutor(max_workers=4) 平行探測
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(_execute_single_canary_probe, src, fn, timeout_sec): src
            for src, fn in probes_to_run.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            src = future_map[future]
            try:
                res = future.result()
                results[src] = res
            except Exception as e:
                results[src] = {
                    "source": src,
                    "status": "upstream_unreachable_or_ratelimited",
                    "latency_ms": 0.0,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "http_status": None,
                    "missing_fields": [],
                    "sanitized_reason": sanitize_message(str(e)),
                }

    # 3. 彙總全體狀態
    all_healthy = all(r.get("status") == "healthy" for r in results.values())
    overall_status = "healthy" if all_healthy else ("degraded" if any(r.get("status") == "healthy" for r in results.values()) else "unhealthy")

    snapshot = {
        "status": overall_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sources_checked_count": len(results),
        "sources": results,
        "is_cached": False,
        "age_seconds": 0.0,
    }

    # 4. 白名單結構以 tempfile + os.replace 原子寫入快照檔
    try:
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=cache_dir, delete=False, encoding="utf-8") as tf:
            json.dump(snapshot, tf, indent=2, ensure_ascii=False, allow_nan=False)
            temp_name = tf.name
        os.replace(temp_name, cache_path)
    except Exception as e:
        logger.warning(f"寫入金絲雀健康快照失敗: {e}")

    return snapshot
