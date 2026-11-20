"""
data/mops_fetcher.py - 台股公開資訊觀測站 (MOPS) 重大訊息與月營收動能擷取器

整合官方開放資料與 MOPS JSON 介面：
1. TWSE OpenAPI 每日重大訊息 (t187ap04_L) + TPEx OpenAPI 重大訊息 (mopsfin_t187ap04_O)
   與 MOPS 即時重訊查詢 (mops.twse.com.tw/mops/api/t05st02)
2. TWSE OpenAPI 上市月營收 (t187ap05_L) + TPEx OpenAPI 上櫃月營收 (mopsfin_t187ap05_O)

設計原則：
- 嚴格 Point-in-Time 防前瞻：重訊附帶精確到秒之 `available_at` (Asia/Taipei)，月營收記錄 `revenue_ym` 與 `published_date`。
- 多層快取與非致命降級：L1 記憶體 -> L2 DuckDB -> L3 磁碟 JSON 快取。
- 量化因子化：產出事件催化標籤（庫藏股、擴產/資本支出、自結合併）與稀釋/風險警示（現增、私募、公司債、減資），以及月營收 YoY/MoM 動能標籤。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from data.cache import CacheManager

logger = logging.getLogger("stock_app.mops")

TZ_TAIPEI = ZoneInfo("Asia/Taipei")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOPS_CACHE_DIR = os.path.join(BASE_DIR, "cache", "mops")

POSITIVE_CATALYST_KEYWORDS: Tuple[str, ...] = (
    "買回公司股份",
    "庫藏股",
    "廠務設備",
    "機器設備",
    "資本支出",
    "自結合併",
    "擴建",
    "取得訂單",
    "現金股利",
    "盈餘分配",
)

RISK_WARNING_KEYWORDS: Tuple[str, ...] = (
    "私募普通股",
    "轉換公司債",
    "現金增資",
    "公司債",
    "減資",
    "資通安全",
    "重大損失",
    "更正財報",
)

ALL_KEYWORDS_SORTED: Tuple[str, ...] = tuple(
    sorted(POSITIVE_CATALYST_KEYWORDS + RISK_WARNING_KEYWORDS, key=len, reverse=True)
)


def parse_roc_date(raw_date: Any) -> str:
    """Convert ROC or ISO date (e.g. '1151119', '115/1/5', '2026-11-19') into valid ISO YYYY-MM-DD, else ''."""
    if raw_date is None:
        return ""
    s = str(raw_date).strip()
    if not s:
        return ""
    y = m = d = None
    sep_match = re.match(r"^(\d{2,4})[/\-](\d{1,2})[/\-](\d{1,2})$", s)
    if sep_match:
        y_raw, m, d = int(sep_match.group(1)), int(sep_match.group(2)), int(sep_match.group(3))
        y = y_raw + 1911 if y_raw < 1900 else y_raw
    else:
        digits = re.sub(r"[^\d]", "", s)
        if len(digits) == 7:
            y, m, d = int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:7])
        elif len(digits) == 6:
            y, m, d = int(digits[:2]) + 1911, int(digits[2:4]), int(digits[4:6])
        elif len(digits) == 8:
            y, m, d = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
    if y is not None and m is not None and d is not None:
        try:
            return datetime.date(y, m, d).isoformat()
        except ValueError:
            return ""
    return ""


def parse_roc_time(raw_time: Any) -> str:
    """Convert TWSE/TPEx time string (e.g. '64821', '185232', '18:5:32') into HH:MM:SS, else ''."""
    if raw_time is None:
        return ""
    s = str(raw_time).strip()
    if not s:
        return ""
    sep_match = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", s)
    if sep_match:
        hh, mm, ss = int(sep_match.group(1)), int(sep_match.group(2)), int(sep_match.group(3) or 0)
    else:
        digits = re.sub(r"[^\d]", "", s)
        if not digits or len(digits) > 6:
            return ""
        padded = digits.zfill(6)
        hh, mm, ss = int(padded[:2]), int(padded[2:4]), int(padded[4:6])
    if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return ""


def parse_roc_ym(raw_ym: Any) -> str:
    """Convert ROC/Gregorian year-month (e.g. '11510', '115/1', '2026-10') into YYYY-MM, else ''."""
    if raw_ym is None:
        return ""
    s = str(raw_ym).strip()
    if not s:
        return ""
    y = m = None
    sep_match = re.match(r"^(\d{2,4})[/\-](\d{1,2})$", s)
    if sep_match:
        y_raw, m = int(sep_match.group(1)), int(sep_match.group(2))
        y = y_raw + 1911 if y_raw < 1900 else y_raw
    else:
        digits = re.sub(r"[^\d]", "", s)
        if len(digits) == 5:
            y, m = int(digits[:3]) + 1911, int(digits[3:5])
        elif len(digits) == 4:
            y, m = int(digits[:2]) + 1911, int(digits[2:4])
        elif len(digits) == 6:
            y, m = int(digits[:4]), int(digits[4:6])
    if y is not None and m is not None and 1 <= y <= 2999 and 1 <= m <= 12:
        return f"{y:04d}-{m:02d}"
    return ""


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace(",", "").replace("%", "").strip()
    if not s or s in ("-", "--", "N/A", "不適用"):
        return None
    try:
        f = float(s)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except (ValueError, TypeError):
        return None


def classify_material_subject(subject: str, body: str = "") -> Tuple[str, List[str]]:
    """Classify a material information announcement into ('positive'|'risk'|'neutral', matched_keywords)."""
    text = re.sub(r"\s+", "", f"{subject or ''} {(body or '')[:200]}")
    matched: List[str] = []
    for kw in ALL_KEYWORDS_SORTED:
        if kw in text:
            matched.append(kw)

    if any(kw in RISK_WARNING_KEYWORDS for kw in matched):
        return "risk", matched
    if any(kw in POSITIVE_CATALYST_KEYWORDS for kw in matched):
        return "positive", matched
    return "neutral", matched


def _build_available_at(speech_date_iso: str, speech_time_hms: str) -> str:
    """Build Asia/Taipei Point-in-Time timestamp. Invalid date -> ''; missing time -> end-of-day 23:59:59 (prevents morning lookahead)."""
    if not speech_date_iso or not re.match(r"^\d{4}-\d{2}-\d{2}$", speech_date_iso):
        return ""
    effective_time = speech_time_hms if (speech_time_hms and re.match(r"^\d{2}:\d{2}:\d{2}$", speech_time_hms)) else "23:59:59"
    try:
        dt = datetime.datetime.fromisoformat(f"{speech_date_iso}T{effective_time}").replace(tzinfo=TZ_TAIPEI)
        return dt.isoformat()
    except Exception:
        return ""


class MOPSFetcher:
    """台股公開資訊觀測站 (MOPS) 重大訊息與月營收動能擷取與因子服務"""

    TWSE_MATERIAL_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    TPEX_MATERIAL_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"
    TWSE_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
    TPEX_REVENUE_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"
    MOPS_LIST_API = "https://mops.twse.com.tw/mops/api/t05st02"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    def __init__(self, db_manager: Optional[Any] = None):
        self.db = db_manager

    def _get_db(self):
        if self.db is not None:
            return self.db
        try:
            from data.duckdb_manager import DuckDBManager
            self.db = DuckDBManager()
        except Exception:
            self.db = None
        return self.db

    @classmethod
    def _write_disk_cache(cls, filename: str, payload: Any) -> None:
        try:
            os.makedirs(MOPS_CACHE_DIR, exist_ok=True)
            path = os.path.join(MOPS_CACHE_DIR, filename)
            CacheManager._atomic_write_json(path, payload, indent=2)
        except Exception as e:
            logger.debug(f"MOPS disk cache write skipped ({filename}): {e}")

    @classmethod
    def _read_disk_cache(cls, filename: str) -> Optional[Any]:
        path = os.path.join(MOPS_CACHE_DIR, filename)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @classmethod
    def _parse_material_row(cls, row: Dict[str, Any], suffix: str, market_label: str) -> Optional[Dict[str, Any]]:
        code = str(row.get("公司代號") or row.get("SecuritiesCompanyCode") or "").strip()
        if not code or not re.match(r"^[0-9A-Za-z]{4,6}$", code):
            return None
        name = str(row.get("公司名稱") or row.get("CompanyName") or "").strip()
        # Note: TWSE t187ap04_L intentionally has a trailing space in key "主旨 "
        subject = re.sub(r"\s+", " ", str(row.get("主旨 ") or row.get("主旨") or "").strip())
        if not subject:
            return None
        body = str(row.get("說明") or "").strip()
        clause = str(row.get("符合條款") or "").strip()
        speech_date = parse_roc_date(row.get("發言日期") or row.get("Date"))
        speech_time = parse_roc_time(row.get("發言時間"))
        fact_date = parse_roc_date(row.get("事實發生日"))
        ticker = f"{code}.{suffix}"
        sentiment_tag, matched = classify_material_subject(subject, body)
        available_at = _build_available_at(speech_date, speech_time)
        digest = hashlib.sha256(f"{ticker}|{speech_date}|{speech_time}|{subject}".encode("utf-8")).hexdigest()[:20]
        return {
            "event_id": f"mops_{digest}",
            "ticker": ticker,
            "raw_code": code,
            "company_name": name,
            "market": market_label,
            "speech_date": speech_date,
            "speech_time": speech_time,
            "fact_date": fact_date,
            "available_at": available_at,
            "clause": clause,
            "subject": subject,
            "summary": re.sub(r"\s+", " ", body)[:400],
            "matched_keywords": matched,
            "sentiment_tag": sentiment_tag,
            "source": f"{market_label}_OpenAPI",
        }

    @classmethod
    def fetch_intraday_mops_t05st02(cls, target_dt: Optional[datetime.datetime] = None, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """
        使用 MOPS 即時 JSON API (mops.twse.com.tw/mops/api/t05st02) 擷取當日盤中/盤後最新重訊清單。
        """
        now_tw = target_dt or datetime.datetime.now(TZ_TAIPEI)
        payload = {
            "year": str(now_tw.year - 1911),
            "month": str(now_tw.month),
            "day": f"{now_tw.day:02d}",
        }
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Origin": "https://mops.twse.com.tw",
            "Referer": "https://mops.twse.com.tw/mops/web/t05st02",
            "User-Agent": cls.HEADERS["User-Agent"],
        }
        events: List[Dict[str, Any]] = []
        try:
            resp = requests.post(cls.MOPS_LIST_API, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                logger.warning(f"⚠️ MOPS intraday t05st02 HTTP {resp.status_code}")
                return []
            body = resp.json()
            if not isinstance(body, dict):
                logger.warning("⚠️ MOPS intraday t05st02 unexpected non-dict JSON response")
                return []
            result_obj = body.get("result")
            rows = result_obj.get("data") if isinstance(result_obj, dict) else None
            if not isinstance(rows, list):
                return []
            for row in rows:
                if not isinstance(row, list) or len(row) < 6:
                    continue
                speech_date = parse_roc_date(row[0])
                speech_time = parse_roc_time(row[1])
                code = str(row[2] or "").strip()
                name = str(row[3] or "").strip()
                subject = re.sub(r"\s+", " ", str(row[4] or "").strip())
                if not code or not re.match(r"^[0-9A-Za-z]{4,6}$", code) or not subject:
                    continue
                meta = row[5] if isinstance(row[5], dict) else {}
                params = meta.get("parameters") if isinstance(meta.get("parameters"), dict) else {}
                market_kind = str(params.get("marketKind") or "").strip().lower()
                if market_kind == "sii":
                    suffix, market_label = "TW", "TWSE"
                elif market_kind == "otc":
                    suffix, market_label = "TWO", "TPEx"
                else:
                    # Skip rotc / pub / unknown boards so non-TWSE equities do not pollute .TW
                    continue
                ticker = f"{code}.{suffix}"
                sentiment_tag, matched = classify_material_subject(subject)
                available_at = _build_available_at(speech_date, speech_time)
                digest = hashlib.sha256(f"{ticker}|{speech_date}|{speech_time}|{subject}".encode("utf-8")).hexdigest()[:20]
                events.append({
                    "event_id": f"mops_{digest}",
                    "ticker": ticker,
                    "raw_code": code,
                    "company_name": name,
                    "market": market_label,
                    "speech_date": speech_date,
                    "speech_time": speech_time,
                    "fact_date": speech_date,
                    "available_at": available_at,
                    "clause": "",
                    "subject": subject,
                    "summary": subject,
                    "matched_keywords": matched,
                    "sentiment_tag": sentiment_tag,
                    "source": "MOPS_t05st02",
                })
        except Exception as e:
            logger.warning(f"⚠️ MOPS intraday t05st02 fetch failed: {e}")
        return events

    @classmethod
    def fetch_material_events(cls, include_intraday_mops: bool = True, timeout: float = 12.0) -> List[Dict[str, Any]]:
        """Fetch TWSE + TPEx daily material announcements and optional intraday MOPS t05st02 with per-feed disk fallback."""
        by_id: Dict[str, Dict[str, Any]] = {}

        for url, suffix, label in (
            (cls.TWSE_MATERIAL_URL, "TW", "TWSE"),
            (cls.TPEX_MATERIAL_URL, "TWO", "TPEx"),
        ):
            cache_file = f"material_{label}.json"
            feed_events: List[Dict[str, Any]] = []
            feed_ok = False
            try:
                resp = requests.get(url, headers=cls.HEADERS, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        feed_ok = True
                        for row in data:
                            if isinstance(row, dict):
                                parsed = cls._parse_material_row(row, suffix, label)
                                if parsed:
                                    feed_events.append(parsed)
                else:
                    logger.warning(f"⚠️ {label} 重大訊息 HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ 抓取 {label} 重大訊息失敗: {e}")

            if feed_ok:
                if feed_events:
                    cls._write_disk_cache(cache_file, feed_events[:500])
            else:
                cached = cls._read_disk_cache(cache_file)
                if isinstance(cached, list):
                    logger.info(f"📦 {label} 重大訊息改用本機快取備援 ({len(cached)} 筆)")
                    feed_events = [item for item in cached if isinstance(item, dict) and item.get("event_id")]

            for item in feed_events:
                by_id[item["event_id"]] = item

        if include_intraday_mops:
            for item in cls.fetch_intraday_mops_t05st02(timeout=min(timeout, 10.0)):
                by_id.setdefault(item["event_id"], item)

        return sorted(
            by_id.values(),
            key=lambda x: (x.get("speech_date", ""), x.get("speech_time", "")),
            reverse=True,
        )

    @classmethod
    def _parse_revenue_row(cls, row: Dict[str, Any], suffix: str, market_label: str) -> Optional[Dict[str, Any]]:
        code = str(row.get("公司代號") or "").strip()
        if not code or not re.match(r"^[0-9A-Za-z]{4,6}$", code):
            return None
        ym_iso = parse_roc_ym(row.get("資料年月"))
        if not ym_iso:
            return None
        name = str(row.get("公司名稱") or "").strip()
        industry = str(row.get("產業別") or "").strip()
        pub_date = parse_roc_date(row.get("出表日期"))
        cur_rev = _safe_float(row.get("營業收入-當月營收"))
        prev_rev = _safe_float(row.get("營業收入-上月營收"))
        last_yr_rev = _safe_float(row.get("營業收入-去年當月營收"))
        mom_pct = _safe_float(row.get("營業收入-上月比較增減(%)"))
        yoy_pct = _safe_float(row.get("營業收入-去年同月增減(%)"))
        ytd_rev = _safe_float(row.get("累計營業收入-當月累計營收"))
        ytd_yoy_pct = _safe_float(row.get("累計營業收入-前期比較增減(%)"))
        remark = str(row.get("備註") or "").strip()
        if remark == "-":
            remark = ""

        ticker = f"{code}.{suffix}"
        revenue_tag = None
        if yoy_pct is not None:
            if yoy_pct >= 20.0:
                revenue_tag = "營收高成長"
            elif yoy_pct >= 10.0 and (mom_pct is not None and mom_pct > 0):
                revenue_tag = "營收雙增"
            elif yoy_pct <= -20.0:
                revenue_tag = "營收衰退警訊"

        return {
            "ticker": ticker,
            "raw_code": code,
            "company_name": name,
            "market": market_label,
            "industry": industry,
            "revenue_ym": ym_iso,
            "published_date": pub_date,
            "current_revenue": cur_rev,
            "prev_revenue": prev_rev,
            "last_year_revenue": last_yr_rev,
            "mom_pct": round(mom_pct, 2) if mom_pct is not None else None,
            "yoy_pct": round(yoy_pct, 2) if yoy_pct is not None else None,
            "ytd_revenue": ytd_rev,
            "ytd_yoy_pct": round(ytd_yoy_pct, 2) if ytd_yoy_pct is not None else None,
            "revenue_tag": revenue_tag,
            "remark": remark[:200],
            "source": f"{market_label}_OpenAPI",
        }

    @classmethod
    def fetch_monthly_revenue(cls, timeout: float = 12.0) -> List[Dict[str, Any]]:
        """Fetch TWSE + TPEx monthly operating revenue announcements with per-feed disk fallback."""
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for url, suffix, label in (
            (cls.TWSE_REVENUE_URL, "TW", "TWSE"),
            (cls.TPEX_REVENUE_URL, "TWO", "TPEx"),
        ):
            cache_file = f"revenue_{label}.json"
            feed_rows: List[Dict[str, Any]] = []
            feed_ok = False
            try:
                resp = requests.get(url, headers=cls.HEADERS, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        feed_ok = True
                        for row in data:
                            if isinstance(row, dict):
                                parsed = cls._parse_revenue_row(row, suffix, label)
                                if parsed:
                                    feed_rows.append(parsed)
                else:
                    logger.warning(f"⚠️ {label} 月營收 HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ 抓取 {label} 月營收失敗: {e}")

            if feed_ok:
                if feed_rows:
                    cls._write_disk_cache(cache_file, feed_rows[:2500])
            else:
                cached = cls._read_disk_cache(cache_file)
                if isinstance(cached, list):
                    logger.info(f"📦 {label} 月營收改用本機快取備援 ({len(cached)} 筆)")
                    feed_rows = [r for r in cached if isinstance(r, dict) and r.get("ticker") and r.get("revenue_ym")]

            for item in feed_rows:
                by_key[(item["ticker"], item["revenue_ym"])] = item

        return list(by_key.values())

    def sync_to_duckdb(self, include_intraday_mops: bool = True) -> Dict[str, int]:
        """Fetch latest MOPS material announcements & monthly revenue and persist to DuckDB."""
        events = self.fetch_material_events(include_intraday_mops=include_intraday_mops)
        revenues = self.fetch_monthly_revenue()

        db = self._get_db()
        saved_events = 0
        saved_revs = 0
        if db and getattr(db, "enabled", False):
            if events and hasattr(db, "save_mops_events_batch"):
                saved_events = db.save_mops_events_batch(events)
            if revenues and hasattr(db, "save_monthly_revenue_batch"):
                saved_revs = db.save_monthly_revenue_batch(revenues)

        logger.info(f"✅ MOPS 公開資訊同步完成: 重訊 {saved_events} 筆, 月營收 {saved_revs} 筆")
        return {"events": saved_events, "revenues": saved_revs}

    @staticmethod
    def _finite_or_none(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            return f if math.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def build_factor_summary(cls, revenue: Optional[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute deterministic Taiwan quant factor tags & bounded score adjustment from MOPS data."""
        tags: List[str] = []
        score_delta = 0.0
        positive_catalysts: List[str] = []
        risk_warnings: List[str] = []

        yoy = cls._finite_or_none(revenue.get("yoy_pct")) if revenue else None
        mom = cls._finite_or_none(revenue.get("mom_pct")) if revenue else None
        ytd_yoy = cls._finite_or_none(revenue.get("ytd_yoy_pct")) if revenue else None

        if revenue:
            rev_tag = revenue.get("revenue_tag")
            if yoy is not None:
                if yoy >= 20.0:
                    score_delta += 8.0
                    tags.append(f"營收YoY+{yoy:.0f}%")
                elif yoy >= 10.0 and mom is not None and mom > 0:
                    score_delta += 5.0
                    tags.append("營收雙增")
                elif yoy <= -20.0:
                    score_delta -= 6.0
                    tags.append("營收衰退警訊")
            elif rev_tag:
                tags.append(rev_tag)

        for ev in events or []:
            st = ev.get("sentiment_tag")
            kws = ev.get("matched_keywords") or []
            if isinstance(kws, str):
                try:
                    kws = json.loads(kws)
                except Exception:
                    kws = [kws]
            if st == "positive":
                positive_catalysts.extend(kws or ["正向重訊"])
            elif st == "risk":
                risk_warnings.extend(kws or ["風險重訊"])

        if positive_catalysts:
            score_delta += 4.0
            if any(k in positive_catalysts for k in ("買回公司股份", "庫藏股")):
                tags.append("庫藏股催化")
            elif any(k in positive_catalysts for k in ("資本支出", "廠務設備", "機器設備", "擴建")):
                tags.append("擴產CapEx")
            else:
                tags.append("正向重訊")

        if risk_warnings:
            score_delta -= 6.0
            tags.append("重訊稀釋/風險警示")

        return {
            "score_delta": max(-10.0, min(12.0, round(score_delta, 2))),
            "tags": list(dict.fromkeys(tags)),
            "positive_catalysts": list(dict.fromkeys(positive_catalysts)),
            "risk_warnings": list(dict.fromkeys(risk_warnings)),
            "revenue_ym": revenue.get("revenue_ym") if revenue else None,
            "yoy_pct": yoy,
            "mom_pct": mom,
            "ytd_yoy_pct": ytd_yoy,
            "recent_events_count": len(events or []),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = MOPSFetcher().sync_to_duckdb()
    print("MOPS sync complete:", res)
