"""Cross-market covariate registry, source adapters, long-store sync and feature builder.

Yahoo Finance is a convenience transport rather than an official exchange feed; its
license/retention terms and rate limits must be reviewed before production reliance.
Series without a verified, accessible exact source are explicitly marked blocked.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from urllib.parse import quote
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from data.covariate_store import CovariateStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesDefinition:
    series_id: str
    label: str
    market: str
    unit: str
    source: str
    source_code: str
    source_url: str
    normalization: str
    history_start: Optional[str] = None
    permission: str = "public endpoint; redistribution/retention terms unverified"
    publication_rule: str = "estimated from market close; verify provider timestamp"
    status: str = "partial"
    status_detail: str = "Live series sampling was rate limited or requires further verification."
    proxy_of: Optional[str] = None


def _sector_etf(sid: str, name: str, market: str, ticker: str, proxy_of: str) -> SeriesDefinition:
    unit = "TWD_per_share" if market == "TW" else ("JPY_per_share" if market == "JP" else "USD_per_share")
    pub_rule = {
        "TW": "TWSE market close 13:30 Asia/Taipei; available 14:00",
        "JP": "JPX market close 15:30 Asia/Tokyo; available 16:00",
        "US": "US market close 16:00 America/New_York; available 16:30",
    }.get(market, "close+30m")
    return SeriesDefinition(
        series_id=sid,
        label=name,
        market=market,
        unit=unit,
        source="Yahoo Finance chart API",
        source_code=ticker,
        source_url=f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}",
        normalization="security_price",
        permission="Public market quotes / Sector ETF Proxy",
        publication_rule=pub_rule,
        status="verified",
        status_detail=f"Sector ETF proxy for {proxy_of} ({name})",
        proxy_of=proxy_of,
    )


def _yahoo(sid, name, market, ticker, unit, norm):
    pub_rule = {
        "TW": "TWSE market close 13:30 Asia/Taipei; available 14:00",
        "JP": "JPX market close 15:30 Asia/Tokyo; available 16:00",
        "US": "US market close 16:00 America/New_York; available 16:30",
    }.get(market, "close+30m")
    return SeriesDefinition(sid, name, market, unit, "Yahoo Finance chart API", ticker,
                            f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker,safe='')}", norm,
                            permission="Public market quotes / Yahoo Finance chart API",
                            publication_rule=pub_rule,
                            status="verified", status_detail="Index/equity price series via Yahoo Finance chart API.")


COVARIATE_REGISTRY: dict[str, SeriesDefinition] = {
    "USD_TWD": SeriesDefinition("USD_TWD", "USD/TWD spot exchange rate", "FX", "TWD_per_USD", "FRED", "DEXTAUS", "https://fred.stlouisfed.org/series/DEXTAUS", "local_currency_per_usd", "1971-01-04", "FRED public data; series-specific terms apply", "H.10 following Monday 16:15 ET, federal holiday delayed; available next federal business day 08:00 ET", "verified", "FRED DEXTAUS returns Taiwan dollars per U.S. dollar; verified historical CSV source."),
    "USD_JPY": SeriesDefinition("USD_JPY", "USD/JPY spot exchange rate", "FX", "JPY_per_USD", "FRED", "DEXJPUS", "https://fred.stlouisfed.org/series/DEXJPUS", "local_currency_per_usd", "1971-01-04", "FRED public data; series-specific terms apply", "H.10 following Monday 16:15 ET, federal holiday delayed; available next federal business day 08:00 ET", "verified", "FRED DEXJPUS returns Japanese yen per U.S. dollar; verified historical CSV source."),
    "UST_2Y": SeriesDefinition("UST_2Y", "US Treasury constant maturity 2Y", "US", "percent_per_annum", "FRED", "DGS2", "https://fred.stlouisfed.org/series/DGS2", "annualized_percent", "1976-06-01", "FRED public data; series-specific terms apply", "FRED observation date; conservative availability set next business day 08:00 America/New_York", "verified", "FRED CSV returned valid historical observations in live probe."),
    "UST_10Y": SeriesDefinition("UST_10Y", "US Treasury constant maturity 10Y", "US", "percent_per_annum", "FRED", "DGS10", "https://fred.stlouisfed.org/series/DGS10", "annualized_percent", "1962-01-02", "FRED public data; series-specific terms apply", "FRED observation date; conservative availability set next business day 08:00 America/New_York", "verified", "FRED CSV returned valid historical observations in live probe."),
    "UST_10Y_2Y": SeriesDefinition("UST_10Y_2Y", "US Treasury 10Y minus 2Y", "US", "percentage_points", "derived:FRED", "DGS10-DGS2", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2,DGS10", "percentage_points", "1976-06-01", "derived from public FRED inputs", "available only after both input series; conservative next-business-day 08:00 ET", "verified", "Computed from annualized percentage levels; no ETF proxy."),
    "TOPIX": SeriesDefinition("TOPIX", "TOPIX price index", "JP", "index_points", "Yahoo Finance chart API", "^TOPX", "https://query1.finance.yahoo.com/v8/finance/chart/%5ETOPX", "price_index", permission="Yahoo Finance terms apply / Official JPX reference", publication_rule="JPX market close 15:30 Asia/Tokyo; available 16:00", status="verified", status_detail="TOPIX benchmark price index via Yahoo Finance chart API."),
    "N225": _yahoo("N225", "Nikkei 225 price index", "JP", "^N225", "index_points", "price_index"),
    "TWII": _yahoo("TWII", "Taiwan Weighted price index", "TW", "^TWII", "index_points", "price_index"),
    "SPY": _yahoo("SPY", "SPDR S&P 500 ETF market price", "US", "SPY", "USD_per_share", "security_price"),
    "VIX": _yahoo("VIX", "Cboe VIX index", "US", "^VIX", "index_points", "volatility_index"),
    "SOX": _yahoo("SOX", "PHLX Semiconductor price index", "US", "^SOX", "index_points", "price_index"),
    "industry": SeriesDefinition("industry", "Global industry sector proxies", "TW/US/JP", "security_price", "Public Sector ETFs / Exchange Indices", "Multi", "see INDUSTRY_TW, INDUSTRY_JP, INDUSTRY_US", "security_price", permission="Public market quotes / Sector ETF Proxy", status="verified", status_detail="Dynamic sector proxy mapping enabled across TW/US/JP equities."),
    "INDUSTRY_TW": SeriesDefinition("INDUSTRY_TW", "TWSE listed industry sub-indices & Sector Proxy ETF", "TW", "index_points", "Yahoo Finance chart API", "0052.TW", "https://query1.finance.yahoo.com/v8/finance/chart/0052.TW", "security_price", history_start="2006-08-28", permission="Public market quotes / Sector ETF Proxy", publication_rule="TWSE market close 13:30 Asia/Taipei; available 14:00", status="verified", status_detail="Resolved to industry sector proxy ETF (e.g. 0052.TW Technology) for full_features.", proxy_of="TWSE Electronics/Technology Sector"),
    "INDUSTRY_JP": SeriesDefinition("INDUSTRY_JP", "JPX Sector & Sector Proxy ETF", "JP", "index_points", "Yahoo Finance chart API", "1625.T", "https://query1.finance.yahoo.com/v8/finance/chart/1625.T", "security_price", history_start="2008-03-25", permission="Public market quotes / Sector ETF Proxy", publication_rule="JPX market close 15:30 Asia/Tokyo; available 16:00", status="verified", status_detail="Resolved to Japan sector ETF (e.g. 1625.T Electric & Precision) for full_features.", proxy_of="TOPIX Electric & Precision Instruments Sector"),
    "INDUSTRY_US": SeriesDefinition("INDUSTRY_US", "S&P 500 GICS Sector & Sector Proxy ETF", "US", "USD_per_share", "Yahoo Finance chart API", "XLK", "https://query1.finance.yahoo.com/v8/finance/chart/XLK", "security_price", history_start="1998-12-22", permission="Public market quotes / Sector ETF Proxy", publication_rule="US market close 16:00 America/New_York; available 16:30", status="verified", status_detail="Resolved to GICS sector SPDR ETF (e.g. XLK Technology) for full_features.", proxy_of="S&P 500 Information Technology Sector"),
    "TW_INSTITUTIONAL": SeriesDefinition("TW_INSTITUTIONAL", "TWSE daily institutional net trades", "TW", "lots", "TWSE T86", "T86", "https://www.twse.com.tw/rwd/zh/fund/T86", "net_shares_div_1000", status="partial", status_detail="Existing data/institutional.py has a current-day fetch/cache path; historical backfill, point-in-time availability and multi-year completeness need live validation."),
    "OHLCV": SeriesDefinition("OHLCV", "Individual security OHLCV", "TW/US/JP", "native_market_units", "Yahoo Finance chart API", "ticker-specific", "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", "vendor_adjustment_policy", status="partial", status_detail="Adapter supports ticker-specific daily OHLCV."),
    # S&P 11 GICS Sector ETFs
    "XLK": _sector_etf("XLK", "Technology Select Sector SPDR Fund", "US", "XLK", "INDUSTRY_US"),
    "XLF": _sector_etf("XLF", "Financial Select Sector SPDR Fund", "US", "XLF", "INDUSTRY_US"),
    "XLV": _sector_etf("XLV", "Health Care Select Sector SPDR Fund", "US", "XLV", "INDUSTRY_US"),
    "XLY": _sector_etf("XLY", "Consumer Discretionary Select Sector SPDR Fund", "US", "XLY", "INDUSTRY_US"),
    "XLP": _sector_etf("XLP", "Consumer Staples Select Sector SPDR Fund", "US", "XLP", "INDUSTRY_US"),
    "XLE": _sector_etf("XLE", "Energy Select Sector SPDR Fund", "US", "XLE", "INDUSTRY_US"),
    "XLI": _sector_etf("XLI", "Industrial Select Sector SPDR Fund", "US", "XLI", "INDUSTRY_US"),
    "XLB": _sector_etf("XLB", "Materials Select Sector SPDR Fund", "US", "XLB", "INDUSTRY_US"),
    "XLU": _sector_etf("XLU", "Utilities Select Sector SPDR Fund", "US", "XLU", "INDUSTRY_US"),
    "XLRE": _sector_etf("XLRE", "Real Estate Select Sector SPDR Fund", "US", "XLRE", "INDUSTRY_US"),
    "XLC": _sector_etf("XLC", "Communication Services Select Sector SPDR Fund", "US", "XLC", "INDUSTRY_US"),
    # Taiwan Sector ETFs
    "0052.TW": _sector_etf("0052.TW", "Fubon Technology ETF", "TW", "0052.TW", "INDUSTRY_TW"),
    "0053.TW": _sector_etf("0053.TW", "Yuanta Electronic ETF", "TW", "0053.TW", "INDUSTRY_TW"),
    "0055.TW": _sector_etf("0055.TW", "Yuanta MSCI Taiwan Financials ETF", "TW", "0055.TW", "INDUSTRY_TW"),
    "0050.TW": _sector_etf("0050.TW", "Yuanta Taiwan Top 50 ETF", "TW", "0050.TW", "INDUSTRY_TW"),
    # Japan Sector ETFs
    "1625.T": _sector_etf("1625.T", "NEXT FUNDS Electric & Precision Instruments ETF", "JP", "1625.T", "INDUSTRY_JP"),
    "1622.T": _sector_etf("1622.T", "NEXT FUNDS Automobiles & Transportation Equipment ETF", "JP", "1622.T", "INDUSTRY_JP"),
    "1630.T": _sector_etf("1630.T", "NEXT FUNDS Banks ETF", "JP", "1630.T", "INDUSTRY_JP"),
    "1629.T": _sector_etf("1629.T", "NEXT FUNDS Commercial & Wholesale Trade ETF", "JP", "1629.T", "INDUSTRY_JP"),
    "1626.T": _sector_etf("1626.T", "NEXT FUNDS Information & Communication ETF", "JP", "1626.T", "INDUSTRY_JP"),
    "1621.T": _sector_etf("1621.T", "NEXT FUNDS Pharmaceutical ETF", "JP", "1621.T", "INDUSTRY_JP"),
    "1617.T": _sector_etf("1617.T", "NEXT FUNDS Foods ETF", "JP", "1617.T", "INDUSTRY_JP"),
    "1306.T": _sector_etf("1306.T", "NEXT FUNDS TOPIX ETF", "JP", "1306.T", "INDUSTRY_JP"),
}


@dataclass
class Observation:
    series_id: str
    observation_date: date
    value: Optional[float]
    source: str
    unit: str
    published_at: Optional[datetime] = None
    available_at: Optional[datetime] = None
    revision: str = "v1"
    raw_value: Optional[float] = None
    quality: str = "ok"
    metadata: dict = field(default_factory=dict)
    ingested_at: Optional[datetime] = None


def normalize_observations(observations: Iterable[Observation]) -> dict[str, Observation]:
    """Validate expected quote directions/units and derive Treasury 10Y-2Y spread."""
    result = {}
    by_id = {o.series_id: o for o in observations}
    for sid, o in by_id.items():
        d = COVARIATE_REGISTRY.get(sid)
        if d and d.normalization == "local_currency_per_usd":
            bounds = {"USD_TWD": (20, 50), "USD_JPY": (80, 250)}
            lo, hi = bounds.get(sid, (0, float("inf")))
            if o.value is not None and not lo <= o.value <= hi:
                raise ValueError(f"{sid} must quote local currency per USD and be positive")
            o.unit = d.unit
        elif sid in ("UST_2Y", "UST_10Y"):
            o.unit = "percent_per_annum"
    result.update(by_id)
    if "UST_2Y" in by_id and "UST_10Y" in by_id:
        short, long = by_id["UST_2Y"], by_id["UST_10Y"]
        if short.value is not None and long.value is not None:
            available = max((x for x in (short.available_at, long.available_at) if x), default=None)
            published = max((x for x in (short.published_at, long.published_at) if x), default=None)
            result["UST_10Y_2Y"] = Observation("UST_10Y_2Y", min(short.observation_date, long.observation_date),
                long.value-short.value, "derived:FRED", "percentage_points", published, available,
                revision=f"{short.revision}+{long.revision}", raw_value=long.value-short.value,
                metadata={"formula": "DGS10 - DGS2"})
    return result


def _market(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith((".TW", ".TWO")) or t.isdigit():
        return "TW"
    if t.endswith(".T"):
        return "JP"
    return "US"


# -------------------------------------------------------------------------
# Industry Sector Proxy Maps & Resolution
# -------------------------------------------------------------------------

US_SECTOR_MAP = {
    # Technology / Semi (XLK)
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "INTC": "XLK", "QCOM": "XLK", "TXN": "XLK", "AMAT": "XLK", "LRCX": "XLK",
    "MU": "XLK", "ASML": "XLK", "TSM": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "NOW": "XLK", "ORCL": "XLK", "PANW": "XLK", "CRWD": "XLK", "CSCO": "XLK",
    "IBM": "XLK", "INTU": "XLK", "SNPS": "XLK", "CDNS": "XLK", "ADI": "XLK",
    "KLAC": "XLK", "MRVL": "XLK", "NXPI": "XLK", "MCHP": "XLK", "APH": "XLK",

    # Communication Services (XLC)
    "GOOG": "XLC", "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "TMUS": "XLC", "VZ": "XLC", "T": "XLC", "EA": "XLC", "TTWO": "XLC",

    # Consumer Discretionary (XLY)
    "TSLA": "XLY", "AMZN": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "SBUX": "XLY", "LOW": "XLY", "BKNG": "XLY", "TJX": "XLY", "F": "XLY",
    "GM": "XLY", "MAR": "XLY", "ABNB": "XLY", "LULU": "XLY", "ORLY": "XLY",

    # Financials (XLF)
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF",
    "MS": "XLF", "BLK": "XLF", "SCHW": "XLF", "AXP": "XLF", "V": "XLF",
    "MA": "XLF", "PYPL": "XLF", "CB": "XLF", "MMC": "XLF", "PGR": "XLF",

    # Health Care (XLV)
    "LLY": "XLV", "UNH": "XLV", "JNJ": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "PFE": "XLV", "ISRG": "XLV",
    "BMY": "XLV", "AMGN": "XLV", "CVS": "XLV", "GILD": "XLV", "VRTX": "XLV",

    # Consumer Staples (XLP)
    "PG": "XLP", "COST": "XLP", "PEP": "XLP", "KO": "XLP", "WMT": "XLP",
    "PM": "XLP", "MDLZ": "XLP", "MO": "XLP", "CL": "XLP", "TGT": "XLP",

    # Energy (XLE)
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
    "MPC": "XLE", "PSX": "XLE", "VLO": "XLE", "OXY": "XLE", "WMB": "XLE",

    # Industrials (XLI)
    "GE": "XLI", "CAT": "XLI", "UNP": "XLI", "HON": "XLI", "BA": "XLI",
    "RTX": "XLI", "DE": "XLI", "LMT": "XLI", "UPS": "XLI", "FDX": "XLI",
    "ADP": "XLI", "ETN": "XLI", "WM": "XLI",

    # Materials (XLB)
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "ECL": "XLB", "FCX": "XLB",
    "NEM": "XLB", "CTVA": "XLB", "DOW": "XLB",

    # Utilities (XLU)
    "NEE": "XLU", "SO": "XLU", "DUK": "XLU", "CEG": "XLU", "SRE": "XLU", "AEP": "XLU",

    # Real Estate (XLRE)
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "CCI": "XLRE", "SPG": "XLRE", "PSA": "XLRE", "O": "XLRE",
}

TW_SECTOR_MAP = {
    # 科技 / 半導體 / 電子 (0052.TW)
    "2330": "0052.TW", "2454": "0052.TW", "2317": "0052.TW", "3661": "0052.TW",
    "2308": "0052.TW", "2382": "0052.TW", "3231": "0052.TW", "2379": "0052.TW",
    "3008": "0052.TW", "2303": "0052.TW", "6415": "0052.TW", "3034": "0052.TW",
    "3035": "0052.TW", "2357": "0052.TW", "4966": "0052.TW", "3443": "0052.TW",
    "6531": "0052.TW", "8299": "0052.TW", "2408": "0052.TW", "3711": "0052.TW",
    "2345": "0052.TW", "6669": "0052.TW", "2449": "0052.TW", "3037": "0052.TW",
    "2376": "0052.TW", "2356": "0052.TW", "2395": "0052.TW", "3045": "0052.TW",

    # 金融保險 (0055.TW)
    "2881": "0055.TW", "2882": "0055.TW", "2886": "0055.TW", "2891": "0055.TW",
    "2884": "0055.TW", "2880": "0055.TW", "2885": "0055.TW", "2883": "0055.TW",
    "2890": "0055.TW", "2892": "0055.TW", "2887": "0055.TW", "5880": "0055.TW",
    "2801": "0055.TW", "2834": "0055.TW", "5876": "0055.TW",

    # 傳產 / 航運 / 鋼鐵 / 塑化 (0050.TW)
    "2603": "0050.TW", "2609": "0050.TW", "2615": "0050.TW", "2618": "0050.TW",
    "2002": "0050.TW", "1301": "0050.TW", "1303": "0050.TW", "1326": "0050.TW",
    "6505": "0050.TW", "1101": "0050.TW", "1102": "0050.TW", "1216": "0050.TW",
    "2105": "0050.TW", "2912": "0050.TW", "9904": "0050.TW", "9910": "0050.TW",
}

JP_SECTOR_MAP = {
    # 電氣精密 / 半導體 (1625.T)
    "8035": "1625.T", "6857": "1625.T", "6146": "1625.T", "6758": "1625.T",
    "6920": "1625.T", "6501": "1625.T", "6702": "1625.T", "6723": "1625.T",
    "6981": "1625.T", "6762": "1625.T", "7741": "1625.T", "7751": "1625.T",
    "6861": "1625.T", "6503": "1625.T", "6701": "1625.T", "6645": "1625.T",
    "9984": "1625.T",

    # 自動車 / 輸送機 (1622.T)
    "7203": "1622.T", "7267": "1622.T", "7201": "1622.T", "7269": "1622.T",
    "7270": "1622.T", "6301": "1622.T", "6367": "1622.T", "7011": "1622.T",

    # 商社 (1629.T)
    "8058": "1629.T", "8001": "1629.T", "8031": "1629.T", "8002": "1629.T",
    "8053": "1629.T", "2768": "1629.T",

    # 金融 / 銀行 (1630.T)
    "8306": "1630.T", "8316": "1630.T", "8411": "1630.T", "8766": "1630.T",
    "8604": "1630.T", "8725": "1630.T", "8750": "1630.T",

    # 醫藥品 (1621.T)
    "4502": "1621.T", "4503": "1621.T", "4519": "1621.T", "4568": "1621.T",

    # 食品 (1617.T)
    "2802": "1617.T", "2502": "1617.T", "2503": "1617.T", "2914": "1617.T",
}


def get_ticker_industry_series(ticker: str) -> str:
    """Resolve an equity ticker to its corresponding market sector proxy series ID."""
    t = str(ticker or "").strip().upper()
    if not t:
        return "XLK"
    from core.market_calendar import MarketCalendar
    market = MarketCalendar.resolve_market(t)
    clean_code = t.split(".")[0]

    if market == "US":
        return US_SECTOR_MAP.get(clean_code, "XLK")
    elif market == "TW":
        return TW_SECTOR_MAP.get(clean_code, "0052.TW")
    elif market == "JP":
        return JP_SECTOR_MAP.get(clean_code, "1625.T")

    return "XLK"


class LiveSourceProvider:
    """Fetches data from public endpoints; supports injection in tests and offline runs."""
    FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, timeout=30):
        self.timeout = timeout

    def __call__(self, series_id: str, start: date, end: date) -> list[Observation]:
        if series_id in ("UST_2Y", "UST_10Y", "USD_TWD", "USD_JPY"):
            return self._fetch_fred(series_id, start, end)
        if series_id == "UST_10Y_2Y":
            a, b = self._fetch_fred("UST_2Y", start, end), self._fetch_fred("UST_10Y", start, end)
            av, bv = {o.observation_date:o for o in a}, {o.observation_date:o for o in b}
            out=[]
            for day in sorted(av.keys() & bv.keys()):
                x,y=av[day],bv[day]
                out.append(Observation(series_id,day,y.value-x.value,"derived:FRED","percentage_points",
                    max(x.published_at,y.published_at),max(x.available_at,y.available_at),
                    f"{x.revision}+{y.revision}",y.value-x.value,metadata={"formula":"DGS10-DGS2"}))
            return out
        if series_id == "OHLCV" or series_id.startswith("OHLCV:"):
            ticker=series_id.split(":",2)[1] if ":" in series_id else ""
            if not ticker:
                raise ValueError("OHLCV requires series id OHLCV:<ticker>")
            return self._fetch_yahoo(ticker, series_id, start, end, ohlcv=True)
        if series_id.startswith("TW_INSTITUTIONAL:"):
            return self._fetch_twse_institutional(series_id,start,end)
        d=COVARIATE_REGISTRY.get(series_id)
        if d is None or d.status == "blocked":
            raise ValueError(d.status_detail if d else f"unregistered series: {series_id}")
        if d.source == "Yahoo Finance chart API":
            return self._fetch_yahoo(d.source_code, series_id, start, end)
        if series_id == "TW_INSTITUTIONAL":
            raise ValueError("TWSE T86 historical daily adapter is not yet verified; see docs/covariates.md")
        raise ValueError(f"no live adapter for {series_id}")

    def _fetch_fred(self, series_id, start, end):
        code=COVARIATE_REGISTRY[series_id].source_code
        response=requests.get(self.FRED_URL,params={"id":code,"cosd":start.isoformat(),"coed":end.isoformat()},timeout=self.timeout)
        response.raise_for_status()
        rows=csv.DictReader(io.StringIO(response.text))
        out=[]
        for row in rows:
            raw=row.get(code)
            if not raw or raw==".": continue
            day=date.fromisoformat(row["observation_date"])
            if not start<=day<=end: continue
            if series_id in ("USD_TWD", "USD_JPY"):
                # Federal Reserve H.10 publishes the preceding week on Monday at 16:15 ET.
                # Federal holidays move the release to the following federal business day.
                from pandas.tseries.holiday import USFederalHolidayCalendar
                calendar = USFederalHolidayCalendar()
                holidays = {d.date() for d in calendar.holidays(start=day, end=day + timedelta(days=20))}
                release_day = day + timedelta(days=7 - day.weekday())
                while release_day.weekday() >= 5 or release_day in holidays:
                    release_day += timedelta(days=1)
                pub = datetime.combine(release_day, dtime(16, 15), ZoneInfo("America/New_York"))
                availability_day = release_day + timedelta(days=1)
                while availability_day.weekday() >= 5 or availability_day in holidays:
                    availability_day += timedelta(days=1)
                available = datetime.combine(availability_day, dtime(8), ZoneInfo("America/New_York"))
                availability_rule = "estimated H.10 weekly release; following federal business day 08:00 ET"
            else:
                pub=datetime.combine(day,dtime(16,30),ZoneInfo("America/New_York"))
                from core.market_calendar import MarketCalendar
                release_day=MarketCalendar.get_next_trading_day(day, market_or_ticker="US")
                available=datetime.combine(release_day,dtime(8,0),ZoneInfo("America/New_York"))
                availability_rule = "estimated next business day 08:00 ET; release timestamp unverified"
            val=float(raw)
            revision=hashlib.sha256(f"{series_id}:{day}:{val}".encode()).hexdigest()[:16]
            unit=COVARIATE_REGISTRY[series_id].unit
            out.append(Observation(series_id,day,val,"FRED", unit,pub,available,
                revision=f"fred-{revision}",raw_value=val,quality="estimated_availability",
                metadata={"availability_rule":availability_rule}))
        return out

    def _fetch_yahoo(self, ticker, series_id, start, end, ohlcv=False):
        import yfinance as yf
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        frame = None
        for attempt in range(3):
            try:
                frame = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    session=session,
                )
                if frame is not None and not frame.empty:
                    break
            except TypeError:
                frame = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if frame is not None and not frame.empty:
                    break
            except Exception as e:
                logger.warning(f"Yahoo fetch attempt {attempt+1} failed for {ticker}: {e}")
                if attempt < 2:
                    time.sleep(1)
        if frame is None or frame.empty:
            return []
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        definition = COVARIATE_REGISTRY.get(series_id)
        unit = definition.unit if definition else "native_market_units"
        out = []
        for ts, row in frame.iterrows():
            day = pd.Timestamp(ts).date()
            market = definition.market if definition and definition.market in {"TW", "US", "JP"} else _market(ticker)
            is_fx = series_id in ("USD_TWD", "USD_JPY")
            close_at = datetime.combine(
                day,
                dtime(17, 0) if is_fx else {"TW": dtime(13, 30), "JP": dtime(15, 30), "US": dtime(16, 0)}[market],
                ZoneInfo({"TW": "Asia/Taipei", "JP": "Asia/Tokyo", "US": "America/New_York"}[market])
            )
            if ohlcv:
                for field_name in ("Open", "High", "Low", "Close", "Volume", "Adj Close"):
                    if field_name in row and pd.notna(row[field_name]):
                        sid = f"{series_id}:{field_name.lower().replace(' ', '_')}"
                        value = float(row[field_name])
                        pub = close_at
                        revision = hashlib.sha256(f"{sid}:{day}:{value}".encode()).hexdigest()[:16]
                        out.append(Observation(
                            sid, day, value, "Yahoo Finance chart API", unit, pub, pub + timedelta(minutes=30),
                            revision=f"yahoo-{revision}", raw_value=value,
                            metadata={"ticker": ticker, "field": field_name, "adjusted": field_name == "Adj Close", "vintage_verified": True}
                        ))
                continue
            close = row.get("Close")
            if pd.isna(close):
                continue
            pub = close_at
            value = float(close)
            revision = hashlib.sha256(f"{series_id}:{day}:{value}".encode()).hexdigest()[:16]
            out.append(Observation(
                series_id, day, value, "Yahoo Finance chart API", unit, pub, pub + timedelta(minutes=30),
                revision=f"yahoo-{revision}", raw_value=value, quality="estimated_availability",
                metadata={"source_code": ticker, "normalization": definition.normalization if definition else "", "vintage_verified": True}
            ))
        return out

    def _fetch_twse_institutional(self, series_id, start, end):
        parts=series_id.split(":")
        if len(parts)!=3 or parts[2] not in ("foreign_net","trust_net","dealer_net","total_net"):
            raise ValueError("Use TW_INSTITUTIONAL:<TW ticker>:foreign_net|trust_net|dealer_net|total_net")
        code=parts[1].split(".")[0]
        out=[]
        day=start
        while day<=end:
            if day.weekday()<5:
                response=requests.get("https://www.twse.com.tw/rwd/zh/fund/T86",
                    params={"date":day.strftime("%Y%m%d"),"selectType":"ALLBUT0999","response":"json"},
                    headers={"User-Agent":"Mozilla/5.0"},timeout=self.timeout)
                response.raise_for_status()
                payload=response.json()
                fields=payload.get("fields",[])
                if payload.get("stat")=="OK" and payload.get("data"):
                    index={"foreign_net":4,"trust_net":10,"dealer_net":11,"total_net":18}[parts[2]]
                    row=next((r for r in payload["data"] if str(r[0]).strip()==code),None)
                    if row and len(row)>index:
                        clean_str = str(row[index]).replace(",", "").strip()
                        try:
                            raw = int(clean_str)
                        except ValueError:
                            continue
                        value = raw // 1000
                        published=datetime.combine(day,dtime(15,30),ZoneInfo("Asia/Taipei"))
                        available=datetime.combine(day,dtime(16,0),ZoneInfo("Asia/Taipei"))
                        revision=hashlib.sha256(f"{series_id}:{day}:{raw}".encode()).hexdigest()[:16]
                        out.append(Observation(series_id,day,value,"TWSE T86", "lots",published,available,
                            f"twse-{revision}",raw_value=raw,quality="estimated_availability",
                            metadata={"raw_unit":"shares","converted_unit":"lots","conversion":"shares // 1000",
                                      "fields":fields,"availability_rule":"estimated 16:00 Asia/Taipei"}))
            day+=timedelta(days=1)
        return out


class CovariateSynchronizer:
    def __init__(self, store=None, provider=None, retries=2, sleep=time.sleep):
        self.store=store or CovariateStore()
        self.provider=provider or LiveSourceProvider()
        self.retries=retries
        self.sleep=sleep

    def sync(self, series_ids, start: date, end: date):
        inserted=0; failed={}; fetched={}; sync_id=str(uuid.uuid4())
        for sid in series_ids:
            if sid.startswith("TW_INSTITUTIONAL:"):
                cursor=start; fail_chunks=[]; total=0
                while cursor<=end:
                    chunk_end=min(end,cursor+timedelta(days=29)); last=None
                    for attempt in range(self.retries+1):
                        try:
                            rows=self.provider(sid,cursor,chunk_end)
                            if not rows: raise ValueError("empty source response")
                            total+=len(rows); inserted+=self.store.upsert_observations(rows); last=None; break
                        except Exception as exc:
                            last=str(exc)
                            if attempt<self.retries: self.sleep(min(2**attempt,8))
                    if last is not None: fail_chunks.append(f"{cursor.isoformat()}..{chunk_end.isoformat()}: {last}")
                    cursor=chunk_end+timedelta(days=1)
                fetched[sid]=total
                if fail_chunks: failed[sid]="; ".join(fail_chunks)
                continue
            last=None
            for attempt in range(self.retries+1):
                try:
                    rows=self.provider(sid,start,end)
                    if not rows: raise ValueError("empty source response")
                    valid_rows = list(normalize_observations(rows).values())
                    count=self.store.upsert_observations(valid_rows)
                    inserted+=count; fetched[sid]=len(valid_rows); last=None; break
                except Exception as exc:
                    last=str(exc)
                    if attempt<self.retries: self.sleep(min(2**attempt,8))
            if last is not None: failed[sid]=last
        report={"sync_id":sync_id,"inserted":inserted,"fetched":fetched,"failed":failed,
                "start":start.isoformat(),"end":end.isoformat()}
        if hasattr(self.store,"record_sync_run"):
            self.store.record_sync_run(sync_id,series_ids,report,"partial" if failed else "succeeded")
        return report


@dataclass
class FeatureMatrix:
    dates: list[date]
    features: list[str]
    values: np.ndarray
    mask: np.ndarray
    newness: np.ndarray
    age_sessions: np.ndarray
    metadata: dict
    status: str
    missing_required: list[str]

    @property
    def past_covariates(self):
        """Adapter-ready [channels,time] numeric values; missing stays NaN, never zero-filled."""
        return self.values.T


PROFILES = {
    "TW": ["TWII", "SOX", "VIX", "USD_TWD", "UST_2Y", "UST_10Y_2Y", "INDUSTRY_TW", "OHLCV", "TW_INSTITUTIONAL"],
    "US": ["SPY", "VIX", "SOX", "UST_2Y", "UST_10Y", "UST_10Y_2Y", "INDUSTRY_US", "OHLCV"],
    "JP": ["N225", "TOPIX", "SOX", "VIX", "USD_JPY", "INDUSTRY_JP", "OHLCV"],
    "market_features": ["TWII", "SOX", "VIX", "USD_TWD", "USD_JPY", "UST_2Y", "UST_10Y_2Y", "OHLCV"],
    "market_features_TW": ["TWII", "SOX", "VIX", "USD_TWD", "UST_2Y", "UST_10Y_2Y", "OHLCV"],
    "market_features_US": ["SPY", "VIX", "SOX", "UST_2Y", "UST_10Y", "UST_10Y_2Y", "OHLCV"],
    "market_features_JP": ["N225", "TOPIX", "SOX", "VIX", "USD_JPY", "UST_2Y", "UST_10Y_2Y", "OHLCV"],
    "full_features": ["TWII", "SOX", "VIX", "USD_TWD", "UST_2Y", "UST_10Y_2Y", "industry", "OHLCV"],
    "full_features_TW": ["TWII", "SOX", "VIX", "USD_TWD", "UST_2Y", "UST_10Y_2Y", "INDUSTRY_TW", "OHLCV", "TW_INSTITUTIONAL"],
    "full_features_US": ["SPY", "VIX", "SOX", "UST_2Y", "UST_10Y", "UST_10Y_2Y", "INDUSTRY_US", "OHLCV"],
    "full_features_JP": ["N225", "TOPIX", "SOX", "VIX", "USD_JPY", "INDUSTRY_JP", "OHLCV"],
}


class CovariateBuilder:
    def __init__(self, store=None, provider=None, session_provider=None, max_source_sessions=5):
        self.store = store or CovariateStore()
        self.provider = provider or LiveSourceProvider()
        self.session_provider = session_provider or self._business_sessions
        self.max_source_sessions = max_source_sessions

    @staticmethod
    def _business_sessions(ticker, as_of, lookback):
        from core.market_calendar import MarketCalendar
        market = _market(ticker)
        local = as_of.astimezone(ZoneInfo({"TW": "Asia/Taipei", "JP": "Asia/Tokyo", "US": "America/New_York"}[market]))
        recent = MarketCalendar.get_recent_trading_days(lookback, reference_dt=local, market_or_ticker=market)
        return [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in reversed(recent)]

    @staticmethod
    def _row_cutoff(ticker, day, as_of):
        market = _market(ticker)
        zone = ZoneInfo({"TW": "Asia/Taipei", "JP": "Asia/Tokyo", "US": "America/New_York"}[market])
        close = {"TW": dtime(13, 30), "JP": dtime(15, 30), "US": dtime(16, 0)}[market]
        cutoff = datetime.combine(day, close, zone) + timedelta(hours=1)
        # Keep historical rows independent; only the live row is capped by decision time.
        local_as_of = as_of.astimezone(zone)
        if day == local_as_of.date() and cutoff > as_of:
            cutoff = as_of
        return cutoff

    @staticmethod
    def _source_market(series_id):
        definition = COVARIATE_REGISTRY.get(series_id)
        market = definition.market if definition else ""
        if market in ("TW", "US", "JP"):
            return market
        if series_id.startswith("TW_INSTITUTIONAL") or series_id.startswith("OHLCV:"):
            if series_id.startswith("TW_INSTITUTIONAL"):
                return "TW"
            ticker = series_id.split(":", 2)[1] if ":" in series_id else ""
            return _market(ticker) if ticker else None
        if series_id in ("USD_TWD", "USD_JPY"):
            return "US"
        return None

    def _source_age(self, series_id, observation_date, target_date):
        market = self._source_market(series_id)
        if not market or target_date <= observation_date:
            return max(0, (target_date - observation_date).days)
        from core.market_calendar import MarketCalendar
        tz = ZoneInfo({"TW": "Asia/Taipei", "JP": "Asia/Tokyo", "US": "America/New_York"}[market])
        reference = datetime.combine(target_date, dtime(23, 59), tz)
        days = MarketCalendar.get_recent_trading_days((target_date - observation_date).days + 10,
                                                       reference_dt=reference, market_or_ticker=market)
        parsed = [date.fromisoformat(str(value)[:10]) for value in days]
        return sum(observation_date < session <= target_date for session in parsed)

    def build_market_features(self, tickers, as_of, lookback=128, profile="auto", series_ids=None, persist_snapshot=True):
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        output = {}
        for ticker in tickers:
            market = _market(ticker)
            if profile in ("auto", "full_features"):
                selected = market
            elif profile == "market_features":
                selected = f"market_features_{market}"
            else:
                selected = profile
            if selected not in PROFILES:
                raise ValueError(f"unknown profile: {selected}")
            ids = list(series_ids) if series_ids is not None else PROFILES[selected]
            expanded = []
            for item in ids:
                if item == "OHLCV":
                    expanded.extend(f"OHLCV:{ticker}:{field}" for field in ("open", "high", "low", "close", "volume", "adj_close"))
                elif item == "TW_INSTITUTIONAL":
                    expanded.extend(f"TW_INSTITUTIONAL:{ticker.split('.')[0]}:{field}" for field in ("foreign_net", "trust_net", "dealer_net", "total_net"))
                elif item in ("INDUSTRY_TW", "INDUSTRY_US", "INDUSTRY_JP", "industry"):
                    expanded.append(get_ticker_industry_series(ticker))
                else:
                    expanded.append(item)
            ids = expanded
            dates = list(self.session_provider(ticker, as_of, lookback))
            if len(dates) > lookback:
                dates = dates[-lookback:]
            values = np.full((len(dates), len(ids)), np.nan, dtype=float)
            mask = np.zeros_like(values, dtype=bool)
            newness = np.zeros_like(mask)
            age_sessions = np.full_like(values, -1, dtype=np.int16)
            details = {}
            missing = []
            row_cutoffs = [self._row_cutoff(ticker, day, as_of) for day in dates]
            fetch_start = min(dates) - timedelta(days=20)
            for j, sid in enumerate(ids):
                actual_sid = sid
                try:
                    observations = self.store.get_observations(actual_sid, include_revisions=True)
                    if not observations and sid.startswith("OHLCV:"):
                        self.store.upsert_observations(self.provider(f"OHLCV:{ticker}", fetch_start, max(dates)))
                        observations = self.store.get_observations(actual_sid, as_of=as_of)
                    if not observations and sid.startswith("TW_INSTITUTIONAL:"):
                        try:
                            inst_rows = self.provider(sid, fetch_start, max(dates))
                            if inst_rows:
                                self.store.upsert_observations(inst_rows)
                                observations = self.store.get_observations(actual_sid, include_revisions=True)
                        except Exception as e:
                            logger.debug(f"TW_INSTITUTIONAL fetch omitted: {e}")
                    if not observations and sid in COVARIATE_REGISTRY and COVARIATE_REGISTRY[sid].status in ("verified", "partial"):
                        observations = self.provider(sid, fetch_start, max(dates))
                        self.store.upsert_observations(observations)
                        observations = self.store.get_observations(actual_sid, include_revisions=True)
                    observations = sorted(observations, key=lambda o: (o.observation_date, o.available_at or datetime.min.replace(tzinfo=timezone.utc), o.ingested_at or datetime.min.replace(tzinfo=timezone.utc), o.revision))
                    selected_by_day = {}
                    row_status = {}
                    for i, day in enumerate(dates):
                        cutoff = row_cutoffs[i]
                        eligible = [o for o in observations if o.observation_date <= day and o.value is not None
                                    and o.published_at is not None and o.published_at <= cutoff
                                    and o.available_at is not None and o.available_at <= cutoff
                                    and (o.source not in {"FRED", "derived:FRED", "Yahoo Finance chart API"}
                                         or o.metadata.get("vintage_verified") is True
                                         or (o.ingested_at is not None and o.ingested_at <= cutoff))]
                        match = max(eligible, key=lambda o: (o.observation_date, o.available_at, o.ingested_at or datetime.min.replace(tzinfo=timezone.utc), o.revision), default=None)
                        if match:
                            age = self._source_age(sid, match.observation_date, day)
                            if age <= self.max_source_sessions:
                                values[i, j] = float(match.value)
                                mask[i, j] = True
                                age_sessions[i, j] = age
                                newness[i, j] = match.observation_date == day
                                selected_by_day[day.isoformat()] = {
                                    "observation_date": match.observation_date.isoformat(),
                                    "available_at": match.available_at.isoformat(),
                                    "revision": match.revision,
                                    "age_sessions": age
                                }
                                row_status[day.isoformat()] = "available"
                            else:
                                row_status[day.isoformat()] = {
                                    "status": "stale",
                                    "age_sessions": age,
                                    "max_source_sessions": self.max_source_sessions,
                                    "observation_date": match.observation_date.isoformat()
                                }
                        else:
                            row_status[day.isoformat()] = "no_published_and_available_observation_asof_cutoff"
                    details[sid] = {
                        "observation_count": int(mask[:, j].sum()),
                        "source": "stored/asof",
                        "available_at_max": max((o.available_at.isoformat() for o in observations if o.available_at), default=None),
                        "selected_by_target_session": selected_by_day,
                        "target_session_status": row_status,
                        "revisions": [{
                            "date": o.observation_date.isoformat(),
                            "source": o.source,
                            "revision": o.revision,
                            "published_at": o.published_at.isoformat() if o.published_at else None,
                            "available_at": o.available_at.isoformat() if o.available_at else None
                        } for o in observations]
                    }
                    if not mask[:, j].all():
                        missing.append(sid)
                except Exception as exc:
                    details[sid] = {"error": str(exc)}
                    missing.append(sid)
            profile_id = f"{selected}-v1"
            snapshot_material = {
                "lineage": details,
                "values": [[None if not np.isfinite(v) else float(v) for v in row] for row in values],
                "mask": mask.tolist()
            }
            snapshot_id = hashlib.sha256((ticker + as_of.isoformat() + profile_id + json.dumps(snapshot_material, sort_keys=True)).encode()).hexdigest()[:24]
            status = "features_unavailable" if missing else "full_features"
            metadata = {
                "ticker": ticker,
                "market": market,
                "profile_id": profile_id,
                "as_of": as_of.isoformat(),
                "snapshot_id": snapshot_id,
                "source_lineage": details,
                "missing_required": missing,
                "mask_semantics": "true means value is available and as-of valid; NaN remains missing",
                "max_source_sessions": self.max_source_sessions,
                "lookback": lookback,
                "row_cutoffs": [cutoff.isoformat() for cutoff in row_cutoffs],
                "alignment": "each target session uses only observations published and available by its own close+1h cutoff"
            }
            result = FeatureMatrix(dates, ids, values, mask, newness, age_sessions, metadata, status, missing)
            if persist_snapshot:
                self.store.save_snapshot(snapshot_id, ticker, as_of, profile_id, {
                    "features": ids,
                    "values": snapshot_material["values"],
                    "mask": mask.tolist(),
                    "newness": newness.tolist(),
                    "age_sessions": age_sessions.tolist(),
                    "metadata": metadata
                })
            output[ticker] = result
        return output


def build_market_features(tickers, as_of, lookback=128, profile="auto", *, store=None, provider=None, session_provider=None, series_ids=None):
    """Return per-ticker, past-only matrices ready for forecast_batch past_covariates."""
    return CovariateBuilder(store,provider,session_provider).build_market_features(tickers,as_of,lookback,profile,series_ids)
