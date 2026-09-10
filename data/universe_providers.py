"""Official security-universe providers with a market-neutral record shape."""

from __future__ import annotations

import csv
import html
import io
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests

from data.twse_daily_fetcher import TWSEDailyFetcher


@dataclass
class UniverseRecord:
    source_id: str
    market: str
    exchange: str
    local_symbol: str
    normalized_symbol: str
    name_local: str
    name_en: str = ""
    isin: Optional[str] = None
    security_type: str = "EQUITY"
    listing_status: str = "ACTIVE"
    market_category: Optional[str] = None
    financial_status: Optional[str] = None
    board_lot: Optional[int] = None
    source: str = ""
    source_updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TwseTpexProvider:
    source_id = "TW"
    ISIN_CLASSIFICATION_URL = "https://isin.twse.com.tw/isin/class_main.jsp"

    def __init__(self):
        self.last_snapshot_df = pd.DataFrame()

    @staticmethod
    def _records(df: pd.DataFrame, exchange: str) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for row in df.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).strip()
            raw_code = str(row.get("raw_code", "")).strip()
            if not raw_code or not ticker:
                continue
            records.append(
                UniverseRecord(
                    source_id="TW",
                    market="TW",
                    exchange=exchange,
                    local_symbol=raw_code,
                    normalized_symbol=ticker,
                    name_local=str(row.get("name", "") or "").strip(),
                    source=f"{exchange}_OPENAPI",
                )
            )
        return records

    @classmethod
    def _fetch_stock_master(cls, market: str, issuetype: str) -> pd.DataFrame:
        response = requests.get(
            cls.ISIN_CLASSIFICATION_URL,
            params={
                "market": market,
                "issuetype": issuetype,
                "industry_code": "",
                "Page": "1",
                "chklike": "Y",
            },
            timeout=45,
        )
        response.raise_for_status()
        response.encoding = "ms950"
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ValueError(f"no stock master table for market={market}")
        frame = tables[0]
        # pandas versions differ on whether the first HTML row is promoted to headers.
        if "有價證券代號" not in frame.columns and not frame.empty:
            header = [str(value).strip() for value in frame.iloc[0].tolist()]
            if "有價證券代號" in header:
                frame = frame.iloc[1:].copy()
                frame.columns = header
        return frame

    @staticmethod
    def _master_records(df: pd.DataFrame, exchange: str, suffix: str) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for row in df.to_dict(orient="records"):
            raw_code = str(row.get("有價證券代號", "")).strip()
            if not raw_code or raw_code.lower() == "nan":
                continue
            if raw_code.isdigit():
                raw_code = raw_code.zfill(4)
            records.append(
                UniverseRecord(
                    source_id="TW",
                    market="TW",
                    exchange=exchange,
                    local_symbol=raw_code,
                    normalized_symbol=f"{raw_code}{suffix}",
                    name_local=str(row.get("有價證券名稱", "") or "").strip(),
                    isin=str(row.get("國際證券編碼", "") or "").strip() or None,
                    security_type="EQUITY",
                    market_category=str(row.get("產業別", "") or "").strip() or None,
                    source="TWSE_ISIN_STOCK_CLASSIFICATION",
                    source_updated_at=str(row.get("公開發行/上市(櫃)/發行日", "") or "").strip() or None,
                )
            )
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        twse = TWSEDailyFetcher.fetch_twse_quotes()
        tpex = TWSEDailyFetcher.fetch_tpex_quotes()
        twse_master = self._fetch_stock_master("1", "1")
        tpex_master = self._fetch_stock_master("2", "4")
        if twse.empty or tpex.empty or twse_master.empty or tpex_master.empty:
            raise ValueError("TWSE and TPEX quote/master snapshots are both required")
        self.last_snapshot_df = pd.concat([twse, tpex], ignore_index=True)
        return self._master_records(twse_master, "TWSE", ".TW") + self._master_records(tpex_master, "TPEX", ".TWO")


class NasdaqTraderProvider:
    source_id = "US-NASDAQ-TRADER"
    NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    EXCHANGES = {
        "N": "NYSE",
        "A": "NYSE_AMERICAN",
        "P": "NYSE_ARCA",
        "Z": "BATS",
        "V": "IEX",
    }

    @classmethod
    def parse_files(cls, nasdaq_text: str, other_text: str) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []

        for row in csv.DictReader(io.StringIO(nasdaq_text), delimiter="|"):
            symbol = (row.get("Symbol") or "").strip()
            if not symbol or symbol.startswith("File Creation Time"):
                continue
            records.append(
                UniverseRecord(
                    source_id=cls.source_id,
                    market="US",
                    exchange="NASDAQ",
                    local_symbol=symbol,
                    normalized_symbol=symbol,
                    name_local=(row.get("Security Name") or "").strip(),
                    name_en=(row.get("Security Name") or "").strip(),
                    security_type="EQUITY",
                    market_category=(row.get("Market Category") or "").strip() or None,
                    financial_status=(row.get("Financial Status") or "").strip() or None,
                    board_lot=_int_or_none(row.get("Round Lot Size")),
                    source="NASDAQ_TRADER_NASDAQLISTED",
                )
            )

        for row in csv.DictReader(io.StringIO(other_text), delimiter="|"):
            symbol = (row.get("ACT Symbol") or "").strip()
            if not symbol or symbol.startswith("File Creation Time"):
                continue
            exchange_code = (row.get("Exchange") or "").strip()
            exchange = cls.EXCHANGES.get(exchange_code, exchange_code or "OTHER_US")
            records.append(
                UniverseRecord(
                    source_id=cls.source_id,
                    market="US",
                    exchange=exchange,
                    local_symbol=symbol,
                    normalized_symbol=symbol,
                    name_local=(row.get("Security Name") or "").strip(),
                    name_en=(row.get("Security Name") or "").strip(),
                    security_type="ETF" if (row.get("ETF") or "").strip() == "Y" else "EQUITY",
                    board_lot=_int_or_none(row.get("Round Lot Size")),
                    source="NASDAQ_TRADER_OTHERLISTED",
                )
            )
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        nasdaq = _get_text(self.NASDAQ_URL)
        other = _get_text(self.OTHER_URL)
        return self.parse_files(nasdaq, other)


class HkexProvider:
    source_id = "HK-HKEX"
    PAGE_URL = "https://www.hkex.com.hk/Services/Trading/Securities/Securities-Lists?sc_lang=en"
    DIRECT_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"

    @staticmethod
    def parse_rows(rows: Iterable[Dict[str, Any]]) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for row in rows:
            code = _first_value(row, "Stock Code", "Stock code", "Code", "股份代號")
            code = str(code or "").strip()
            if not code or not re.fullmatch(r"\d{1,5}", code):
                continue
            code = code.zfill(5)
            category = str(_first_value(row, "Category", "Security Type", "股份類別") or "EQUITY").strip()
            records.append(
                UniverseRecord(
                    source_id=HkexProvider.source_id,
                    market="HK",
                    exchange="HKEX",
                    local_symbol=code,
                    normalized_symbol=f"{int(code):04d}.HK",
                    name_local=str(_first_value(row, "Name of Securities (Chinese)", "Chinese Name", "中文股份名稱", "Name of Securities") or "").strip(),
                    name_en=str(_first_value(row, "Name of Securities (English)", "English Name", "英文股份名稱", "Name of Securities") or "").strip(),
                    isin=str(_first_value(row, "ISIN", "ISIN Code", "國際證券識別碼") or "").strip() or None,
                    security_type=category.upper(),
                    board_lot=_int_or_none(_first_value(row, "Board Lot", "Board Lot Size", "每手股數")),
                    source="HKEX_SECURITIES_LIST",
                )
            )
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        response = requests.get(self.DIRECT_URL, timeout=30)
        if response.status_code != 200 or not response.content:
            page = requests.get(self.PAGE_URL, timeout=30)
            page.raise_for_status()
            links = re.findall(r"href=[\"']([^\"']+\.(?:xlsx|xls|csv)(?:\?[^\"']*)?)[\"']", page.text, re.I)
            links = [urljoin(self.PAGE_URL, link) for link in links]
            preferred = next((u for u in links if "secur" in u.lower() or "list" in u.lower()), None)
            if not preferred:
                raise RuntimeError("HKEX securities-list download link not found")
            response = requests.get(preferred, timeout=30)
            response.raise_for_status()
        # HKEX's current workbook has title/date rows before the field header.
        frame = pd.read_excel(BytesIO(response.content), header=2)
        return self.parse_rows(frame.to_dict(orient="records"))


class JpxProvider:
    source_id = "JP-JPX"
    LIST_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_e.xlsx"

    @staticmethod
    def parse_frame(frame: pd.DataFrame) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for row in frame.to_dict(orient="records"):
            code = _normalize_code(row.get("Local Code"), 4)
            if not code:
                continue
            section = str(row.get("Section/Products") or "").strip()
            name = str(row.get("Name (English)") or "").strip()
            security_type = "ETF" if "ETF" in section.upper() else "EQUITY"
            records.append(
                UniverseRecord(
                    source_id=JpxProvider.source_id,
                    market="JP",
                    exchange="JPX",
                    local_symbol=code,
                    normalized_symbol=f"{code}.T",
                    name_local=name,
                    name_en=name,
                    security_type=security_type,
                    market_category=section or None,
                    source="JPX_TSE_LISTED_ISSUES",
                    source_updated_at=str(row.get("Effective Date") or "").strip() or None,
                )
            )
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        response = requests.get(self.LIST_URL, timeout=45)
        response.raise_for_status()
        frame = pd.read_excel(BytesIO(response.content))
        return self.parse_frame(frame)


class SseProvider:
    source_id = "CN-SSE"
    QUERY_URL = "https://query.sse.com.cn/sseQuery/commonQuery.do"
    STOCK_TYPES = ("1", "2", "8")  # A shares, B shares, STAR market

    @staticmethod
    def parse_payload(payload: Dict[str, Any]) -> List[UniverseRecord]:
        rows = payload.get("pageHelp", {}).get("data", [])
        records: List[UniverseRecord] = []
        for row in rows:
            stock_type = str(row.get("STOCK_TYPE") or "1").strip()
            raw_code = row.get("B_STOCK_CODE") if stock_type == "2" else row.get("A_STOCK_CODE")
            code = _normalize_code(raw_code, 6)
            if not code:
                continue
            name_local = str(row.get("SEC_NAME_CN") or row.get("COMPANY_ABBR") or "").strip()
            name_en = str(row.get("FULL_NAME_IN_ENGLISH") or row.get("COMPANY_ABBR_EN") or "").strip()
            records.append(
                UniverseRecord(
                    source_id=SseProvider.source_id,
                    market="CN",
                    exchange="SSE",
                    local_symbol=code,
                    normalized_symbol=f"{code}.SS",
                    name_local=name_local,
                    name_en=name_en,
                    security_type="EQUITY",
                    market_category=str(row.get("LIST_BOARD") or "").strip() or None,
                    source="SSE_COMMON_QUERY_STOCK_LIST",
                    source_updated_at=str(row.get("LIST_DATE") or "").strip() or None,
                )
            )
        return records

    def _fetch_type(self, stock_type: str) -> List[UniverseRecord]:
        page_no = 1
        page_size = 2000
        records: List[UniverseRecord] = []
        while True:
            params = {
                "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
                "STOCK_TYPE": stock_type,
                "COMPANY_STATUS": "2,4,5,7,8",
                "type": "inParams",
                "isPagination": "true",
                "pageHelp.cacheSize": "1",
                "pageHelp.beginPage": str(page_no),
                "pageHelp.pageSize": str(page_size),
                "pageHelp.pageNo": str(page_no),
            }
            response = requests.get(
                self.QUERY_URL,
                params=params,
                headers={"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"},
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            page_records = self.parse_payload(payload)
            records.extend(page_records)
            total = int(payload.get("pageHelp", {}).get("total") or len(records))
            if not page_records or len(records) >= total:
                break
            page_no += 1
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for stock_type in self.STOCK_TYPES:
            records.extend(self._fetch_type(stock_type))
        return records


class SzseProvider:
    source_id = "CN-SZSE"
    DOWNLOAD_URL = "https://www.szse.cn/api/report/ShowReport"
    TABS = (("tab1", "A"), ("tab2", "B"))

    @staticmethod
    def parse_frame(frame: pd.DataFrame, share_type: str) -> List[UniverseRecord]:
        code_column = "A股代码" if share_type == "A" else "B股代码"
        name_column = "A股簡稱" if share_type == "A" else "B股簡稱"
        if name_column not in frame.columns:
            name_column = "A股简称" if share_type == "A" else "B股简称"
        records: List[UniverseRecord] = []
        for row in frame.to_dict(orient="records"):
            code = _normalize_code(row.get(code_column), 6)
            if not code:
                continue
            name_local = str(row.get(name_column) or "").strip()
            name_en = str(row.get("英文名称") or row.get("公司全称") or "").strip()
            records.append(
                UniverseRecord(
                    source_id=SzseProvider.source_id,
                    market="CN",
                    exchange="SZSE",
                    local_symbol=code,
                    normalized_symbol=f"{code}.SZ",
                    name_local=name_local,
                    name_en=name_en,
                    security_type="EQUITY",
                    market_category=str(row.get("板块") or "").strip() or None,
                    source="SZSE_OFFICIAL_STOCK_LIST",
                    source_updated_at=str(row.get("A股上市日期") or row.get("B股上市日期") or "").strip() or None,
                )
            )
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for tab, share_type in self.TABS:
            response = requests.get(
                self.DOWNLOAD_URL,
                params={"SHOWTYPE": "xlsx", "CATALOGID": "1110", "TABKEY": tab, "random": "0.1"},
                headers={"Referer": "https://www.szse.cn/market/product/stock/list/index.html", "User-Agent": "Mozilla/5.0"},
                timeout=45,
            )
            response.raise_for_status()
            frame = pd.read_excel(BytesIO(response.content))
            records.extend(self.parse_frame(frame, share_type))
        return records


class EuronextProvider:
    source_id = "EU-EURONEXT"
    PAGE_URL = "https://live.euronext.com/en/products/equities/list"
    BASE_URL = "https://live.euronext.com"
    DOWNLOAD_URL = "https://live.euronext.com/product_directory/data/stocks-all-places/download"
    MIC_FILTER = (
        "ALXB,ALXL,ALXP,BGEM,ENXB,ENXL,ETLX,EXGM,MERK,MIVX,MLXB,MTAA,MTAH,"
        "TNLA,TNLB,XAMC,XAMS,XATL,XBRU,XESM,XLDN,XLIS,XMLI,XMSM,XOAS,XOSL,XPAR,XPMC"
    )
    MARKET_MICS = {
        "Euronext Amsterdam": "XAMS",
        "Euronext Brussels": "XBRU",
        "Euronext Dublin": "XESM",
        "Euronext Lisbon": "XLIS",
        "Euronext Milan": "XMIL",
        "Oslo Børs": "XOSL",
        "Euronext Paris": "XPAR",
        "Euronext Growth Brussels": "ENXB",
        "Euronext Growth Dublin": "ENXL",
        "Euronext Growth Lisbon": "ENXL",
        "Euronext Growth Milan": "ETLX",
        "Euronext Growth Oslo": "MOTX",
        "Euronext Growth Paris": "ALXP",
        "Euronext Access Lisbon": "ALXL",
        "Euronext Access Brussels": "ALXB",
        "Euronext Access Paris": "ALXP",
        "Euronext Expand Oslo": "MERK",
        "Euronext Global Equity Market": "BGEM",
        "EuroTLX": "ETLX",
        "Trading After Hours": "MTAH",
    }

    @staticmethod
    def parse_payload(payload: Dict[str, Any]) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for row in payload.get("aaData", []):
            if len(row) < 4:
                continue
            name = _html_text(row[0], attribute="data-order")
            isin = _html_text(row[1])
            symbol = _html_text(row[2])
            mic = _html_text(row[3])
            if not symbol or not mic or not re.fullmatch(r"[A-Z0-9.\-]+", symbol):
                continue
            records.append(
                UniverseRecord(
                    source_id=EuronextProvider.source_id,
                    market="EU",
                    exchange="EURONEXT",
                    local_symbol=symbol,
                    normalized_symbol=f"{symbol}.{mic}",
                    name_local=name,
                    name_en=name,
                    isin=isin or None,
                    security_type="EQUITY",
                    market_category=mic,
                    source="EURONEXT_STOCKS_ALL_PLACES",
                )
            )
        return records

    @classmethod
    def parse_csv(cls, content: str) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        reader = csv.DictReader(StringIO(content), delimiter=";")
        for row in reader:
            symbol = str(row.get("Symbol") or "").strip()
            isin = str(row.get("ISIN") or "").strip()
            market = str(row.get("Market") or "").strip()
            name = str(row.get("Name") or "").strip()
            if not symbol or not isin or not market or symbol.lower() == "nan":
                continue
            mic = cls.MARKET_MICS.get(market, market.upper().replace(" ", "_"))
            records.append(
                UniverseRecord(
                    source_id=cls.source_id,
                    market="EU",
                    exchange="EURONEXT",
                    local_symbol=symbol,
                    normalized_symbol=f"{symbol}.{mic}",
                    name_local=name,
                    name_en=name,
                    isin=isin,
                    security_type="EQUITY",
                    market_category=market,
                    source="EURONEXT_STOCKS_ALL_PLACES",
                )
            )
        return records

    def fetch_records(self) -> List[UniverseRecord]:
        response = requests.get(
            self.DOWNLOAD_URL,
            params={"mics": self.MIC_FILTER},
            headers={"Accept": "text/csv", "User-Agent": "Mozilla/5.0", "Referer": self.PAGE_URL},
            timeout=60,
        )
        response.raise_for_status()
        return self.parse_csv(response.content.decode("utf-8-sig"))


class LseProvider:
    source_id = "GB-LSE"
    PAGE_API = "https://api.londonstockexchange.com/api/v1/pages"
    PAGE_PATH = "equities-trading/asset-classes/shares-trading/uk-and-european-securities"

    @staticmethod
    def parse_frame(frame: pd.DataFrame, venue: str) -> List[UniverseRecord]:
        records: List[UniverseRecord] = []
        for row in frame.to_dict(orient="records"):
            symbol = str(row.get("Mnemonic") or "").strip()
            isin = str(row.get("ISIN") or "").strip()
            if not symbol or not isin or symbol.lower() == "nan":
                continue
            name = str(row.get("Issuer Name") or row.get("Long Name") or row.get("Short Name") or "").strip()
            records.append(
                UniverseRecord(
                    source_id=LseProvider.source_id,
                    market="GB",
                    exchange="LSE",
                    local_symbol=symbol,
                    normalized_symbol=f"{symbol}.L",
                    name_local=name,
                    name_en=name,
                    isin=isin,
                    security_type=str(row.get("MiFIR Identifier") or row.get("Security Type") or "EQUITY").strip(),
                    market_category=venue,
                    source="LSE_UK_EUROPEAN_SECURITIES",
                )
            )
        return records

    @staticmethod
    def _documents(value: Any) -> List[Dict[str, str]]:
        found: List[Dict[str, str]] = []
        if isinstance(value, dict):
            if value.get("url") and value.get("title"):
                found.append({"url": str(value["url"]), "title": str(value["title"])})
            for child in value.values():
                found.extend(LseProvider._documents(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(LseProvider._documents(child))
        return found

    def fetch_records(self) -> List[UniverseRecord]:
        response = requests.get(
            self.PAGE_API,
            params={"path": self.PAGE_PATH},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.londonstockexchange.com/"},
            timeout=45,
        )
        response.raise_for_status()
        documents = self._documents(response.json())
        documents = [
            item for item in documents
            if item["url"].lower().endswith((".xlsx", ".xls"))
            and "securities" in item["title"].lower()
        ]
        if not documents:
            raise ValueError("LSE official securities-list downloads not found")
        records: List[UniverseRecord] = []
        for document in documents:
            workbook = requests.get(document["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
            workbook.raise_for_status()
            frame = pd.read_excel(BytesIO(workbook.content), header=3)
            records.extend(self.parse_frame(frame, document["title"]))
        return records


def _get_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _first_value(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_code(value: Any, width: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(width)
    return text


def _html_text(value: Any, attribute: Optional[str] = None) -> str:
    text = str(value or "")
    if attribute:
        match = re.search(rf'{re.escape(attribute)}=["\']([^"\']+)', text, flags=re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None
