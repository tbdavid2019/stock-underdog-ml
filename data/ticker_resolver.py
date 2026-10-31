"""Normalize ticker input and resolve common company-name aliases."""

import re
import unicodedata
from typing import Optional


# Keep aliases explicit and conservative. An unknown name must never be
# converted into a ticker by guessing.
COMPANY_ALIASES = {
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "tesla inc.": "TSLA",
    "特斯拉": "TSLA",
    "特斯拉公司": "TSLA",
    "umc": "UMC",
    "聯電": "2303.TW",
    "聯華電子": "2303.TW",
    "統一": "1216.TW",
    "統一企業": "1216.TW",
    "台積電": "2330.TW",
    "台灣積體電路": "2330.TW",
    "tsmc": "2330.TW",
    "聯發科": "2454.TW",
    "聯發科技": "2454.TW",
    "mediatek": "2454.TW",
    "鴻海": "2317.TW",
    "鴻海精密": "2317.TW",
    "hon hai": "2317.TW",
    "軟銀": "9984.T",
    "軟銀集團": "9984.T",
    "softbank": "9984.T",
    "softbank group": "9984.T",
    "東京威力科創": "8035.T",
    "tokyo electron": "8035.T",
    "愛德萬測試": "6857.T",
    "advantest": "6857.T",
    "迪思科": "6146.T",
    "disco": "6146.T",
    "信越化學": "4063.T",
    "shin-etsu": "4063.T",
    "雷泰光電": "6920.T",
    "lasertec": "6920.T",
    "任天堂": "7974.T",
    "nintendo": "7974.T",
    "索尼": "6758.T",
    "sony": "6758.T",
    "豐田": "7203.T",
    "toyota": "7203.T",
    "優衣庫": "9983.T",
    "fast retailing": "9983.T",
}


def clean_query(value: str) -> str:
    """Trim and normalize Unicode variants without inventing input."""
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def normalize_ticker(value: str) -> str:
    """Normalize a ticker; bare numeric Taiwan codes receive the TW suffix."""
    query = clean_query(value).upper()
    if re.fullmatch(r"\d{4,6}", query):
        return f"{query}.TW"
    if re.fullmatch(r"\d{4,6}\.(TW|TWO)", query):
        code, market = query.split(".", 1)
        return f"{code}.{market}"
    return query


def resolve_alias(value: str) -> Optional[str]:
    """Return a known company alias mapping, or ``None`` when unknown."""
    query = clean_query(value).casefold()
    return COMPANY_ALIASES.get(query)
