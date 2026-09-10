import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from data.market_universe import MarketUniverseSync
from data.universe_providers import (
    EuronextProvider,
    HkexProvider,
    JpxProvider,
    NasdaqTraderProvider,
    LseProvider,
    SseProvider,
    SzseProvider,
    TwseTpexProvider,
)
from data.duckdb_manager import DuckDBManager


class StubProvider:
    source_id = "TEST"

    def __init__(self, records=None, error=None):
        self.records = records or []
        self.error = error

    def fetch_records(self):
        if self.error:
            raise self.error
        return self.records


def tw_record(symbol="2330", name="台積電"):
    return {
        "source_id": "TW-TWSE",
        "market": "TW",
        "exchange": "TWSE",
        "local_symbol": symbol,
        "normalized_symbol": f"{symbol}.TW",
        "name_local": name,
        "name_en": "",
        "security_type": "EQUITY",
        "listing_status": "ACTIVE",
        "source": "TWSE_OPENAPI",
    }


class TestUniverseProviders(unittest.TestCase):
    def test_nasdaq_symbol_directory_parses_both_official_files(self):
        nasdaq = (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares|Exchange|"
            "\nAAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N|NASDAQ|\n"
            "File Creation Time: 0909202612:00||||||||\n"
        )
        other = (
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol|\n"
            "IBM|International Business Machines|N|IBM|N|100|N|IBM|\n"
            "File Creation Time: 0909202612:00||||||||\n"
        )

        records = NasdaqTraderProvider.parse_files(nasdaq, other)

        self.assertEqual([r.local_symbol for r in records], ["AAPL", "IBM"])
        self.assertEqual(records[0].exchange, "NASDAQ")
        self.assertEqual(records[1].exchange, "NYSE")
        self.assertEqual(records[0].market, "US")
        self.assertEqual(records[0].security_type, "EQUITY")

    def test_hkex_rows_preserve_local_chinese_name_and_board_lot(self):
        rows = [
            {
                "Stock Code": "00700",
                "Name of Securities (Chinese)": "騰訊控股",
                "Name of Securities (English)": "TENCENT HOLDINGS LIMITED",
                "Board Lot": 100,
                "Category": "Equity",
            }
        ]

        records = HkexProvider.parse_rows(rows)

        self.assertEqual(records[0].local_symbol, "00700")
        self.assertEqual(records[0].normalized_symbol, "0700.HK")
        self.assertEqual(records[0].name_local, "騰訊控股")
        self.assertEqual(records[0].board_lot, 100)
        self.assertEqual(records[0].market, "HK")

    def test_jpx_rows_normalize_four_digit_code(self):
        frame = pd.DataFrame([
            {
                "Effective Date": 20260901,
                "Local Code": 7203,
                "Name (English)": "TOYOTA MOTOR CORPORATION",
                "Section/Products": "Prime Market (Domestic)",
            }
        ])

        records = JpxProvider.parse_frame(frame)

        self.assertEqual(records[0].local_symbol, "7203")
        self.assertEqual(records[0].normalized_symbol, "7203.T")
        self.assertEqual(records[0].name_en, "TOYOTA MOTOR CORPORATION")
        self.assertEqual(records[0].market_category, "Prime Market (Domestic)")

    def test_sse_payload_preserves_chinese_name_and_six_digit_code(self):
        payload = {
            "pageHelp": {
                "data": [{
                    "STOCK_TYPE": "8",
                    "A_STOCK_CODE": "688001",
                    "SEC_NAME_CN": "華興源創",
                    "FULL_NAME_IN_ENGLISH": "SUZHOU HYC TECHNOLOGY CO.,LTD",
                    "LIST_BOARD": "2",
                }]
            }
        }

        records = SseProvider.parse_payload(payload)

        self.assertEqual(records[0].local_symbol, "688001")
        self.assertEqual(records[0].normalized_symbol, "688001.SS")
        self.assertEqual(records[0].name_local, "華興源創")
        self.assertEqual(records[0].exchange, "SSE")

    def test_szse_rows_parse_a_share_code(self):
        frame = pd.DataFrame([
            {
                "板块": "主板",
                "公司全称": "平安银行股份有限公司",
                "英文名称": "Ping An Bank Co., Ltd.",
                "A股代码": 1,
                "A股简称": "平安银行",
            }
        ])

        records = SzseProvider.parse_frame(frame, "A")

        self.assertEqual(records[0].local_symbol, "000001")
        self.assertEqual(records[0].normalized_symbol, "000001.SZ")
        self.assertEqual(records[0].name_local, "平安银行")
        self.assertEqual(records[0].exchange, "SZSE")

    def test_euronext_payload_parses_html_cells(self):
        payload = {
            "aaData": [[
                '<a data-order="2CRSI">2CRSI</a>',
                "FR0013341781",
                "AL2SI",
                '<div title="Euronext Growth Paris">ALXP</div>',
            ]]
        }

        records = EuronextProvider.parse_payload(payload)

        self.assertEqual(records[0].local_symbol, "AL2SI")
        self.assertEqual(records[0].normalized_symbol, "AL2SI.ALXP")
        self.assertEqual(records[0].name_en, "2CRSI")
        self.assertEqual(records[0].isin, "FR0013341781")

    def test_euronext_official_csv_parses_all_market_rows(self):
        content = (
            "Name;ISIN;Symbol;Market;Currency\n"
            '"2020 BULKERS";BMG9156K1018;2020;"Oslo Børs";NOK\n'
            "2CRSI;FR0013341781;AL2SI;\"Euronext Growth Paris\";EUR\n"
        )

        records = EuronextProvider.parse_csv(content)

        self.assertEqual([record.local_symbol for record in records], ["2020", "AL2SI"])
        self.assertEqual(records[0].normalized_symbol, "2020.XOSL")
        self.assertEqual(records[1].normalized_symbol, "AL2SI.ALXP")

    def test_lse_rows_parse_official_directory_columns(self):
        frame = pd.DataFrame([
            {
                "Segment": "SET1",
                "Issuer Name": "BARCLAYS PLC",
                "ISIN": "GB0031348658",
                "Security Type": "DE",
                "MiFIR Identifier": "SHRS",
                "Mnemonic": "BARC",
                "Short Name": "BARCLAYS",
            }
        ])

        records = LseProvider.parse_frame(frame, "SETS")

        self.assertEqual(records[0].local_symbol, "BARC")
        self.assertEqual(records[0].normalized_symbol, "BARC.L")
        self.assertEqual(records[0].name_en, "BARCLAYS PLC")
        self.assertEqual(records[0].isin, "GB0031348658")
        self.assertEqual(records[0].exchange, "LSE")

    @patch("data.universe_providers.TWSEDailyFetcher.fetch_twse_quotes")
    @patch("data.universe_providers.TWSEDailyFetcher.fetch_tpex_quotes")
    @patch("data.universe_providers.TwseTpexProvider._fetch_stock_master")
    def test_twse_tpex_provider_uses_official_stock_master(self, master_mock, tpex_mock, twse_mock):
        twse_mock.return_value = pd.DataFrame(
            [{"ticker": "2330.TW", "raw_code": "2330", "name": "台積電", "market": "TWSE"}]
        )
        tpex_mock.return_value = pd.DataFrame(
            [{"ticker": "6488.TWO", "raw_code": "6488", "name": "環球晶", "market": "TPEX"}]
        )
        master_mock.side_effect = [
            pd.DataFrame(
                [{
                    "國際證券編碼": "TW0002330008",
                    "有價證券代號": "2330",
                    "有價證券名稱": "台積電",
                    "有價證券別": "股票",
                    "產業別": "半導體業",
                }]
            ),
            pd.DataFrame(
                [{
                    "國際證券編碼": "TW0006488009",
                    "有價證券代號": "6488",
                    "有價證券名稱": "環球晶",
                    "有價證券別": "股票",
                    "產業別": "半導體業",
                }]
            ),
        ]

        records = TwseTpexProvider().fetch_records()

        self.assertEqual({r.local_symbol for r in records}, {"2330", "6488"})
        self.assertEqual({r.exchange for r in records}, {"TWSE", "TPEX"})
        self.assertEqual({r.security_type for r in records}, {"EQUITY"})
        self.assertEqual(records[0].isin, "TW0002330008")


class TestMarketUniverseSync(unittest.TestCase):
    def test_failed_refresh_returns_last_successful_cache_as_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "universe.json")
            fresh = MarketUniverseSync(
                StubProvider([tw_record()]), cache_path=cache_path, min_records=1
            ).refresh()
            self.assertFalse(fresh.stale)

            stale = MarketUniverseSync(
                StubProvider(error=RuntimeError("upstream down")),
                cache_path=cache_path,
                min_records=1,
            ).refresh()

            self.assertTrue(stale.stale)
            self.assertEqual(stale.records[0]["local_symbol"], "2330")
            self.assertIn("upstream down", stale.error)
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            self.assertEqual(cached["records"][0]["local_symbol"], "2330")

    def test_empty_refresh_does_not_overwrite_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "universe.json")
            MarketUniverseSync(
                StubProvider([tw_record()]), cache_path=cache_path, min_records=1
            ).refresh()

            result = MarketUniverseSync(
                StubProvider([]), cache_path=cache_path, min_records=1
            ).refresh()

            self.assertTrue(result.stale)
            self.assertEqual(len(result.records), 1)
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            self.assertEqual(cached["records"][0]["local_symbol"], "2330")

    def test_suspiciously_small_refresh_does_not_replace_full_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "universe.json")
            full = [tw_record(str(code)) for code in range(1000, 1004)]
            MarketUniverseSync(
                StubProvider(full), cache_path=cache_path, min_records=1
            ).refresh()

            result = MarketUniverseSync(
                StubProvider([tw_record("1000")]), cache_path=cache_path, min_records=1
            ).refresh()

            self.assertTrue(result.stale)
            self.assertEqual(result.count, 4)

    def test_duckdb_persists_latest_source_snapshot_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            db = DuckDBManager(db_path=os.path.join(directory, "market.duckdb"))
            db.enabled = True
            db._init_schema()
            records = [tw_record()]

            first = db.save_market_universe_snapshot("2026-09-10", records)
            second = db.save_market_universe_snapshot("2026-09-10", records)
            rows = db.get_latest_market_universe("TW-TWSE")

            self.assertEqual(first, 1)
            self.assertEqual(second, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["normalized_symbol"], "2330.TW")

    def test_fallback_rehydrates_duckdb_from_json_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "universe.json")
            first_db = DuckDBManager(db_path=os.path.join(directory, "first.duckdb"))
            first_db.enabled = True
            first_db._init_schema()
            MarketUniverseSync(
                StubProvider([tw_record()]),
                cache_path=cache_path,
                db_manager=first_db,
                min_records=1,
            ).refresh()

            second_db = DuckDBManager(db_path=os.path.join(directory, "second.duckdb"))
            second_db.enabled = True
            second_db._init_schema()
            result = MarketUniverseSync(
                StubProvider(error=RuntimeError("temporary outage")),
                cache_path=cache_path,
                db_manager=second_db,
                min_records=1,
            ).refresh()

            self.assertTrue(result.stale)
            self.assertEqual(len(second_db.get_latest_market_universe("TW-TWSE")), 1)


if __name__ == "__main__":
    unittest.main()
