"""
test/test_public_ticker_seo.py - Tests for Public Ticker SEO, Escaped SSR Dossiers, Sitemap & Robots
"""

import unittest
import re
from fastapi.testclient import TestClient
from api.main import app, _escape_json_ld, _render_ssr_template
from data.duckdb_manager import DuckDBManager


class TestPublicTickerSEO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure at least one test record is in tw_daily_bars or predictions
        db = DuckDBManager()
        with db._get_connection(read_only=False) as con:
            con.execute("DELETE FROM tw_daily_bars WHERE ticker = '2330.TW' AND date = '2026-11-05'")
            con.execute("""
                INSERT INTO tw_daily_bars (date, ticker, raw_code, name, open, high, low, close, volume, market)
                VALUES ('2026-11-05', '2330.TW', '2330', '台積電', 1040.0, 1055.0, 1035.0, 1050.0, 35000000, 'TW')
            """)
            con.execute("""
                INSERT OR REPLACE INTO company_profiles (ticker, raw_code, name, industry, business_summary, data_json, updated_at)
                VALUES ('2330.TW', '2330', '台積電', '半導體', '全球晶圓代工龍頭，具備 2nm/3nm 先進製程與 CoWoS 封裝技術。', '{}', '2026-11-05')
            """)

    def test_escape_json_ld_safety(self):
        malicious = {
            "name": "</script><script>alert('xss')</script>",
            "desc": "Testing & < > symbols"
        }
        escaped = _escape_json_ld(malicious)
        self.assertNotIn("</script>", escaped)
        self.assertNotIn("<script>", escaped)
        self.assertIn("\\u003c/script\\u003e", escaped)
        self.assertIn("\\u003cscript\\u003e", escaped)
        self.assertIn("\\u0026", escaped)

    def test_ssr_template_autoescape(self):
        context = {
            "ticker": "XSS.TW",
            "name": "<img src=x onerror=alert(1)>",
            "market": "TW",
            "industry": "測試",
            "as_of": "2026-11-06",
            "current_price": 100.0,
            "predicted_price": 110.0,
            "potential": 10.0,
            "pe": 15.0,
            "pb": 2.0,
            "ma60": 95.0,
            "ma120": 90.0,
            "trust_net_5d": 100,
            "foreign_net_5d": 200,
            "business_summary": "<b>bold & unescaped?</b>",
            "peers": [],
            "safe_json_ld": "{}"
        }
        rendered = _render_ssr_template("ticker_seo.html", context)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&lt;img src=x", rendered)
        self.assertNotIn("<b>bold &", rendered)
        self.assertIn("&lt;b&gt;bold &amp;", rendered)

    def test_valid_ticker_returns_200_and_dossier(self):
        response = self.client.get("/s/2330.TW")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        body = response.text
        self.assertIn("台積電", body)
        self.assertIn("2330.TW", body)
        self.assertIn("canonical", body)
        self.assertIn("https://stockdata.david888.com/s/2330.TW", body)
        self.assertIn("application/ld+json", body)

    def test_valid_numeric_ticker_resolves_and_renders(self):
        response = self.client.get("/s/2330")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("台積電", body)

    def test_malformed_ticker_returns_404_noindex(self):
        malformed_inputs = [
            "<script>alert(1)</script>",
            "2330;DROP TABLE predictions;--",
            "TOOLONGTICKEREXCEEDINGTWELVECHARS",
            "invalid!chars",
            "   ",
        ]
        for bad_t in malformed_inputs:
            response = self.client.get(f"/s/{bad_t}")
            self.assertEqual(response.status_code, 404, f"Expected 404 for bad ticker {bad_t}")
            self.assertIn("noindex", response.headers.get("x-robots-tag", "").lower())
            self.assertIn("nofollow", response.headers.get("x-robots-tag", "").lower())

    def test_unknown_valid_format_ticker_returns_404_noindex(self):
        response = self.client.get("/s/UNKNOWN9999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("noindex", response.headers.get("x-robots-tag", "").lower())
        self.assertIn("nofollow", response.headers.get("x-robots-tag", "").lower())

    def test_sitemap_xml_returns_valid_xml_with_cache(self):
        response = self.client.get("/sitemap.xml")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/xml", response.headers.get("content-type", ""))
        self.assertIn("max-age=3600", response.headers.get("cache-control", ""))
        content = response.text
        self.assertIn("<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">", content)
        self.assertIn("<loc>https://stockdata.david888.com/</loc>", content)
        self.assertIn("<loc>https://stockdata.david888.com/s/2330.TW</loc>", content)
        # Check lastmod format YYYY-MM-DD
        self.assertTrue(re.search(r"<lastmod>\d{4}-\d{2}-\d{2}</lastmod>", content))

        # Test cached response on immediate second hit
        response2 = self.client.get("/sitemap.xml")
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response.text, response2.text)

    def test_robots_txt(self):
        response = self.client.get("/robots.txt")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers.get("content-type", ""))
        content = response.text
        self.assertIn("User-agent: *", content)
        self.assertIn("Allow: /s/", content)
        self.assertIn("Allow: /llms.txt", content)
        self.assertIn("Allow: /skill", content)
        self.assertIn("Allow: /.well-known/", content)
        self.assertIn("Disallow: /api/", content)
        self.assertIn("Disallow: /mcp/", content)
        self.assertIn("Sitemap: https://stockdata.david888.com/sitemap.xml", content)

    def test_peer_deduplication_in_dossier(self):
        """測試同產業同儕若有多筆不同時間戳預測記錄時，在公開頁中完全去重 (ROW_NUMBER() = 1)"""
        db = DuckDBManager()
        with db._get_connection(read_only=False) as con:
            con.execute("DELETE FROM predictions WHERE ticker = '1101.TW'")
            con.execute("""
                INSERT INTO predictions (index_name, model_name, strategy_type, ticker, current_price, predicted_price, potential, timestamp)
                VALUES
                ('TW0050', 'TimesFM', 'pred', '1101.TW', 32.0, 35.0, 9.3, '2026-11-05 10:00:00'),
                ('TW0050', 'Timestral', 'pred', '1101.TW', 32.0, 36.0, 12.5, '2026-11-06 10:00:00')
            """)

        resp = self.client.get("/s/2330.TW")
        self.assertEqual(resp.status_code, 200)
        # Verify 1101.TW peer link occurs exactly once in the rendered HTML
        peer_link_matches = re.findall(r'href="/s/1101\.TW"', resp.text)
        self.assertEqual(len(peer_link_matches), 1, "Expected peer 1101.TW to be deduplicated to exactly 1 link")


if __name__ == "__main__":
    unittest.main()
