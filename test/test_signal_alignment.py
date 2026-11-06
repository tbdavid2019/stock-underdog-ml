"""
test/test_signal_alignment.py
Unit tests for deterministic multi-source signal alignment synthesis.
"""

import unittest
import tempfile
import os
import shutil
import json
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from data.duckdb_manager import DuckDBManager
from data.signal_alignment import compute_signal_alignment
from api.main import app
from api.routes.predictions import get_duckdb


class TestSignalAlignment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_align.duckdb")
        self.db = DuckDBManager(db_path=self.db_path)
        app.dependency_overrides[get_duckdb] = lambda: self.db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_duckdb, None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _seed_sample_data(self):
        """Seed sample predictions, institutional flows, podcasts, and discovery data."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        old_iso = (now - timedelta(days=45)).isoformat()
        recent_iso = (now - timedelta(days=5)).isoformat()

        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            # 1. Predictions
            con.execute("""
                INSERT INTO predictions (
                    index_name, model_name, strategy_type, ticker, current_price, predicted_price,
                    potential, timestral_potential, timesfm_potential, moirai_potential,
                    trust_net_5d, foreign_net_5d, timestamp
                ) VALUES
                ('TW0050', 'ensemble', 'resonance', '2330.TW', 1000.0, 1050.0, 5.0, 6.0, 4.5, 4.8, 5000, 12000, ?),
                ('TW0050', 'ensemble', 'resonance', '2454.TW', 1200.0, 1100.0, -8.3, -7.0, -8.5, -9.0, 3000, 4000, ?),
                ('SP500', 'ensemble', 'resonance', 'NVDA', 130.0, 145.0, 11.5, 12.0, 10.0, 12.5, NULL, NULL, ?),
                ('TW0050', 'ensemble', 'resonance', '2634.TW', 50.0, 50.1, 0.2, 0.1, 0.2, 0.3, 0, 0, ?)
            """, [now_iso, now_iso, now_iso, now_iso])

            # 2. Podcasts (recent for 2330.TW and NVDA, old/stale for 2634.TW)
            con.execute("""
                INSERT INTO podcast_episodes (
                    id, title, channel_id, published_at, ticker_sentiments
                ) VALUES
                ('ep1', '台積電先進製程展望', 'ch1', ?, ?),
                ('ep2', '美股 AI 晶片狂潮', 'ch2', ?, ?),
                ('ep3', '航太產業舊聞', 'ch3', ?, ?)
            """, [
                recent_iso, json.dumps({"2330.TW": {"stance": "bullish", "label": "↑ 看多"}}),
                recent_iso, json.dumps({"NVDA": {"stance": "bullish", "label": "↑ 看多"}}),
                old_iso, json.dumps({"2634.TW": {"stance": "bullish", "label": "↑ 看多"}})
            ])

            # 3. Discovery Candidates (2330.TW in pool, 2454.TW observe)
            con.execute("""
                INSERT INTO discovery_candidates (
                    ticker, market, name, theme_id, theme_name, total_score, recommendation, as_of
                ) VALUES
                ('2330.TW', 'TW', '台積電', 'ai_semiconductors', 'AI 晶片先進封裝', 88.5, 'already_in_pool', ?),
                ('2454.TW', 'TW', '聯發科', 'edge_ai', '邊緣運算', 55.0, 'observe', ?),
                ('NVDA', 'US', 'NVIDIA', 'ai_accelerators', 'AI 加速運算', 92.0, 'propose_add', ?)
            """, [now_iso, now_iso, now_iso])

    def test_bullish_alignment(self):
        """2330.TW: model(+1), inst(+1), podcast(+1), discovery(+1) -> bullish_alignment."""
        self._seed_sample_data()
        res = compute_signal_alignment("2330.TW", db=self.db)

        self.assertEqual(res["ticker"], "2330.TW")
        self.assertEqual(res["market"], "TW")
        self.assertEqual(res["alignment_status"], "bullish_alignment")
        self.assertEqual(res["active_sources_count"], 4)
        self.assertEqual(res["total_applicable_sources"], 4)
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["dimensions"]["model"]["score"], 1)
        self.assertEqual(res["dimensions"]["institutional"]["score"], 1)
        self.assertEqual(res["dimensions"]["podcast"]["score"], 1)
        self.assertEqual(res["dimensions"]["discovery"]["score"], 1)

    def test_divergence_warning(self):
        """2454.TW: model(-1) conflicts with institutional(+1) -> divergence_warning."""
        self._seed_sample_data()
        res = compute_signal_alignment("2454.TW", db=self.db)

        self.assertEqual(res["ticker"], "2454.TW")
        self.assertEqual(res["market"], "TW")
        self.assertEqual(res["alignment_status"], "divergence_warning")
        self.assertEqual(res["dimensions"]["model"]["score"], -1)
        self.assertEqual(res["dimensions"]["institutional"]["score"], 1)
        self.assertIsNotNone(res["score"])

    def test_overseas_stock_excludes_institutional(self):
        """NVDA (US): institutional dimension must be applicable=False and score=None."""
        self._seed_sample_data()
        res = compute_signal_alignment("NVDA", db=self.db)

        self.assertEqual(res["ticker"], "NVDA")
        self.assertEqual(res["market"], "US")
        inst_dim = res["dimensions"]["institutional"]
        self.assertFalse(inst_dim["applicable"])
        self.assertIsNone(inst_dim["score"])
        self.assertEqual(inst_dim["details"]["reason"], "overseas_stock_not_applicable")

        # Total applicable sources for US should be 3 (model, podcast, discovery)
        self.assertEqual(res["total_applicable_sources"], 3)
        self.assertEqual(res["active_sources_count"], 3)
        self.assertEqual(res["alignment_status"], "bullish_alignment")
        self.assertEqual(res["score"], 1.0)

    def test_stale_podcast_exclusion(self):
        """2634.TW: podcast is > 30 days old -> stale=True and excluded from active count."""
        self._seed_sample_data()
        res = compute_signal_alignment("2634.TW", db=self.db)

        pod_dim = res["dimensions"]["podcast"]
        self.assertTrue(pod_dim["stale"])
        self.assertIsNone(pod_dim["score"])
        self.assertEqual(pod_dim["details"]["reason"], "stale_over_30d")

    def test_stale_model_exclusion(self):
        """測試時序預測資料逾 14 天（10 交易日）時標記為 stale=True 且 score=None"""
        old_time = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            con.execute("""
                INSERT INTO predictions (
                    index_name, model_name, strategy_type, ticker, current_price, predicted_price,
                    potential, timestamp
                ) VALUES
                ('TW0050', 'ensemble', 'resonance', '2317.TW', 100.0, 110.0, 10.0, ?)
            """, [old_time])

        res = compute_signal_alignment("2317.TW", db=self.db)
        model_dim = res["dimensions"]["model"]
        self.assertTrue(model_dim["stale"])
        self.assertIsNone(model_dim["score"])
        self.assertEqual(model_dim["details"]["reason"], "model_data_stale_over_10_business_days")

    def test_institutional_1d_fallback(self):
        """測試 5 日法人買賣超為 None 時，能回退至 tw_institutional_daily 1 日（昨日）買賣超"""
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        today_date = now_dt.strftime("%Y-%m-%d")
        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            con.execute("""
                INSERT INTO predictions (
                    index_name, model_name, strategy_type, ticker, current_price, predicted_price,
                    potential, trust_net_5d, foreign_net_5d, timestamp
                ) VALUES
                ('TW0050', 'ensemble', 'resonance', '2881.TW', 80.0, 85.0, 6.0, NULL, NULL, ?)
            """, [now_iso])
            con.execute("""
                INSERT INTO tw_institutional_daily (
                    date, ticker, raw_code, name, foreign_net, trust_net, dealer_net, total_net
                ) VALUES
                (?, '2881.TW', '2881', '富邦金', 3000, 1500, 0, 4500)
            """, [today_date])

        res = compute_signal_alignment("2881.TW", db=self.db)
        inst_dim = res["dimensions"]["institutional"]
        self.assertEqual(inst_dim["details"].get("period"), "1d")
        self.assertEqual(inst_dim["score"], 1)

    def test_unparseable_podcast_date_marked_not_fresh(self):
        """測試 Podcast 發布時間若損壞無法解析時標記為不新鮮 (stale=True)"""
        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            con.execute("""
                INSERT INTO podcast_episodes (
                    id, title, channel_id, published_at, ticker_sentiments
                ) VALUES
                ('ep_corrupt', '壞死時間節目', 'ch_corrupt', 'invalid_time_format', ?)
            """, [json.dumps({"3008.TW": {"stance": "bullish", "label": "↑ 看多"}})])

        res = compute_signal_alignment("3008.TW", db=self.db)
        pod_dim = res["dimensions"]["podcast"]
        self.assertTrue(pod_dim["stale"])
        self.assertIsNone(pod_dim["score"])

    def test_stale_discovery_candidate(self):
        """測試前沿候選題材 as_of 逾 30 天時標記為 stale=True 且 score=None"""
        old_time = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            con.execute("""
                INSERT INTO discovery_candidates (
                    ticker, market, name, theme_id, theme_name, total_score, recommendation, as_of
                ) VALUES
                ('2308.TW', 'TW', '台達電', 'green_energy', '綠能電網', 85.0, 'propose_add', ?)
            """, [old_time])

        res = compute_signal_alignment("2308.TW", db=self.db)
        disc_dim = res["dimensions"]["discovery"]
        self.assertTrue(disc_dim["stale"])
        self.assertIsNone(disc_dim["score"])
        self.assertEqual(disc_dim["details"]["reason"], "discovery_data_stale_over_30d")

    def test_insufficient_sources(self):
        """Unknown ticker with no data should return insufficient_sources."""
        res = compute_signal_alignment("99999.TW", db=self.db)
        self.assertEqual(res["alignment_status"], "insufficient_sources")
        self.assertIsNone(res["score"])
        self.assertLess(res["active_sources_count"], 2)

    def test_api_endpoint(self):
        """GET /api/v1/predictions/alignment/{ticker} returns 200 with alignment JSON."""
        self._seed_sample_data()
        resp = self.client.get("/api/v1/predictions/alignment/2330.TW")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "2330.TW")
        self.assertEqual(data["alignment_status"], "bullish_alignment")
        self.assertIn("dimensions", data)
        self.assertIn("model", data["dimensions"])


if __name__ == "__main__":
    unittest.main()
