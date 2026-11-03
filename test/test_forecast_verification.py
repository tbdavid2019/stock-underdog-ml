import datetime as dt
import tempfile
import unittest
from pathlib import Path

from data.forecast_store import ForecastStore
from backtest.verification import resolve_target_session, session_close, verify_forecast
from scripts.verify_forecasts import classify_legacy


class ForecastStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ForecastStore(Path(self.tmp.name) / "forecast.duckdb")

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_forecast_retry_and_revision_are_immutable(self):
        self.store.record_run({"run_id": "r1", "model": "m", "model_revision": "a"})
        self.store.record_forecast({"forecast_id": "f1", "run_id": "r1", "ticker": "2330.TW", "market": "TW", "model_revision": "a", "issued_at": "2026-01-02T06:00:00Z", "last_complete_bar": "2026-01-01", "target_date": "2026-01-02", "horizon": 1, "prediction": 100, "metadata": {"x": 1}})
        self.store.record_forecast({"forecast_id": "f1", "run_id": "r1", "ticker": "2330.TW", "market": "TW", "model_revision": "a", "issued_at": "2026-01-02T06:00:00Z", "last_complete_bar": "2026-01-01", "target_date": "2026-01-02", "horizon": 1, "prediction": 100, "metadata": {"x": 1}})
        with self.assertRaises(ValueError):
            self.store.record_forecast({"forecast_id": "f1", "run_id": "r1", "ticker": "2330.TW", "market": "TW", "model_revision": "a", "issued_at": "2026-01-02T06:00:00Z", "last_complete_bar": "2026-01-01", "target_date": "2026-01-02", "horizon": 1, "prediction": 101})
        self.store.record_forecast({"forecast_id": "f2", "run_id": "r1", "ticker": "2330.TW", "market": "TW", "model_revision": "b", "issued_at": "2026-01-02T06:00:00Z", "last_complete_bar": "2026-01-01", "target_date": "2026-01-02", "horizon": 1, "prediction": 101})
        self.assertEqual(len(self.store.list_forecasts()), 2)

    def test_strict_json_and_actual_revision_invalidates_snapshot(self):
        with self.assertRaises(ValueError):
            self.store.record_run({"run_id": "bad", "metadata": {"x": float("nan")}})
        self.store.record_run({"run_id": "r", "model": "m"})
        self.store.record_forecast({"forecast_id": "f", "run_id": "r", "ticker": "AAPL", "market": "US", "model_revision": "a", "issued_at": "2026-01-02T06:00:00Z", "last_complete_bar": "2026-01-01", "target_date": "2026-01-02", "horizon": 1, "prediction": 100})
        self.store.record_snapshot({"snapshot_id": "s", "payload": {"rank": 1}})
        self.store.settle_forecast("f", {"actual": 101, "source": "fixture", "price_basis": "close", "source_revision": "v1"})
        self.store.settle_forecast("f", {"actual": 102, "source": "fixture", "price_basis": "close", "source_revision": "v2"})
        self.assertEqual(len(self.store.list_actuals("f")), 2)
        self.assertEqual(self.store.list_invalidated_snapshots()[0]["snapshot_id"], "s")

    def test_target_session_and_exact_date_settlement(self):
        self.assertEqual(str(resolve_target_session("2026-11-25", "US", 1)), "2026-11-27")
        self.assertEqual(str(resolve_target_session("2026-03-06", "JP", 1)), "2026-03-09")
        self.store.record_run({"run_id": "r", "model": "m"})
        self.store.record_forecast({"forecast_id": "f", "run_id": "r", "ticker": "AAPL", "market": "US", "model_revision": "a", "issued_at": "2026-01-02T06:00:00Z", "last_complete_bar": "2026-01-02", "target_date": "2026-01-05", "horizon": 1, "prediction": 100})
        result = verify_forecast(self.store, "f", [{"date": "2026-01-06", "close": 110}], now=dt.datetime(2026, 1, 6, tzinfo=dt.timezone.utc))
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["reason"], "target_price_missing")

    def test_early_close_dst_and_legacy_dryrun(self):
        self.assertEqual(session_close(dt.date(2026, 11, 27), "US").hour, 13)
        self.assertEqual(session_close(dt.date(2026, 3, 9), "US").utcoffset(), dt.timedelta(hours=-4))
        report = classify_legacy([{"ticker": "OLD"}, {"ticker": "REPLAY", "checkpoint_revision": "x", "data_cutoff": "2026-01-01", "issued_at": "2026-01-02T00:00:00Z"}, {"ticker": "NEW", "checkpoint_revision": "x", "data_cutoff": "2026-01-01", "issued_at": "2026-01-02T00:00:00Z", "evidence_type": "prospective"}])
        self.assertEqual(report["counts"], {"legacy_unverified": 1, "replay": 1, "prospective": 1})

    def test_outbox_is_idempotent_and_readonly_store_does_not_create(self):
        self.store.enqueue_supabase("e1", {"kind": "forecast"})
        self.store.enqueue_supabase("e1", {"kind": "forecast"})
        self.assertEqual(len(self.store.list_outbox()), 1)
        ro = ForecastStore(Path(self.tmp.name) / "forecast.duckdb", read_only=True)
        self.assertEqual(len(ro.list_forecasts()), 0)

    def test_early_close_pending_and_incomplete_bar_do_not_settle(self):
        target = dt.date(2026, 11, 27)
        self.store.record_run({"run_id": "r", "model": "m"})
        self.store.record_forecast({"forecast_id": "fc", "run_id": "r", "ticker": "AAPL", "market": "US", "model_revision": "a", "issued_at": "2026-11-27T15:00:00Z", "last_complete_bar": "2026-11-25", "target_date": target.isoformat(), "horizon": 1, "prediction": 100})
        now_before_close = dt.datetime(2026, 11, 27, 17, 30, tzinfo=dt.timezone.utc)
        result = verify_forecast(self.store, "fc", [], now=now_before_close, dry_run=True)
        self.assertEqual(result["reason"], "target_session_not_closed")
        now_after_close = dt.datetime(2026, 11, 27, 19, 0, tzinfo=dt.timezone.utc)
        result = verify_forecast(self.store, "fc", [{"date": target.isoformat(), "close": 100, "complete": False}], now=now_after_close, dry_run=True)
        self.assertEqual(result["reason"], "target_bar_incomplete")


if __name__ == "__main__":
    unittest.main()
