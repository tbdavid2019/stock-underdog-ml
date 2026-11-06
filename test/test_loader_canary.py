"""
test/test_loader_canary.py - Tests for Upstream Canary Health Lane.
Validates:
1. Zero writable DuckDB connections during canary execution.
2. 30-minute cooldown reuses existing snapshot without network I/O.
3. Deterministic 3-state failure taxonomy (healthy, upstream_unreachable_or_ratelimited, schema_drift_parse_error).
4. Whitelist fields and 100% secret/IP redaction.
"""

import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from data.loader_canary import (
    run_canary_checks,
    sanitize_message,
    _execute_single_canary_probe,
)


class TestLoaderCanary(unittest.TestCase):
    TEST_CACHE = "test/scratch/test_canary_latest.json"

    def setUp(self):
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(self.TEST_CACHE):
            try:
                os.remove(self.TEST_CACHE)
            except Exception:
                pass

    def tearDown(self):
        if os.path.exists(self.TEST_CACHE):
            try:
                os.remove(self.TEST_CACHE)
            except Exception:
                pass

    def test_zero_duckdb_connections(self):
        # Assert duckdb.connect is never called
        with patch("duckdb.connect") as mock_duckdb:
            # Mock probes to return healthy without hitting real web
            mock_probe = lambda timeout: (200, [], None)
            with patch.dict("data.loader_canary.PROBE_REGISTRY", {"mock_src": mock_probe}):
                res = run_canary_checks(sources=["mock_src"], force=True, cache_path=self.TEST_CACHE)
                self.assertEqual(res["status"], "healthy")
                mock_duckdb.assert_not_called()

    def test_cooldown_reuses_recent_snapshot(self):
        # 1. First run creates snapshot
        mock_probe = MagicMock(return_value=(200, [], None))
        with patch.dict("data.loader_canary.PROBE_REGISTRY", {"mock_src": mock_probe}):
            res1 = run_canary_checks(sources=["mock_src"], force=True, cache_path=self.TEST_CACHE)
            self.assertFalse(res1["is_cached"])
            self.assertEqual(mock_probe.call_count, 1)

            # 2. Second run within cooldown interval (force=False) must reuse snapshot and NOT call probe
            res2 = run_canary_checks(sources=["mock_src"], force=False, cooldown_seconds=1800, cache_path=self.TEST_CACHE)
            self.assertTrue(res2["is_cached"])
            self.assertEqual(mock_probe.call_count, 1)  # Probe NOT called again!

            # 3. Third run with force=True bypasses cooldown and calls probe
            res3 = run_canary_checks(sources=["mock_src"], force=True, cache_path=self.TEST_CACHE)
            self.assertFalse(res3["is_cached"])
            self.assertEqual(mock_probe.call_count, 2)

    def test_three_state_taxonomy_classification(self):
        # 1. Healthy: 200, no missing fields, no error
        res_h = _execute_single_canary_probe("src_h", lambda t: (200, [], None), timeout=5.0)
        self.assertEqual(res_h["status"], "healthy")
        self.assertEqual(res_h["http_status"], 200)
        self.assertEqual(res_h["missing_fields"], [])

        # 2. Upstream unreachable or rate limited (HTTP 429)
        res_429 = _execute_single_canary_probe("src_429", lambda t: (429, [], "Too Many Requests"), timeout=5.0)
        self.assertEqual(res_429["status"], "upstream_unreachable_or_ratelimited")
        self.assertEqual(res_429["http_status"], 429)

        # 3. Upstream unreachable or rate limited (HTTP 503)
        res_503 = _execute_single_canary_probe("src_503", lambda t: (503, [], "Service Unavailable"), timeout=5.0)
        self.assertEqual(res_503["status"], "upstream_unreachable_or_ratelimited")

        # 4. Schema drift parse error: HTTP 200 but missing required fields
        res_drift = _execute_single_canary_probe("src_drift", lambda t: (200, ["Close", "Date"], None), timeout=5.0)
        self.assertEqual(res_drift["status"], "schema_drift_parse_error")
        self.assertEqual(res_drift["missing_fields"], ["Close", "Date"])

    def test_secret_token_and_ip_redaction(self):
        # Raw error message containing internal IP, API token and query parameters
        leaked_msg = (
            "Failed connecting to http://192.168.1.50:8000/api/v1/data?token=super_secret_token_123456&key=xyz: "
            "Internal proxy at 10.0.0.1 refused connection"
        )
        sanitized = sanitize_message(leaked_msg)
        self.assertNotIn("192.168.1.50", sanitized)
        self.assertNotIn("10.0.0.1", sanitized)
        self.assertNotIn("super_secret_token_123456", sanitized)
        self.assertIn("[INTERNAL_IP]", sanitized)
        self.assertIn("[REDACTED", sanitized)


if __name__ == "__main__":
    unittest.main()
