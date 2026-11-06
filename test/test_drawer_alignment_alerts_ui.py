"""
test/test_drawer_alignment_alerts_ui.py
Unit tests verifying the presence, accessibility, and UI structure of
the Signal Alignment Synthesis Card and Quant Price Alerts Controls in the Stock Drawer.
"""

import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestDrawerAlignmentAlertsUI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_drawer_contains_alignment_and_alerts_subtabs(self):
        """Verify GET / renders drawer subtabs for alignment and alerts."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Subtab buttons
        self.assertIn("drawerActiveSubTab = 'alignment'", html)
        self.assertIn("loadDrawerSignalAlignment", html)
        self.assertIn("訊號共振評估", html)

        self.assertIn("drawerActiveSubTab = 'alerts'", html)
        self.assertIn("loadDrawerAlerts", html)
        self.assertIn("量化價位告警", html)

    def test_signal_alignment_card_structure(self):
        """Verify signal alignment card includes consensus status, 4 dimensions, and no white-on-amber contrast violation."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Hero Consensus Badge
        self.assertIn("drawerSignalAlignment.alignment_status === 'bullish_alignment'", html)
        self.assertIn("drawerSignalAlignment.alignment_status === 'divergence_warning'", html)
        self.assertIn("drawerSignalAlignment.alignment_status === 'bearish_alignment'", html)

        # Contrast check: amber divergence warning badge must have dark text #2B251A, not white
        self.assertIn("bg-[#FFB010] text-[#2B251A]", html)

        # 4 dimensions
        self.assertIn("1. AI 時序模型預測", html)
        self.assertIn("2. 台股法人籌碼流向", html)
        self.assertIn("3. Podcast 智慧情報", html)
        self.assertIn("4. 前沿 AI 學術雷達", html)

    def test_quant_alerts_controls_structure(self):
        """Verify alert creation controls, inputs, and active alerts list exist."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Controls & Form
        self.assertIn("newAlertConditionType", html)
        self.assertIn("above_price", html)
        self.assertIn("below_p10", html)
        self.assertIn("above_p50", html)
        self.assertIn("newAlertTargetPrice", html)
        self.assertIn("createDrawerAlert", html)

        # Active Alerts List
        self.assertIn("drawerAlerts", html)
        self.assertIn("deleteDrawerAlert", html)


if __name__ == "__main__":
    unittest.main()
