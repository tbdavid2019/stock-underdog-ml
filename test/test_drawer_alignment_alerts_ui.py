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

    def test_calendar_modal_does_not_leak_drawer_subtabs(self):
        """Regression test: Full Earnings Call Calendar Modal must not contain stock drawer alignment/alerts subtabs."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        cal_marker = "6. FULL EARNINGS CALL CALENDAR MODAL"
        self.assertIn(cal_marker, html)
        calendar_section = html.split(cal_marker, 1)[1]
        self.assertNotIn("drawerActiveSubTab", calendar_section)
        self.assertNotIn("createDrawerAlert", calendar_section)

    def test_all_three_drawer_layouts_render_alignment_and_alerts_as_top_level_subtabs(self):
        """Verify Mobile Card, Desktop Table, and Desktop Card drawers all expose alignment & alerts outside tender."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        mobile_block = html.split("<!-- Mobile Drawer Expansion -->", 1)[1].split("<!-- 4B. Desktop", 1)[0]
        table_block = html.split("<!-- Expandable Company Details Drawer Row -->", 1)[1].split("<!-- 5. VIEW MODE", 1)[0]
        card_block = html.split("<!-- Card View Expandable Drawer -->", 1)[1].split("6. FULL EARNINGS CALL CALENDAR MODAL", 1)[0]

        for name, block in [("mobile", mobile_block), ("table", table_block), ("card", card_block)]:
            self.assertIn("v-if=\"drawerActiveSubTab === 'alignment'\"", block, f"{name} drawer missing alignment panel")
            self.assertIn("v-if=\"drawerActiveSubTab === 'alerts'\"", block, f"{name} drawer missing alerts panel")
            # Ensure alignment/alerts are NOT nested inside tender's top_clients/top_rivals
            tender_tail = block.split("v-if=\"drawerActiveSubTab === 'tender'\"", 1)[1].split("v-if=\"drawerActiveSubTab === 'alignment'\"", 1)[0]
            self.assertNotIn("drawerSignalAlignment", tender_tail, f"{name} drawer has alignment leaked inside tender")


if __name__ == "__main__":
    unittest.main()
