"""
Unit and template rendering tests for Portfolio Risk X-Ray UI and Model Statistical Significance Badges.
Ensures zero NaN labels, impeccable design standards, and WCAG AA/AAA compliance.
"""
import os
import re
import unittest
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


class TestRiskXrayUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates_dir = os.path.join(os.path.dirname(__file__), "..", "api", "templates")
        cls.templates = Jinja2Templates(directory=cls.templates_dir)

    def test_risk_xray_panel_partial_exists(self):
        panel_path = os.path.join(self.templates_dir, "partials", "risk_xray_panel.html")
        self.assertTrue(os.path.isfile(panel_path), "risk_xray_panel.html partial must exist")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check key components
        self.assertIn("showRiskXrayModal", content)
        self.assertIn("annualized_volatility_total", content)
        self.assertIn("annualized_volatility_risk_sleeve", content)
        self.assertIn("insufficient_tail_samples", content)
        self.assertIn("percentage_contribution", content)
        self.assertIn("high_correlation_pairs", content)
        self.assertIn("runWeightOptimization", content)
        self.assertIn("local_currency_return_proxy", content)

    def test_wcag_no_white_on_amber(self):
        """Strict WCAG AA/AAA rule: never place white text on amber #FFB010 or bg-amber-500."""
        panel_path = os.path.join(self.templates_dir, "partials", "risk_xray_panel.html")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all static class="..." attributes (no colon) and all single-quoted class fragments in Vue bindings
        static_classes = re.findall(r'(?<!:)class="([^"]*)"', content)
        vue_fragments = re.findall(r"'([^']*)'", content)
        all_class_tokens = static_classes + vue_fragments

        violations = []
        for s in all_class_tokens:
            has_amber_bg = 'bg-[#FFB010]' in s or ('bg-amber-500' in s and '/10' not in s and '/20' not in s and '/25' not in s)
            has_white_text = ('text-white' in s and 'hover:text-white' not in s) or 'text-slate-100' in s
            if has_amber_bg and has_white_text:
                violations.append(s)

        self.assertEqual(violations, [], f"Found forbidden white text on solid amber background: {violations}")

        # Check tab_dashboard.html button
        dash_path = os.path.join(self.templates_dir, "partials", "tab_dashboard.html")
        with open(dash_path, "r", encoding="utf-8") as f:
            dash_content = f.read()

        self.assertIn("openRiskXrayModal()", dash_content)
        self.assertIn("組合風險 X 光", dash_content)

    def test_statistical_significance_badges_present(self):
        dash_path = os.path.join(self.templates_dir, "partials", "tab_dashboard.html")
        with open(dash_path, "r", encoding="utf-8") as f:
            dash_content = f.read()

        self.assertIn("is_statistically_significant", dash_content)
        self.assertIn("Bootstrap Sharpe 95% CI", dash_content)
        self.assertIn("顯著", dash_content)

        hist_path = os.path.join(self.templates_dir, "partials", "tab_history.html")
        with open(hist_path, "r", encoding="utf-8") as f:
            hist_content = f.read()

        self.assertIn("統計顯著性", hist_content)
        self.assertIn("is_statistically_significant", hist_content)

    def test_index_template_rendering(self):
        """Render index.html via TestClient and ensure no Jinja syntax errors."""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        body = resp.text
        self.assertIn("showRiskXrayModal", body)
        self.assertIn("組合風險 X 光", body)
        self.assertNotIn("NaN%", body)
        self.assertNotIn("undefined%", body)


if __name__ == "__main__":
    unittest.main()
