"""
test/test_typography_rwd.py - Unit test asserting typography floors and mobile RWD standards.
Ensures CJK readability floor (>=12px, font-medium/semibold) and responsive scale harmony.
"""

import unittest
import re
from pathlib import Path


class TestTypographyAndRWD(unittest.TestCase):
    def setUp(self):
        self.template_path = Path("api/templates/index.html")
        self.assertTrue(self.template_path.exists(), "api/templates/index.html not found")
        self.html = self.template_path.read_text(encoding="utf-8")

    def test_no_sub_12px_arbitrary_classes(self):
        """
        Verify that no arbitrary sub-12px Tailwind font classes (text-[10px], text-[11px]) exist.
        CJK characters (繁體中文) require at least 12px to prevent stroke clumping and illegibility.
        """
        sub_12px_matches = re.findall(r'text-\[(?:[0-9]|10|11)px\]', self.html)
        self.assertEqual(
            len(sub_12px_matches), 
            0, 
            f"Found {len(sub_12px_matches)} sub-12px font classes: {set(sub_12px_matches)}. "
            "CJK text must have a minimum floor of 12px (text-xs)."
        )

    def test_stock_card_price_labels_readability(self):
        """
        Verify that stock card price labels ('當前現價', '預測目標價') do not use text-[10px] or text-[11px].
        They should be at least text-xs with font-semibold or font-bold.
        """
        self.assertNotIn('text-[11px]" :class="isDark ? \'text-dark-muted\' : \'text-claude-muted\'">當前現價', self.html)
        self.assertNotIn('text-[11px]" :class="isDark ? \'text-sky-400\' : \'text-claude-terracotta\'">預測目標價', self.html)

    def test_stock_card_tags_readability(self):
        """
        Verify that strategy tags (parseTags) do not use text-[11px].
        """
        tag_match = re.search(r'v-for="tag in parseTags\(item\.tags\)".*?\sclass="([^"]+)"', self.html, re.DOTALL)
        if tag_match:
            class_str = tag_match.group(1)
            self.assertNotIn("text-[10px]", class_str)
            self.assertNotIn("text-[11px]", class_str)
            self.assertIn("text-xs", class_str)

    def test_heading_hierarchy_no_skipped_levels(self):
        """
        Verify heading levels do not skip directly from h1 to h3 without an h2.
        (Caught by Impeccable detect).
        """
        h1_pos = self.html.find("<h1")
        h2_pos = self.html.find("<h2")
        h3_pos = self.html.find("<h3")
        self.assertTrue(h1_pos != -1, "Missing h1")
        self.assertTrue(h2_pos != -1, "Missing h2 - heading level skipped from h1 to h3")
        self.assertTrue(h1_pos < h2_pos, "h1 must appear before h2")

    def test_cjk_typography_and_viewport_optimization(self):
        """
        Verify viewport meta tag and text-size-adjust are in place.
        """
        self.assertIn('name="viewport"', self.html)
        self.assertTrue(
            "-webkit-text-size-adjust" in self.html or "text-size-adjust" in self.html,
            "CSS should include text-size-adjust to prevent mobile browser unwanted text inflation"
        )

    def test_viewport_fit_cover_and_overflow_prevention(self):
        """
        Verify viewport-fit=cover and horizontal overflow protection on <body>.
        """
        self.assertIn('viewport-fit=cover', self.html, "meta viewport should include viewport-fit=cover for notch/home-indicator safety")
        self.assertIn('overflow-x-hidden', self.html, "body should include overflow-x-hidden to prevent horizontal shake/scroll")

    def test_mobile_input_anti_zoom_rule(self):
        """
        Verify 16px minimum font-size rule for mobile inputs/selects to prevent iOS Safari auto-zoom.
        """
        self.assertIn('@media (max-width: 640px)', self.html)
        self.assertIn('font-size: 16px !important;', self.html)

    def test_fixed_bottom_nav_bar(self):
        """
        Verify that header.html contains the fixed bottom navigation bar with safe-area padding for mobile.
        """
        header_path = Path("api/templates/partials/header.html")
        self.assertTrue(header_path.exists(), "api/templates/partials/header.html not found")
        header_html = header_path.read_text(encoding="utf-8")
        self.assertIn("fixed bottom-0", header_html)
        self.assertIn("md:hidden", header_html)
        self.assertIn("safe-area-inset-bottom", header_html)

    def test_mobile_card_views_for_tables(self):
        """
        Verify mobile card view alternatives for wide tables in tab_history.html and tab_institutional.html.
        """
        history_path = Path("api/templates/partials/tab_history.html")
        self.assertTrue(history_path.exists(), "tab_history.html not found")
        history_html = history_path.read_text(encoding="utf-8")
        self.assertIn("dailyHistoryRows", history_html)
        self.assertIn("block md:hidden", history_html)
        self.assertIn("hidden md:block overflow-x-auto", history_html)

        inst_path = Path("api/templates/partials/tab_institutional.html")
        self.assertTrue(inst_path.exists(), "tab_institutional.html not found")
        inst_html = inst_path.read_text(encoding="utf-8")
        self.assertIn("institutionalFlows", inst_html)
        self.assertIn("block md:hidden", inst_html)
        self.assertIn("hidden md:block overflow-x-auto", inst_html)

    def test_company_hover_touch_guard(self):
        """
        Verify that showCompanyHover in app.js checks for fine hover pointer to avoid touch occlusion.
        """
        app_js_path = Path("api/static/js/app.js")
        self.assertTrue(app_js_path.exists(), "api/static/js/app.js not found")
        app_js = app_js_path.read_text(encoding="utf-8")
        self.assertIn("(hover: hover) and (pointer: fine)", app_js)
        self.assertIn("Math.min(window.innerWidth - 24, 384)", app_js)

    def test_2md_saffron_palette_and_wcag_contrast_rules(self):
        """
        Verify that:
        1. header.html no longer contains the 888 StockQuant text next to the logo badge.
        2. index.html defines the 2md.aiurl.tw color tokens (#FFB010, #17324D, #FFF9E8, #2B251A, #FFF0CC) and WCAG AA/AAA light-mode contrast overrides.
        3. No bg-amber-500 text-white remains in any template (header.html or partials/*.html), ensuring WCAG AAA dark-ink-on-saffron contrast.
        """
        header_path = Path("api/templates/partials/header.html")
        self.assertTrue(header_path.exists(), "header.html not found")
        header_html = header_path.read_text(encoding="utf-8")

        # 1. header.html removes the 888 StockQuant text next to the logo badge
        self.assertNotIn('888&nbsp;<span class="text-amber-500">StockQuant</span>', header_html)
        self.assertNotIn('888 StockQuant</span>', header_html)

        # 2. index.html defines 2md.aiurl.tw color tokens and WCAG contrast rules
        for token in ['#FFB010', '#17324D', '#FFF9E8', '#2B251A', '#FFF0CC']:
            self.assertIn(token, self.html, f"Expected 2md color token {token} in index.html")

        self.assertIn("html:not(.dark) .bg-white", self.html)
        self.assertIn("html:not(.dark) .bg-amber-500", self.html)
        self.assertIn("#065F46", self.html)  # Bullish WCAG AAA
        self.assertIn("#991B1B", self.html)  # Bearish WCAG AAA

        # 3. No bg-amber-500 text-white remains in any template
        templates_dir = Path("api/templates")
        violating_files = []
        for html_file in templates_dir.glob("**/*.html"):
            content = html_file.read_text(encoding="utf-8")
            if "bg-amber-500 text-white" in content:
                violating_files.append(str(html_file))
        self.assertEqual(violating_files, [], f"Found bg-amber-500 text-white in: {violating_files}")


if __name__ == "__main__":
    unittest.main()
