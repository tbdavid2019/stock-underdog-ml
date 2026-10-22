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

    def test_model_scorecard_structure_and_labels(self):
        """
        Verify that modelScorecard includes headToHeadCount, hitCount, lowHitWarning,
        and that tab_history.html displays '天中猜對' and '天同場比準度' as well as
        '🏆 該股表現較佳' without 'bg-amber-500 text-white'.
        """
        app_js_path = Path("api/static/js/app.js")
        self.assertTrue(app_js_path.exists(), "api/static/js/app.js not found")
        app_js = app_js_path.read_text(encoding="utf-8")

        self.assertIn("headToHeadCount", app_js)
        self.assertIn("hitCount", app_js)
        self.assertIn("lowHitWarning", app_js)
        self.assertIn("tfHitRate >= 50.0", app_js)
        self.assertIn("tmHitRate >= 50.0", app_js)

        history_path = Path("api/templates/partials/tab_history.html")
        self.assertTrue(history_path.exists(), "tab_history.html not found")
        history_html = history_path.read_text(encoding="utf-8")

        self.assertIn("天中猜對", history_html)
        self.assertIn("天同場比準度", history_html)
        self.assertIn("🏆 該股表現較佳", history_html)
        self.assertIn("lowHitWarning", history_html)

    def test_model_scorecard_js_logic_floor_and_h2h(self):
        """
        Execute modelScorecard logic via node to verify:
        1. 50% directional hit rate floor prevents awarding leader badge when < 50%.
        2. Exact counts for hitCount, headToHeadCount, tfWins, tmWins, ties.
        3. Clear overallWinner string formatting.
        """
        import json
        import subprocess

        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        idx = app_js.find("modelScorecard() {")
        self.assertNotEqual(idx, -1)
        idx2 = app_js.find("methods: {", idx)
        self.assertNotEqual(idx2, -1)
        fn_body = app_js[idx + len("modelScorecard() {"):idx2].rsplit("}", 2)[0]

        node_script = f"""
        const modelScorecardFn = new Function({json.dumps(fn_body)});

        // Scenario 1: Both hit rates < 50% (3 days verified, each hits 1)
        const ctx1 = {{
            dailyHistoryRows: [
                {{ nextActualPrice: 100, timesfmErrorPct: 2.5, timesfmHit: true, timestralErrorPct: 3.0, timestralHit: false, winner: 'TimesFM 勝' }},
                {{ nextActualPrice: 102, timesfmErrorPct: 4.0, timesfmHit: false, timestralErrorPct: 1.5, timestralHit: true, winner: 'Timestral 勝' }},
                {{ nextActualPrice: 105, timesfmErrorPct: 3.5, timesfmHit: false, timestralErrorPct: 4.0, timestralHit: false, winner: 'TimesFM 勝' }},
            ]
        }};
        const res1 = modelScorecardFn.call(ctx1);

        // Scenario 2: Timestral >= 50% (2/3), TimesFM < 50% (1/3)
        const ctx2 = {{
            dailyHistoryRows: [
                {{ nextActualPrice: 100, timesfmErrorPct: 2.5, timesfmHit: true, timestralErrorPct: 3.0, timestralHit: false, winner: 'TimesFM 勝' }},
                {{ nextActualPrice: 102, timesfmErrorPct: 4.0, timesfmHit: false, timestralErrorPct: 1.5, timestralHit: true, winner: 'Timestral 勝' }},
                {{ nextActualPrice: 105, timesfmErrorPct: 3.5, timesfmHit: false, timestralErrorPct: 2.0, timestralHit: true, winner: 'Timestral 勝' }},
            ]
        }};
        const res2 = modelScorecardFn.call(ctx2);

        // Scenario 3: Head-to-head metrics check (including ties)
        const ctx3 = {{
            dailyHistoryRows: [
                {{ nextActualPrice: 100, timesfmErrorPct: 2.0, timesfmHit: true, timestralErrorPct: 2.0, timestralHit: true, winner: '平手' }},
                {{ nextActualPrice: 102, timesfmErrorPct: 1.0, timesfmHit: true, timestralErrorPct: 3.0, timestralHit: true, winner: 'TimesFM 勝' }},
            ]
        }};
        const res3 = modelScorecardFn.call(ctx3);

        console.log(JSON.stringify({{ res1, res2, res3 }}));
        """

        result = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Scenario 1 assertions: neither model can be leader if hit rate < 50%
        res1 = data["res1"]
        self.assertFalse(res1["timesfm"]["isLeader"])
        self.assertFalse(res1["timestral"]["isLeader"])
        self.assertTrue(res1["timesfm"]["lowHitWarning"])
        self.assertTrue(res1["timestral"]["lowHitWarning"])
        self.assertEqual(res1["timesfm"]["hitCount"], 1)
        self.assertEqual(res1["timesfm"]["count"], 3)
        self.assertEqual(res1["headToHeadCount"], 3)
        self.assertIn("⚠️ 兩模型於該股方向勝率皆低於 50%", res1["overallWinner"])

        # Scenario 2 assertions: Timestral is eligible and wins
        res2 = data["res2"]
        self.assertTrue(res2["timestral"]["isLeader"])
        self.assertFalse(res2["timesfm"]["isLeader"])
        self.assertFalse(res2["timestral"]["lowHitWarning"])
        self.assertTrue(res2["timesfm"]["lowHitWarning"])
        self.assertEqual(res2["timestral"]["hitCount"], 2)
        self.assertEqual(res2["timestral"]["count"], 3)
        self.assertIn("🏆 該股表現較佳：Timestral", res2["overallWinner"])

        # Scenario 3 assertions: ties and headToHeadCount
        res3 = data["res3"]
        self.assertEqual(res3["headToHeadCount"], 2)
        self.assertEqual(res3["ties"], 1)
        self.assertEqual(res3["timesfm"]["wins"], 1)
        self.assertEqual(res3["timestral"]["wins"], 0)


if __name__ == "__main__":
    unittest.main()
