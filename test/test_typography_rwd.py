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
        CJK characters (繁體中文) require at least 12px/13px to prevent stroke clumping and illegibility.
        """
        # Check index.html and all partials
        all_templates = list(Path("api/templates").rglob("*.html"))
        sub_12px_found = {}
        for tmpl in all_templates:
            if tmpl.name == "header.html":
                continue # Header logo micro-badge for English STOCK/QUANT
            content = tmpl.read_text(encoding="utf-8")
            matches = re.findall(r'text-\[(?:[0-9]|10|11)px\]', content)
            if matches:
                sub_12px_found[tmpl.name] = set(matches)
        self.assertEqual(
            len(sub_12px_found), 
            0, 
            f"Found sub-12px font classes in templates: {sub_12px_found}. "
            "CJK text must have a minimum floor of 13px (text-xs) or 14px (text-sm)."
        )

    def test_stock_card_price_labels_readability(self):
        """
        Verify that stock card price labels ('現價', '目標價 / 均線') in tab_dashboard.html
        do not use sub-12px classes and are at least text-xs (13px) or text-sm.
        """
        dash_html = Path("api/templates/partials/tab_dashboard.html").read_text(encoding="utf-8")
        self.assertNotIn('text-[11px]" :class="isDark ? \'text-slate-400\' : \'text-slate-500\'">現價', dash_html)
        self.assertNotIn('text-[10px]" :class="isDark ? \'text-slate-400\' : \'text-slate-500\'">現價', dash_html)
        self.assertIn('>現價</div>', dash_html)
        self.assertIn('>目標價 / 均線</div>', dash_html)

    def test_stock_card_tags_readability(self):
        """
        Verify that strategy tags (parseTags) in tab_dashboard.html do not use sub-12px classes
        and have a minimum readability floor of text-[13px] or text-sm.
        """
        dash_html = Path("api/templates/partials/tab_dashboard.html").read_text(encoding="utf-8")
        tag_matches = re.findall(r'v-for="tag in parseTags\(item\.tags\)[^"]*".*?\sclass="([^"]+)"', dash_html, re.DOTALL)
        self.assertGreaterEqual(len(tag_matches), 1, "Must find at least one parseTags badge in tab_dashboard.html")
        for class_str in tag_matches:
            self.assertNotIn("text-[10px]", class_str)
            self.assertNotIn("text-[11px]", class_str)
            self.assertTrue("text-[13px]" in class_str or "text-sm" in class_str or "text-xs" in class_str)

    def test_parse_tags_js_deduplication_and_filtering(self):
        """
        Verify app.js parseTags logic: deduplicates duplicate badges and discards fake consensus keywords.
        """
        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        # 1. Deduplication with Set
        self.assertIn("const seen = new Set();", app_js)
        self.assertIn("if (seen.has(cleanLabel)) continue;", app_js)
        # 2. Rejection of fake consensus terms
        self.assertIn("if (/三模型|三大模型|雙大模型|雙ML|雙時序|雙模型|雙AI|雙維|四重共振|三重共振|多維共振|雙重共振|多模型共振|3重符合|2重符合|全共振|共振/.test(cleanLabel)) {", app_js)
        # 3. Semantic directional classes in getTagClass
        self.assertIn("case 'ai_bullish':", app_js)
        self.assertIn("case 'ai_bearish':", app_js)
        self.assertIn("case 'ai_strong':", app_js)
        self.assertIn("case 'fundamental':", app_js)

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
        self.assertIn("hitRate >= 50.0", app_js)

        history_path = Path("api/templates/partials/tab_history.html")
        self.assertTrue(history_path.exists(), "tab_history.html not found")
        history_html = history_path.read_text(encoding="utf-8")

        self.assertIn("天中猜對", history_html)
        self.assertIn("天同場比準度", history_html)
        self.assertIn("該股表現較佳", history_html)
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
        self.assertIn("方向勝率皆低於 50%", res1["overallWinner"])

        # Scenario 2 assertions: Timestral is eligible and wins
        res2 = data["res2"]
        self.assertTrue(res2["timestral"]["isLeader"])
        self.assertFalse(res2["timesfm"]["isLeader"])
        self.assertFalse(res2["timestral"]["lowHitWarning"])
        self.assertTrue(res2["timesfm"]["lowHitWarning"])
        self.assertEqual(res2["timestral"]["hitCount"], 2)
        self.assertEqual(res2["timestral"]["count"], 3)
        self.assertIn("該股表現較佳：Timestral", res2["overallWinner"])

        # Scenario 3 assertions: ties and headToHeadCount
        res3 = data["res3"]
        self.assertEqual(res3["headToHeadCount"], 2)
        self.assertEqual(res3["ties"], 1)
        self.assertEqual(res3["timesfm"]["wins"], 1)
        self.assertEqual(res3["timestral"]["wins"], 0)

    def test_cjk_typography_minimum_floor_on_buttons_and_tables(self):
        """
        Verify that action buttons (查看時序歷史軌跡, Yahoo, Goodinfo, 股權關係網絡圖)
        and Bloomberg quant table use text-sm (14px) instead of text-xs (12px).
        """
        dash_path = Path("api/templates/partials/tab_dashboard.html")
        self.assertTrue(dash_path.exists())
        dash_html = dash_path.read_text(encoding="utf-8")

        # 1. Table base font must be text-sm
        self.assertIn('<table class="w-full text-left text-sm border-collapse whitespace-nowrap">', dash_html)

        # 2. History tracking button must be text-sm font-bold
        history_btn_match = re.search(r'<button[^>]+class="[^"]*text-sm[^"]*font-bold[^"]*"[^>]*>.*?查看時序歷史軌跡', dash_html, re.DOTALL)
        self.assertIsNotNone(history_btn_match, "查看時序歷史軌跡 button must be at least text-sm font-bold")

        # 3. Yahoo and Goodinfo links must be text-sm font-bold
        yahoo_match = re.search(r'<a[^>]+class="[^"]*text-sm[^"]*font-bold[^"]*"[^>]*>.*?Yahoo 奇摩股市', dash_html, re.DOTALL)
        self.assertIsNotNone(yahoo_match, "Yahoo link must be at least text-sm font-bold")

        goodinfo_match = re.search(r'<a[^>]+class="[^"]*text-sm[^"]*font-bold[^"]*"[^>]*>.*?Goodinfo 台灣股市', dash_html, re.DOTALL)
        self.assertIsNotNone(goodinfo_match, "Goodinfo link must be at least text-sm font-bold")

        # 4. Drawer subtabs must be text-sm font-bold
        subtab_match = re.search(r'<button[^>]+class="[^"]*text-sm[^"]*font-bold[^"]*"[^>]*>.*?基本面與新聞', dash_html, re.DOTALL)
        self.assertIsNotNone(subtab_match, "基本面與新聞 subtab must be text-sm font-bold")

        # 5. No '三模型看漲' in app.js
        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        self.assertNotIn("'三模型看漲'", app_js, "app.js must not output '三模型看漲'")

    def test_multi_market_chips_structure_and_touch_target(self):
        """
        Verify multi-market inline multi-select chips in tab_dashboard.html meet touch targets (>=44px),
        support toggleMarket, selectAll, clear, and display active market tags in applied filters.
        """
        dash_html = Path("api/templates/partials/tab_dashboard.html").read_text(encoding="utf-8")
        self.assertIn('id="market-multiselect-chips"', dash_html)
        self.assertIn('min-h-[44px]', dash_html)
        self.assertIn("toggleMarket('ALL')", dash_html)
        self.assertIn("toggleMarket(opt.id)", dash_html)
        self.assertIn("isMarketSelected('ALL')", dash_html)
        self.assertIn("isMarketSelected(opt.id)", dash_html)
        self.assertIn("availableMarketOptions", dash_html)
        self.assertIn("selectAllMarkets()", dash_html)
        self.assertIn("clearSelectedMarkets()", dash_html)
        self.assertIn('min-h-[36px] sm:min-h-[40px]', dash_html)
        self.assertIn('市場: {{ getMarketDisplayLabel(mId) }}', dash_html)
        # Dropdown popover is removed from top header for a cleaner header
        self.assertNotIn('id="market-multiselect-dropdown"', dash_html)

        # Verify JS implementation difference between selectAllMarkets and clearSelectedMarkets
        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("this.selectedMarkets = this.availableMarketOptions.map(o => o.id);", app_js)
        self.assertIn("clearSelectedMarkets() {\n            this.selectedMarkets = [];", app_js)

    def test_multi_market_dropdown_structure_and_touch_target(self):
        """Backward compatibility alias for test_multi_market_chips_structure_and_touch_target"""
        return self.test_multi_market_chips_structure_and_touch_target()

    def test_table_and_chips_ux_enhancements(self):
        """
        Verify the table column narrowing, full hover tooltips, wrapping tags,
        and 5-day institutional chips handling for TW vs overseas markets.
        """
        dash_html = Path("api/templates/partials/tab_dashboard.html").read_text(encoding="utf-8")

        # 1. 標的 / 名稱 column narrowing and full tooltip
        self.assertIn('w-36 sm:w-44', dash_html)
        self.assertIn('max-w-[150px] sm:max-w-[175px]', dash_html)
        self.assertIn(':title="formatTicker(item.ticker) + \' \' + (item.name || companyProfileCache[item.ticker]?.name || \'\')"', dash_html)

        # 2. PE / PB column narrowing
        self.assertIn('w-16 sm:w-20', dash_html)
        self.assertIn('tabular-nums font-mono text-xs sm:text-sm', dash_html)

        # 3. Strategy tags wrapping without truncation
        self.assertIn('min-w-[160px] max-w-[280px]', dash_html)
        self.assertIn('flex flex-wrap items-center gap-1.5 py-0.5', dash_html)
        self.assertIn('v-for="tag in parseTags(item.tags)"', dash_html)

        # 4. 5-day institutional chips handling for TW vs overseas markets
        self.assertIn('isTaiwanStock(item)', dash_html)
        self.assertIn('— (海外市場不適用)', dash_html)
        self.assertIn(':title="\'外資/投信5日買賣超僅適用於台股市場（美股/日股無此數據）\'"', dash_html)

    def test_is_taiwan_stock_js_method(self):
        """
        Verify app.js implements isTaiwanStock method correctly distinguishing
        TW stocks from US/JP overseas stocks.
        """
        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        self.assertIn('isTaiwanStock(item)', app_js)
        self.assertIn("SP500", app_js)
        self.assertIn("日經", app_js)
        self.assertIn("美股", app_js)
        self.assertIn(".endsWith('.TW')", app_js)

    def test_podcast_no_gemini_or_local_rule_leaks(self):
        """
        Verify that internal LLM model names (Gemini 2.5 Flash) and internal slot labels (本地規則)
        never leak to user-facing UI in tab_podcast.html and decision services.
        """
        podcast_html = Path("api/templates/partials/tab_podcast.html").read_text(encoding="utf-8")
        self.assertNotIn("Gemini 2.5 Flash", podcast_html, "Internal model name Gemini 2.5 Flash must not leak to user UI")
        self.assertIn("replace(/\\s*\\(本地規則\\)/g, '')", podcast_html, "Cached (本地規則) must be filtered in tab_podcast.html")

        podcast_svc = Path("data/podcast_service.py").read_text(encoding="utf-8")
        self.assertNotIn("⭐ 5星重磅(本地規則)", podcast_svc)
        self.assertNotIn("⚡ 4星重要(本地規則)", podcast_svc)

        decision_svc = Path("data/decision_service.py").read_text(encoding="utf-8")
        self.assertNotIn("⭐ 5星重磅(本地規則)", decision_svc)
        self.assertNotIn("⚡ 4星重要(本地規則)", decision_svc)

        clef_code = Path("data/clef_client.py").read_text(encoding="utf-8")
        self.assertNotIn("(本地規則)", clef_code, "ClefDecisionClient tags must not contain '(本地規則)'")

    def test_duckdb_connection_different_configuration_fallback(self):
        """
        Verify DuckDBManager handles 'different configuration' exception gracefully,
        falling back between read_only=True and read_only=False seamlessly.
        """
        import os
        import tempfile
        import duckdb
        from data.duckdb_manager import DuckDBManager

        db_file = tempfile.mktemp(suffix=".duckdb")
        try:
            mgr = DuckDBManager(db_path=db_file)
            # 模擬背景排程開啟了 read_write 連線
            rw_conn = duckdb.connect(db_file, read_only=False)
            try:
                # 測試 _get_connection(read_only=True) 在遇到配置衝突時透明回退
                ro_conn = mgr._get_connection(read_only=True)
                self.assertIsNotNone(ro_conn)
                # 驗證連線可安全執行 SELECT 查詢
                res = ro_conn.execute("SELECT 1").fetchall()
                self.assertEqual(res, [(1,)])
                ro_conn.close()
            finally:
                rw_conn.close()
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
