"""
test/test_api.py - FastAPI REST & DuckDB Endpoints Unit Tests
"""

import os
import unittest
from fastapi.testclient import TestClient
from api.main import app
from data.duckdb_manager import DuckDBManager
from core.version import APP_VERSION


class TestFastAPIService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 使用測試 DuckDB 資料庫
        cls.test_db_path = "test/scratch/test_api_quant.duckdb"
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

        cls.db = DuckDBManager(db_path=cls.test_db_path)
        
        import datetime
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # 寫入測試模擬數據
        sample_records = [
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 2400.0,
                "predicted_price": 2500.0,
                "potential": 4.17,
                "ma60": 2350.0,
                "pullback_type": "MA60 (+2.1%)",
                "pe": 28.0,
                "pb": 9.5,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "trust_net_5d": 500,
                "foreign_net_5d": 1200,
                "tags": "🏆三重共振 | 土洋合買 | 投信連買4天"
            },
            {
                "index_name": "台灣50",
                "model_name": "Timestral",
                "strategy_type": "Timestral",
                "ticker": "2454.TW",
                "current_price": 3900.0,
                "predicted_price": 3600.0,
                "potential": -7.69,
                "pe": 60.0,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "tags": "Timestral看跌"
            },
            {
                "index_name": "台灣50",
                "model_name": "Timestral",
                "strategy_type": "Timestral",
                "ticker": "2330.TW",
                "current_price": 2400.0,
                "predicted_price": 2520.0,
                "potential": 5.0,
                "timestral_predicted_price": 2520.0,
                "timestral_potential": 5.0,
                "timestral_p10": 2380.0,
                "timestral_p90": 2650.0,
                "dual_model_consensus": True,
                "pe": 28.0,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "tags": "Timestral看漲"
            },
            {
                "index_name": "台灣50",
                "model_name": "玄鐵重劍",
                "strategy_type": "玄鐵重劍",
                "ticker": "1216.TW",
                "current_price": 76.0,
                "ma60": 74.0,
                "pullback_type": "MA60 (+2.7%)",
                "pe": 18.5,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "tags": "玄鐵買點"
            }
        ]
        cls.db.save_predictions_batch(sample_records)
        
        from api.routes.predictions import get_duckdb as get_duckdb_pred
        from api.routes.stats import get_duckdb as get_duckdb_stats
        app.dependency_overrides[get_duckdb_pred] = lambda: cls.db
        app.dependency_overrides[get_duckdb_stats] = lambda: cls.db
        
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("duckdb_records", data)

    def test_latest_predictions(self):
        resp = self.client.get("/api/v1/predictions/latest?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["count"], 1)
        self.assertIn("batch_date", data)
        self.assertIn("is_stale", data)
        item = data["data"][0]
        self.assertIn("analysis_date", item)
        self.assertIn("data_as_of", item)
        self.assertIn("age_hours", item)

    def test_resonance_endpoint(self):
        resp = self.client.get("/api/v1/predictions/resonance")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_xuantie_endpoint(self):
        resp = self.client.get("/api/v1/predictions/xuantie")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_timestral_top_bullish_and_bearish(self):
        bull_resp = self.client.get("/api/v1/predictions/timestral/top-bullish?limit=5")
        self.assertEqual(bull_resp.status_code, 200)
        data = bull_resp.json()
        self.assertTrue(data["success"])

        bear_resp = self.client.get("/api/v1/predictions/timestral/top-bearish?limit=5")
        self.assertEqual(bear_resp.status_code, 200)
        data = bear_resp.json()
        self.assertTrue(data["success"])

    def test_timesfm_top_bullish_and_bearish(self):
        bull_resp = self.client.get("/api/v1/predictions/timesfm/top-bullish?limit=5")
        self.assertEqual(bull_resp.status_code, 200)
        
        bear_resp = self.client.get("/api/v1/predictions/timesfm/top-bearish?limit=5")
        self.assertEqual(bear_resp.status_code, 200)

    def test_moirai_top_bullish_and_bearish(self):
        bull_resp = self.client.get("/api/v1/predictions/moirai/top-bullish?limit=5")
        self.assertEqual(bull_resp.status_code, 200)
        
        bear_resp = self.client.get("/api/v1/predictions/moirai/top-bearish?limit=5")
        self.assertEqual(bear_resp.status_code, 200)

    def test_stats_summary(self):
        resp = self.client.get("/api/v1/stats/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("total_records", data)
        self.assertNotIn("db_path", data)

    def test_root_homepage(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("888 StockQuant", resp.text)

    def test_homepage_has_complete_social_and_search_metadata(self):
        html = self.client.get("/").text
        expected_fragments = [
            '<meta name="description"',
            '<meta property="og:description"',
            '<meta property="og:image"',
            '<meta property="og:url" content="https://stockdata.david888.com/"',
            '<meta name="twitter:card" content="summary_large_image"',
            '<meta name="twitter:description"',
            '<meta name="twitter:image"',
            '<link rel="canonical" href="https://stockdata.david888.com/"',
            '<link rel="apple-touch-icon"',
            '<link rel="icon" type="image/png" sizes="32x32"',
            '<link rel="icon" type="image/svg+xml"',
            '<link rel="manifest" href="/site.webmanifest"',
            '<script type="application/ld+json">',
            '<meta name="theme-color"',
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, html)

    def test_homepage_social_assets_are_served(self):
        assets = {
            "/favicon.svg": "image/svg+xml",
            "/favicon-32x32.png": "image/png",
            "/apple-touch-icon.png": "image/png",
            "/og-image.png": "image/png",
            "/site.webmanifest": "application/manifest+json",
        }
        for path, content_type in assets.items():
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, path)
            self.assertTrue(resp.headers["content-type"].startswith(content_type), path)
            self.assertGreater(len(resp.content), 0, path)

    def test_resolves_company_name_before_history_query(self):
        resp = self.client.get("/api/v1/predictions/resolve/特斯拉")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "TSLA")

    def test_unknown_history_query_does_not_return_records(self):
        resp = self.client.get("/api/v1/predictions/history/這不是股票")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("查無法辨識", resp.json()["detail"])

    def test_ai_plugin_manifest(self):
        resp = self.client.get("/.well-known/ai-plugin.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name_for_model"], "stock_quant_engine")
        self.assertIn("openapi", data["api"]["type"])

    def test_skill_endpoint(self):
        resp = self.client.get("/skill")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("name: stock-quant", resp.text)
        self.assertIn("888-stock-quant", resp.text)
        self.assertIn("check-update", resp.text)

    def test_npx_skill_distribution_and_update_reminders(self):
        """驗證 /llms.txt、/llms-full.txt、/skill、Agent Hub 前台與 README.md 皆同步宣導 npx 888-stock-quant 與使用前檢查更新提示"""
        from pathlib import Path

        for route in ("/llms.txt", "/llms-full.txt", "/skill"):
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("npx -y 888-stock-quant@latest", resp.text, f"{route} missing npx command")
            self.assertIn("check-update", resp.text, f"{route} missing pre-use update check reminder")

        index_resp = self.client.get("/")
        self.assertEqual(index_resp.status_code, 200)
        html = index_resp.text
        self.assertIn("npx -y 888-stock-quant@latest install", html)
        self.assertIn("npx -y 888-stock-quant@latest check-update", html)
        self.assertIn("32 大原生量化 MCP 函數", html)
        self.assertNotIn("17 大原生量化 MCP 函數", html)

        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("npx -y 888-stock-quant@latest install", readme)
        self.assertIn("check-update", readme)

        root_skill = Path("skills/stock-quant/SKILL.md").read_text(encoding="utf-8")
        pkg_skill = Path("packages/888-stock-quant/skills/stock-quant/SKILL.md").read_text(encoding="utf-8")
        self.assertEqual(root_skill, pkg_skill, "NPM bundled SKILL.md must stay synced with skills/stock-quant/SKILL.md")

        # Verify legacy fixed-count resonance wording/emojis are completely eradicated from public/LLM surfaces
        legacy_terms = ("四重共振", "三重共振", "雙模型共振", "雙ML共振", "👑", "🏆", "🔮")
        pkg_readme = Path("packages/888-stock-quant/README.md").read_text(encoding="utf-8")
        ai_plugin = self.client.get("/.well-known/ai-plugin.json").text
        surfaces = {
            "/llms.txt": self.client.get("/llms.txt").text,
            "/llms-full.txt": self.client.get("/llms-full.txt").text,
            "/skill": self.client.get("/skill").text,
            "/.well-known/ai-plugin.json": ai_plugin,
            "/?tab=agent_hub": html,
            "README.md": readme,
            "packages/888-stock-quant/README.md": pkg_readme,
        }
        for name, body in surfaces.items():
            for term in legacy_terms:
                self.assertNotIn(term, body, f"Legacy wording {term!r} must not appear in {name}")

    def test_webmcp_manifest(self):
        resp = self.client.get("/.well-known/mcp.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "stock-quant-engine")
        self.assertEqual(data["transport"], "sse")
        self.assertEqual(len(data["tools"]), 32)
        self.assertIn("screen_quant_stocks", data["tools"])
        self.assertIn("get_industry_chain", data["tools"])
        self.assertIn("get_discovery_signals", data["tools"])
        self.assertIn("get_model_performance_scorecard", data["tools"])
        self.assertIn("get_market_macro_regime", data["tools"])
        self.assertIn("get_fed_rate_monitor", data["tools"])
        self.assertIn("get_us_earnings_calendar", data["tools"])
        self.assertIn("get_commodities_summary", data["tools"])
        self.assertIn("resolve_stock_ticker", data["tools"])
        self.assertIn("get_polymarket_macro_sentiment", data["tools"])
        self.assertIn("get_portfolio_risk_xray", data["tools"])
        self.assertIn("get_canary_loader_health", data["tools"])
        self.assertIn("get_upcoming_earnings_calls", data["tools"])
        self.assertIn("get_ticker_earnings_calls", data["tools"])
        self.assertIn("get_company_network", data["tools"])
        self.assertIn("open_company_network", data["tools"])
        self.assertIn("get_company_tenders", data["tools"])
        self.assertIn("get_watchlist", data["tools"])
        self.assertIn("get_market_live_sessions", data["tools"])
        self.assertIn("get_signal_alignment", data["tools"])
        self.assertIn("get_moirai_top_predictions", data["tools"])

        # Test alias /mcp.json
        resp_alias = self.client.get("/mcp.json")
        self.assertEqual(resp_alias.status_code, 200)
        self.assertEqual(resp_alias.json()["name"], "stock-quant-engine")
        self.assertEqual(len(resp_alias.json()["tools"]), 32)

    def test_macro_investing_endpoints(self):
        from unittest.mock import patch
        with patch("data.investing_service.InvestingService.get_investing_macro_summary") as mock_summary, \
             patch("data.investing_service.InvestingService.get_fed_rate_monitor") as mock_fed, \
             patch("data.investing_service.InvestingService.get_earnings_calendar") as mock_earn, \
             patch("data.investing_service.InvestingService.get_economic_calendar") as mock_econ, \
             patch("data.investing_service.InvestingService.get_commodities_summary") as mock_comm:
            
            mock_summary.return_value = {
                "fed_rate": {"highest_prob_rate": "3.75 - 4.00"},
                "commodities": {"gold": {"name": "Gold"}},
                "earnings_calendar": [{"ticker": "COST"}],
                "economic_calendar": [{"event": "NFP"}],
                "catalyst_alerts": ["Alert 1"]
            }
            mock_fed.return_value = {"meeting_date": "Sep 16, 2026", "highest_prob_rate": "3.75 - 4.00"}
            mock_earn.return_value = [{"ticker": "COST"}]
            mock_econ.return_value = [{"event": "NFP"}]
            mock_comm.return_value = {"gold": {"name": "Gold"}}

            # Summary
            resp = self.client.get("/api/v1/macro/investing/summary")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["success"])
            self.assertEqual(resp.json()["data"]["fed_rate"]["highest_prob_rate"], "3.75 - 4.00")

            # Fed rate
            resp = self.client.get("/api/v1/macro/investing/fed-rate")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["success"])

            # Fed rate alias
            resp_alias_fed = self.client.get("/api/v1/macro/investing/fed-rate-monitor")
            self.assertEqual(resp_alias_fed.status_code, 200)
            self.assertTrue(resp_alias_fed.json()["success"])

            # Earnings calendar
            resp = self.client.get("/api/v1/macro/investing/earnings-calendar")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["success"])

            # Economic calendar
            resp = self.client.get("/api/v1/macro/investing/economic-calendar")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["success"])

            # Commodities
            resp = self.client.get("/api/v1/macro/investing/commodities")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["success"])

    def test_llms_txt_endpoints(self):
        resp1 = self.client.get("/llms.txt")
        self.assertEqual(resp1.status_code, 200)
        self.assertIn("888 Stock Quant", resp1.text)
        self.assertIn("/mcp/sse", resp1.text)
        self.assertIn("success=false", resp1.text)
        self.assertIn("0~100 百分比", resp1.text)

        resp2 = self.client.get("/llms-full.txt")
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("888 Stock Quant", resp2.text)
        self.assertIn("P_{50,h}", resp2.text)

    def test_root_head_method(self):
        resp = self.client.head("/")
        self.assertEqual(resp.status_code, 200)

    def test_cloudflare_webmcp_bridge(self):
        resp = self.client.get("/.webmcp/bridge.js")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Cloudflare & Chrome WebMCP Bridge", resp.text)
        self.assertIn("registerTool", resp.text)
        self.assertIn("get_timesfm_top_predictions", resp.text)
        self.assertIn("get_commodities_summary", resp.text)
        self.assertIn("resolve_stock_ticker", resp.text)
        self.assertIn("get_polymarket_macro_sentiment", resp.text)
        self.assertNotIn("category=${category || 'all'}", resp.text)

    def test_mcp_discovery_manifest(self):
        resp = self.client.get("/mcp")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data.get("tools", [])), 32)
        self.assertIn("screen_quant_stocks", data.get("tools", []))
        self.assertIn("get_industry_chain", data.get("tools", []))
        self.assertIn("get_timesfm_top_predictions", data.get("tools", []))
        self.assertIn("get_moirai_top_predictions", data.get("tools", []))
        self.assertIn("get_commodities_summary", data.get("tools", []))
        self.assertIn("resolve_stock_ticker", data.get("tools", []))
        self.assertIn("get_polymarket_macro_sentiment", data.get("tools", []))
        self.assertIn("get_watchlist", data.get("tools", []))
        self.assertIn("get_market_live_sessions", data.get("tools", []))
        self.assertIn("get_signal_alignment", data.get("tools", []))
        self.assertIn("get_portfolio_risk_xray", data.get("tools", []))
        self.assertIn("get_canary_loader_health", data.get("tools", []))

        resp_wk = self.client.get("/.well-known/mcp.json")
        self.assertEqual(resp_wk.status_code, 200)
        data_wk = resp_wk.json()
        self.assertEqual(data_wk["version"], APP_VERSION)
        self.assertEqual(len(data_wk.get("tools", [])), 32)
        self.assertIn("screen_quant_stocks", data_wk.get("tools", []))
        self.assertIn("get_industry_chain", data_wk.get("tools", []))
        self.assertIn("get_timesfm_top_predictions", data_wk.get("tools", []))
        self.assertIn("get_moirai_top_predictions", data_wk.get("tools", []))
        self.assertIn("get_commodities_summary", data_wk.get("tools", []))
        self.assertIn("resolve_stock_ticker", data_wk.get("tools", []))
        self.assertIn("get_polymarket_macro_sentiment", data_wk.get("tools", []))
        self.assertIn("get_watchlist", data_wk.get("tools", []))
        self.assertIn("get_market_live_sessions", data_wk.get("tools", []))
        self.assertIn("get_signal_alignment", data_wk.get("tools", []))
        self.assertIn("get_upcoming_earnings_calls", data_wk.get("tools", []))
        self.assertIn("get_ticker_earnings_calls", data_wk.get("tools", []))
        self.assertIn("get_company_network", data_wk.get("tools", []))
        self.assertIn("open_company_network", data_wk.get("tools", []))
        self.assertIn("get_company_tenders", data_wk.get("tools", []))

    def test_macro_polymarket_sentiment_endpoint(self):
        from unittest.mock import patch
        with patch("data.polymarket_service.PolymarketService.get_macro_sentiment") as mock_pm:
            mock_pm.return_value = {
                "success": True,
                "stale": False,
                "source": "doh_direct",
                "fed_real_money_odds": {
                    "pause": 5.0,
                    "cut_25bps": 88.0,
                    "cut_50bps": 7.0,
                    "hike_25bps": 0.0
                },
                "count": 1,
                "markets": [
                    {
                        "id": "pm_fed_1",
                        "question": "Fed rate cut in next FOMC?",
                        "category": "fed_rates",
                        "probability": 88.0,
                        "volume_24h": 450000.0,
                        "top_outcome": "Yes"
                    }
                ]
            }
            resp = self.client.get("/api/v1/macro/polymarket/sentiment")
            self.assertEqual(resp.status_code, 200)
            res_json = resp.json()
            self.assertTrue(res_json["success"])
            mock_pm.assert_called_once_with(force_refresh=False, category=None)
            self.assertEqual(res_json["data"]["source"], "doh_direct")
            self.assertEqual(res_json["data"]["fed_real_money_odds"]["cut_25bps"], 88.0)
            self.assertEqual(len(res_json["data"]["markets"]), 1)

    def test_macro_polymarket_failure_is_visible_to_api_clients(self):
        from unittest.mock import patch
        with patch("data.polymarket_service.PolymarketService.get_macro_sentiment") as mock_pm:
            mock_pm.return_value = {
                "success": False,
                "stale": False,
                "error": "upstream unavailable",
                "markets": [],
            }
            resp = self.client.get(
                "/api/v1/macro/polymarket/sentiment?force_refresh=true&category=fed_rates"
            )

            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.json()["success"])
            self.assertEqual(resp.json()["data"]["error"], "upstream unavailable")
            mock_pm.assert_called_once_with(force_refresh=True, category="fed_rates")

    def test_polymarket_frontend_uses_service_contract(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("fed_real_money_odds.cut_25bps || 0) }}", resp.text)
        self.assertNotIn("fed_real_money_odds.cut_25bps || 0) * 100", resp.text)
        app_js = self.client.get("/static/js/app.js")
        self.assertEqual(app_js.status_code, 200)
        self.assertIn("const params = new URLSearchParams()", app_js.text)

    def test_public_version_surfaces_are_current(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["version"], APP_VERSION)
        self.assertEqual(app.version, APP_VERSION)

    def test_earnings_call_endpoints(self):
        # 1. Upcoming calls for ALL, TW, US, JP
        resp_all = self.client.get("/api/v1/earnings-call/upcoming?market=ALL&limit=10")
        self.assertEqual(resp_all.status_code, 200)
        data_all = resp_all.json()
        self.assertTrue(data_all["success"])
        self.assertIsInstance(data_all["data"], list)

        resp_jp = self.client.get("/api/v1/earnings-call/upcoming?market=JP&limit=5")
        self.assertEqual(resp_jp.status_code, 200)
        data_jp = resp_jp.json()
        self.assertTrue(data_jp["success"])
        self.assertEqual(data_jp["market"], "JP")

        resp_kr = self.client.get("/api/v1/earnings-call/upcoming?market=KR&limit=5")
        self.assertEqual(resp_kr.status_code, 200)
        data_kr = resp_kr.json()
        self.assertTrue(data_kr["success"])
        self.assertEqual(data_kr["market"], "KR")

        # 2. Monthly Calendar
        resp_cal = self.client.get("/api/v1/earnings-call/calendar?market=ALL")
        self.assertEqual(resp_cal.status_code, 200)
        self.assertTrue(resp_cal.json()["success"])

        # 3. llms.txt & llms-full.txt
        resp_llms = self.client.get("/llms.txt")
        self.assertEqual(resp_llms.status_code, 200)
        self.assertIn("Earnings Calls", resp_llms.text)
        self.assertIn("/api/v1/earnings-call/upcoming", resp_llms.text)

        resp_llms_full = self.client.get("/llms-full.txt")
        self.assertEqual(resp_llms_full.status_code, 200)
        self.assertIn("earnings_calls", resp_llms_full.text)
        self.assertIn("### 4.4 Global Corporate Earnings Calls & Guidance", resp_llms_full.text)

    def test_homepage_icon_and_spinner_robustness(self):
        """驗證首頁與子分頁圖標與旋轉載入圈之強健性（防止 FOUT 造成英文單字旋轉與按鈕破版）"""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # 1. Google Font 必須使用 display=block 避免未載入前文字閃爍
        self.assertIn("family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block", html)

        # 2. 禁止使用字型連字旋轉（防止 progress_activity 17 字母在畫面中央斜向旋轉）
        self.assertNotIn('animate-spin">progress_activity</span>', html)
        self.assertNotIn('animate-spin text-amber-500">progress_activity</span>', html)

        # 3. 必須採用原生向量 SVG Spinner
        self.assertIn('<svg class="animate-spin', html)

        # 4. CSS 中必須包含防破版限制 (overflow: hidden 與 max-width)
        self.assertIn("max-width: 1.5em;", html)
        self.assertIn("overflow: hidden;", html)

    def test_cdn_scripts_sri_importmap_and_precompiled_css(self):
        """驗證外部 CDN 腳本具備固定版本與 SRI、Import Map 傳遞版本、停用 Tailwind Play CDN，以及 color-scheme / reduced-motion 規範"""
        from pathlib import Path
        import re

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # 1. No Tailwind Play CDN
        self.assertNotIn("cdn.tailwindcss.com", html)
        self.assertIn('/static/css/app.css?v=', html)
        self.assertTrue(Path("api/static/css/app.css").is_file(), "Compiled api/static/css/app.css must exist")
        self.assertTrue(Path("tailwind.config.js").is_file(), "tailwind.config.js must exist")
        self.assertTrue(Path("scripts/build_css.sh").is_file(), "scripts/build_css.sh must exist")

        # 2. Pinned external CDN scripts with SRI + crossorigin + referrerpolicy
        cdn_patterns = [
            r"vue@3\.\d+\.\d+/dist/vue\.global\.prod\.js",
            r"chart\.js@4\.\d+\.\d+/dist/chart\.umd\.min\.js",
            r"hammerjs@2\.0\.8/hammer\.min\.js",
            r"chartjs-plugin-zoom@2\.0\.1/dist/chartjs-plugin-zoom\.min\.js",
        ]
        for pat in cdn_patterns:
            match = re.search(rf'<script[^>]*src="https://cdn\.jsdelivr\.net/npm/{pat}"[^>]*>', html)
            self.assertIsNotNone(match, f"Missing pinned script matching {pat}")
            tag = match.group(0)
            self.assertIn('integrity="sha384-', tag)
            self.assertIn('crossorigin="anonymous"', tag)
            self.assertIn('referrerpolicy="no-referrer"', tag)

        # 3. Import map propagates APP_VERSION fingerprint & app.js drops hardcoded ?v=2.6.0
        self.assertIn('<script type="importmap">', html)
        self.assertIn('"/static/js/api-client.js": "/static/js/api-client.js?v=', html)
        self.assertIn('"/static/js/chart-manager.js": "/static/js/chart-manager.js?v=', html)
        self.assertIn('"/static/js/risk-xray.js": "/static/js/risk-xray.js?v=', html)

        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        self.assertNotIn("?v=2.6.0", app_js)

        # 4. color-scheme synchronization & precision reduced-motion & content-visibility
        self.assertIn('<meta name="color-scheme"', html)
        self.assertIn("colorScheme", html)
        self.assertIn("colorScheme", app_js)
        self.assertNotIn("0.01ms !important", html)
        self.assertIn("content-visibility: auto", html)
        self.assertIn("contain-intrinsic-size", html)

    def test_static_cache_headers_and_manifest_theme_color(self):
        """驗證 /static/ 帶版本指紋可長效 immutable 快取，未帶版本維持 no-cache，且 site.webmanifest theme_color 對齊 #FFB010"""
        versioned = self.client.get("/static/js/app.js?v=test123")
        self.assertEqual(versioned.status_code, 200)
        vc = versioned.headers.get("cache-control", "")
        self.assertIn("immutable", vc)
        self.assertIn("max-age=31536000", vc)

        unversioned = self.client.get("/static/js/app.js")
        self.assertEqual(unversioned.status_code, 200)
        uc = unversioned.headers.get("cache-control", "")
        self.assertIn("no-cache", uc)
        self.assertIn("must-revalidate", uc)

        manifest = self.client.get("/site.webmanifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.json()["theme_color"], "#FFB010")

    def test_asset_fingerprint_changes_on_content_change_even_same_mtime(self):
        """驗證 get_asset_fingerprint() 採用內容雜湊，即使 mtime 相同仍能即時反映內容變更，並驗證 SRI sha384 格式有效"""
        import base64
        import re
        import tempfile
        from api.main import get_asset_fingerprint

        with tempfile.TemporaryDirectory() as tmpdir:
            js_file = os.path.join(tmpdir, "sample.js")
            with open(js_file, "wb") as f:
                f.write(b"console.log('v1');")
            fixed_time = (1700000000, 1700000000)
            os.utime(js_file, fixed_time)
            fp1 = get_asset_fingerprint(scan_dirs=(tmpdir,))

            with open(js_file, "wb") as f:
                f.write(b"console.log('v2');")
            os.utime(js_file, fixed_time)
            fp2 = get_asset_fingerprint(scan_dirs=(tmpdir,))

            self.assertNotEqual(fp1, fp2)

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        digests = re.findall(r'integrity="sha384-([A-Za-z0-9+/=]+)"', resp.text)
        self.assertEqual(len(digests), 4)
        for d in digests:
            self.assertEqual(len(base64.b64decode(d, validate=True)), 48)


if __name__ == "__main__":
    unittest.main()
