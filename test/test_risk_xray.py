"""
test/test_risk_xray.py - Tests for backtest.risk_xray.
Validates:
1. Cash-diluted portfolio preserves Euler sum (sum(PCV) == 1.0, sum(CV) == ann_vol_total) and scaled total VaR.
2. Window between 30 and 59 days withholds 95% and 99% tail estimates (insufficient_tail_samples).
3. Window between 60 and 99 days computes 95% VaR but withholds 99% VaR.
4. Window >= 100 days computes both 95% and 99% VaR/CVaR.
5. Unadjusted 1:4 stock split (-75%) filtered via Corporate Action Jump Guard.
6. Mixed TW (TWD) and US (USD) basket discloses local_currency_return_proxy and FX warnings.
7. Negative correlation hedge yields PCV < 0 and is_hedge=True.
8. Strict json.dumps(..., allow_nan=False) guarantee.
"""

import json
import unittest
import numpy as np
import pandas as pd

from backtest.risk_xray import calculate_portfolio_risk_xray


class TestRiskXRay(unittest.TestCase):

    def _generate_synthetic_series(self, n_days: int, seed: int = 42, corr: float = 0.5):
        rng = np.random.default_rng(seed)
        # Covariance matrix for 2 assets
        vol1, vol2 = 0.20 / np.sqrt(252), 0.25 / np.sqrt(252)
        cov12 = corr * vol1 * vol2
        cov = np.array([[vol1 ** 2, cov12], [cov12, vol2 ** 2]])
        rets = rng.multivariate_normal([0.0005, 0.0003], cov, size=n_days)

        # Build price paths starting at 100.0
        p1 = [100.0]
        p2 = [100.0]
        dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-01-01", periods=n_days + 1)]
        for r1, r2 in rets:
            p1.append(p1[-1] * (1.0 + r1))
            p2.append(p2[-1] * (1.0 + r2))

        s1 = pd.Series(p1, index=dates)
        s2 = pd.Series(p2, index=dates)
        return s1, s2

    def test_cash_diluted_portfolio_and_euler_identities(self):
        # N = 120 days (>= 100 days)
        s1, s2 = self._generate_synthetic_series(n_days=120, seed=101)
        
        # Test full investment (w_cash = 0) vs cash dilution (w_cash = 0.30)
        # Risk sleeve: w1 = 0.40, w2 = 0.60
        # Total portfolio: w1_tot = 0.28, w2_tot = 0.42, w_cash = 0.30
        weights_sleeve = {"A": 0.40, "B": 0.60}
        weights_total = {"A": 0.28, "B": 0.42}

        res_sleeve = calculate_portfolio_risk_xray(
            {"A": s1, "B": s2},
            weights=weights_sleeve
        )
        res_total = calculate_portfolio_risk_xray(
            {"A": s1, "B": s2},
            weights=weights_total
        )

        self.assertEqual(res_total["status"], "ok")
        self.assertAlmostEqual(res_total["cash_weight"], 0.30, places=4)
        self.assertEqual(res_total["effective_bars"], 120)
        self.assertIsNotNone(res_total["effective_n_risk_sleeve"])
        self.assertEqual(res_total["assets"], res_total["euler_contributions"])

        # 1. Total VaR should equal 0.70 * Sleeve VaR
        self.assertIsNotNone(res_total["var_95"])
        self.assertIsNotNone(res_sleeve["var_95"])
        self.assertEqual(res_total["var_95_1d"], res_total["var_95"])
        self.assertAlmostEqual(res_total["var_95_20d"], res_total["var_95"] * np.sqrt(20.0), places=5)
        self.assertEqual(res_total["var_99_1d"], res_total["var_99"])
        self.assertAlmostEqual(res_total["var_99_20d"], res_total["var_99"] * np.sqrt(20.0), places=5)
        self.assertAlmostEqual(res_total["var_95"], 0.70 * res_sleeve["var_95"], places=5)
        self.assertAlmostEqual(res_total["cvar_95"], 0.70 * res_sleeve["cvar_95"], places=5)

        # 2. Total Volatility should equal 0.70 * Sleeve Volatility
        self.assertAlmostEqual(res_total["annualized_volatility_total"], 0.70 * res_total["annualized_volatility_risk_sleeve"], places=5)

        # 3. Euler sum identities: sum(CV) == ann_vol_total, sum(PCV) == 1.0
        cv_sum = sum(item["component_volatility"] for item in res_total["euler_contributions"])
        pcv_sum = sum(item["percentage_contribution"] for item in res_total["euler_contributions"])

        self.assertAlmostEqual(cv_sum, res_total["annualized_volatility_total"], places=5)
        self.assertAlmostEqual(pcv_sum, 1.0, places=5)

        for asset in res_total["assets"]:
            self.assertIn("weight", asset)
            self.assertIn("annualized_volatility", asset)
            self.assertIn("marginal_contribution", asset)

        # Check JSON serialization allow_nan=False
        dumped = json.dumps(res_total, allow_nan=False)
        self.assertTrue(len(dumped) > 0)

    def test_tiered_sample_thresholds_30_to_59(self):
        # 45 days of returns -> 46 price bars
        s1, s2 = self._generate_synthetic_series(n_days=45, seed=202)
        res = calculate_portfolio_risk_xray({"A": s1, "B": s2})

        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["sample_count"], 45)
        self.assertIsNotNone(res["annualized_volatility_total"])
        self.assertIsNotNone(res["hhi_risk_sleeve"])
        self.assertEqual(len(res["euler_contributions"]), 2)

        # 95% and 99% must be None
        self.assertIsNone(res["var_95"])
        self.assertIsNone(res["cvar_95"])
        self.assertIsNone(res["var_99"])
        self.assertIsNone(res["cvar_99"])
        self.assertEqual(res["tail_95_status"], "insufficient_tail_samples")
        self.assertEqual(res["tail_99_status"], "insufficient_tail_samples")

    def test_tiered_sample_thresholds_60_to_99(self):
        # 75 days of returns -> 76 price bars
        s1, s2 = self._generate_synthetic_series(n_days=75, seed=303)
        res = calculate_portfolio_risk_xray({"A": s1, "B": s2})

        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["sample_count"], 75)
        # 95% is computed
        self.assertIsNotNone(res["var_95"])
        self.assertIsNotNone(res["cvar_95"])
        self.assertEqual(res["tail_95_status"], "ok")
        self.assertEqual(res["tail_95_sample_count"], 75)

        # 99% is withheld
        self.assertIsNone(res["var_99"])
        self.assertIsNone(res["cvar_99"])
        self.assertEqual(res["tail_99_status"], "insufficient_tail_samples")
        self.assertIsNone(res["tail_99_sample_count"])

    def test_tiered_sample_thresholds_below_30(self):
        # 20 days of returns
        s1, s2 = self._generate_synthetic_series(n_days=20, seed=404)
        res = calculate_portfolio_risk_xray({"A": s1, "B": s2})

        self.assertEqual(res["status"], "insufficient_history")
        self.assertEqual(res["sample_count"], 20)
        self.assertEqual(res["effective_bars"], 20)
        self.assertEqual(res["assets"], [])
        self.assertIsNone(res["annualized_volatility_total"])
        self.assertIsNone(res["var_95"])
        self.assertEqual(len(res["euler_contributions"]), 0)

    def test_corporate_action_jump_guard(self):
        # Create 100 days of data, insert a 1:4 stock split cliff (-75%) on day 50 for TW stock
        s1, s2 = self._generate_synthetic_series(n_days=100, seed=505)
        s1_split = s1.copy()
        split_date = s1.index[50]
        # Introduce a -75% split drop for all subsequent days
        s1_split.iloc[50:] = s1_split.iloc[50:] * 0.25

        res = calculate_portfolio_risk_xray(
            {"2330.TW": s1_split, "2303.TW": s2},
            price_adjustment_modes={"2330.TW": "raw_close", "2303.TW": "raw_close"}
        )

        self.assertEqual(res["status"], "ok")
        # Split suspect should be caught
        self.assertTrue(len(res["corporate_action_suspects"]) >= 1)
        suspect = res["corporate_action_suspects"][0]
        self.assertEqual(suspect["ticker"], "2330.TW")
        self.assertEqual(suspect["date"], split_date)
        self.assertAlmostEqual(suspect["return"], -0.75, places=2)
        self.assertAlmostEqual(suspect["raw_return"], -0.75, places=2)

        # The suspect date should be excluded, so sample_count = 99 (100 - 1)
        self.assertEqual(res["sample_count"], 99)
        # VaR should not be blown up to ~75%
        self.assertTrue(res["var_95"] < 0.10)

    def test_cross_market_currency_disclosure(self):
        s1, s2 = self._generate_synthetic_series(n_days=65, seed=606)
        res = calculate_portfolio_risk_xray(
            {"2330.TW": s1, "NVDA": s2},
            currencies={"2330.TW": "TWD", "NVDA": "USD"}
        )

        self.assertEqual(res["currency_mode"], "local_currency_return_proxy")
        self.assertIsNotNone(res["fx_unhedged_warning"])
        self.assertIsNotNone(res["calendar_mismatch_warning"])
        self.assertIn("匯率", res["fx_unhedged_warning"])

    def test_negative_hedging_pcv(self):
        # Strong negative correlation (corr = -0.90)
        s1, s2 = self._generate_synthetic_series(n_days=80, seed=707, corr=-0.90)
        # Give A 80% weight and B 20% weight; B acts as a negative hedge to A
        res = calculate_portfolio_risk_xray(
            {"A": s1, "B": s2},
            weights={"A": 0.85, "B": 0.15}
        )
        self.assertEqual(res["status"], "ok")
        contribs = {item["ticker"]: item for item in res["euler_contributions"]}

        # Asset B should have negative component volatility and PCV < 0
        self.assertTrue(contribs["B"]["percentage_contribution"] < 0)
        self.assertTrue(contribs["B"]["is_hedge"])
        # Sum of PCV must still be 1.0!
        pcv_sum = sum(item["percentage_contribution"] for item in res["euler_contributions"])
        self.assertAlmostEqual(pcv_sum, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
