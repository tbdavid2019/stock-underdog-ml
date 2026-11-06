"""
test/test_statistical_validation.py - Tests for backtest.statistical_validation.
Validates:
1. Deterministic bit-identical p-values and confidence intervals under same random_seed.
2. Zero-variance and all-zero return series guards (degenerate_zero_variance, p_value=1.0, 0 NaN).
3. Sample count guards for daily returns (T < 30) and trade returns (M < 10).
4. High alpha positive drift yields p < 0.05 and is_statistically_significant=True.
5. Random zero-mean noise does not produce false significance.
6. JSON serialization with allow_nan=False.
"""

import json
import unittest
import numpy as np

from backtest.statistical_validation import validate_returns_significance, _calc_max_drawdown


class TestStatisticalValidation(unittest.TestCase):

    def test_bit_identical_reproducibility(self):
        rng = np.random.default_rng(12345)
        d_rets = rng.normal(0.001, 0.015, size=60).tolist()
        t_rets = rng.normal(0.02, 0.05, size=25).tolist()

        res1 = validate_returns_significance(d_rets, t_rets, random_seed=42)
        res2 = validate_returns_significance(d_rets, t_rets, random_seed=42)

        self.assertEqual(res1["status"], "ok")
        self.assertEqual(res1["p_value_return"], res2["p_value_return"])
        self.assertEqual(res1["p_value_max_drawdown"], res2["p_value_max_drawdown"])
        self.assertEqual(res1["sharpe_ci_low"], res2["sharpe_ci_low"])
        self.assertEqual(res1["sharpe_ci_high"], res2["sharpe_ci_high"])
        self.assertEqual(res1["mc_95_worst_drawdown"], res2["mc_95_worst_drawdown"])

    def test_zero_variance_guard(self):
        # All-zero daily returns (T = 50)
        zeros = [0.0] * 50
        res = validate_returns_significance(daily_returns=zeros)

        self.assertEqual(res["status"], "degenerate_zero_variance")
        self.assertEqual(res["p_value_return"], 1.0)
        self.assertFalse(res["is_statistically_significant"])
        self.assertEqual(res["observed_sharpe"], 0.0)
        self.assertEqual(res["sharpe_ci_low"], 0.0)
        self.assertEqual(res["sharpe_ci_high"], 0.0)

        # Constant non-zero daily returns (std = 0.0)
        constants = [0.005] * 40
        res_const = validate_returns_significance(daily_returns=constants)
        self.assertEqual(res_const["status"], "degenerate_zero_variance")
        self.assertEqual(res_const["p_value_return"], 1.0)
        self.assertFalse(res_const["is_statistically_significant"])

        # Check JSON allow_nan=False
        dumped = json.dumps(res, allow_nan=False)
        self.assertTrue(len(dumped) > 0)

    def test_minimum_sample_guards(self):
        # T = 15 (< 30) and M = 5 (< 10)
        short_d = [0.01] * 15
        short_t = [0.02] * 5
        res = validate_returns_significance(daily_returns=short_d, trade_returns=short_t)

        self.assertEqual(res["status"], "insufficient_samples")
        self.assertIsNone(res["p_value_return"])
        self.assertIsNone(res["p_value_max_drawdown"])
        self.assertFalse(res["is_statistically_significant"])

    def test_high_alpha_significance(self):
        # Strong consistent positive returns (mean = 1.5% daily, vol = 1.0% daily, T = 80)
        rng = np.random.default_rng(999)
        alpha_rets = rng.normal(0.015, 0.010, size=80).tolist()
        trades = rng.normal(0.05, 0.02, size=30).tolist()

        res = validate_returns_significance(alpha_rets, trades, random_seed=42)

        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["p_value_return"] < 0.01)
        self.assertTrue(res["sharpe_ci_low"] > 0)
        self.assertTrue(res["is_statistically_significant"])

    def test_random_noise_not_significant(self):
        # Pure zero-mean noise (mean = 0.0, vol = 2.0% daily, T = 100)
        rng = np.random.default_rng(777)
        noise_rets = rng.normal(0.0, 0.020, size=100).tolist()

        res = validate_returns_significance(noise_rets, random_seed=42)

        self.assertEqual(res["status"], "ok")
        # Should not be statistically significant
        self.assertFalse(res["is_statistically_significant"])
        # Sharpe CI should cover zero (ci_low < 0 and ci_high > 0)
        self.assertTrue(res["sharpe_ci_low"] <= 0)

    def test_calc_max_drawdown_helper(self):
        # +10%, -20%, +10% -> 1.10 -> 0.88 -> 0.968; max drawdown from 1.10 to 0.88 is 0.22/1.10 = 20%
        rets = np.array([0.10, -0.20, 0.10])
        dd = _calc_max_drawdown(rets)
        self.assertAlmostEqual(dd, 0.20, places=4)


if __name__ == "__main__":
    unittest.main()
