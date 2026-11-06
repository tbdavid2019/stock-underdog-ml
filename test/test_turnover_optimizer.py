"""
test/test_turnover_optimizer.py - Tests for backtest.turnover_optimizer.
Validates:
1. Asymmetric sell penalty suppresses sell turnover compared to zero penalty.
2. Numerical stability under risk_aversion = 0.0 (no division by zero).
3. Constraint feasibility bounds residual <= 1e-8 (max_single_weight, max_sector_weight, max_total_weight).
4. Degenerate/collinear return series and short samples fallback gracefully with 0 NaNs.
5. JSON serialization with allow_nan=False.
"""

import json
import unittest
import numpy as np

from backtest.turnover_optimizer import (
    clip_to_feasible_polytope,
    optimize_portfolio_weights_admm,
)


class TestTurnoverOptimizer(unittest.TestCase):

    def test_asymmetric_sell_penalty_suppresses_turnover(self):
        # 5 assets: initial weights held
        tickers = ["A", "B", "C", "D", "E"]
        w_prev = {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08}
        # Slight variation in expected returns: B & C slightly lower, A & D slightly higher
        mu = {"A": 0.002, "B": -0.001, "C": -0.001, "D": 0.003, "E": 0.000}

        # Case 1: Zero turnover penalties
        res_zero = optimize_portfolio_weights_admm(
            expected_returns=mu,
            current_weights=w_prev,
            gamma_buy=0.0,
            gamma_sell=0.0,
            risk_aversion=1.0,
        )

        # Case 2: Asymmetric penalties (gamma_sell = 0.0045 including Taiwan 0.3% tax)
        res_tax = optimize_portfolio_weights_admm(
            expected_returns=mu,
            current_weights=w_prev,
            gamma_buy=0.0015,
            gamma_sell=0.0045,
            risk_aversion=1.0,
        )

        self.assertEqual(res_zero["status"], "ok")
        self.assertEqual(res_tax["status"], "ok")
        self.assertEqual(res_tax["weights"], res_tax["optimal_weights"])
        self.assertEqual(res_tax["one_way_turnover"], res_tax["turnover"])
        self.assertEqual(res_zero["weights"], res_zero["optimal_weights"])

        # Tax-aware turnover should be strictly lower than zero-penalty turnover
        self.assertTrue(res_tax["sell_turnover"] <= res_zero["sell_turnover"])
        self.assertTrue(res_tax["one_way_turnover"] < res_zero["one_way_turnover"])

    def test_risk_aversion_zero_stability(self):
        # When risk_aversion = 0, optimizer relies entirely on rho >= 1.0 and does not divide by zero
        mu = [0.05, 0.02, -0.01, 0.04]
        res = optimize_portfolio_weights_admm(
            expected_returns=mu,
            risk_aversion=0.0,
            max_single_weight=0.10,
            max_total_weight=1.0,
        )

        self.assertEqual(res["status"], "ok")
        self.assertIn(res["solver_status"], ["converged", "max_iterations_feasible"])
        for val in res["weights"].values():
            self.assertTrue(val >= 0.0)
            self.assertTrue(val <= 0.10 + 1e-8)

        # Verify JSON serializability with allow_nan=False
        dumped = json.dumps(res, allow_nan=False)
        self.assertTrue(len(dumped) > 0)

    def test_feasible_polytope_constraints(self):
        # 10 assets with extreme expected returns
        mu = {f"T_{i}": 0.10 * (i + 1) for i in range(10)}
        sector_map = {f"T_{i}": "tech" if i < 5 else "finance" for i in range(10)}

        res = optimize_portfolio_weights_admm(
            expected_returns=mu,
            sector_map=sector_map,
            max_single_weight=0.08,
            max_sector_weight=0.25,
            max_total_weight=0.60,
        )

        self.assertEqual(res["status"], "ok")
        weights = res["weights"]

        # Check each single weight <= 0.08 + 1e-8
        for tk, w in weights.items():
            self.assertTrue(w >= 0.0)
            self.assertTrue(w <= 0.08 + 1e-8, msg=f"{tk} weight {w} exceeded 0.08")

        # Check tech sector <= 0.25 + 1e-8
        tech_sum = sum(weights[f"T_{i}"] for i in range(5))
        self.assertTrue(tech_sum <= 0.25 + 1e-8, msg=f"Tech sum {tech_sum} exceeded 0.25")

        # Check finance sector <= 0.25 + 1e-8
        fin_sum = sum(weights[f"T_{i}"] for i in range(5, 10))
        self.assertTrue(fin_sum <= 0.25 + 1e-8, msg=f"Finance sum {fin_sum} exceeded 0.25")

        # Check total sum <= 0.60 + 1e-8
        tot_sum = sum(weights.values())
        self.assertTrue(tot_sum <= 0.60 + 1e-8, msg=f"Total sum {tot_sum} exceeded 0.60")

    def test_clip_to_feasible_polytope_direct(self):
        w_raw = np.array([0.25, 0.35, -0.05, 0.40, 0.10])
        w_clipped = clip_to_feasible_polytope(w_raw, max_single_weight=0.10, max_total_weight=0.30)

        self.assertTrue(np.all(w_clipped >= 0.0))
        self.assertTrue(np.all(w_clipped <= 0.10 + 1e-8))
        self.assertTrue(np.sum(w_clipped) <= 0.30 + 1e-8)

    def test_degenerate_short_history_fallback(self):
        # 10 days of returns history (< 30)
        short_history = np.random.normal(0.001, 0.02, size=(10, 3))
        w_prev = {"A": 0.05, "B": 0.05, "C": 0.05}
        mu = {"A": 0.01, "B": 0.02, "C": 0.03}

        res = optimize_portfolio_weights_admm(
            expected_returns=mu,
            returns_history=short_history,
            current_weights=w_prev,
        )

        self.assertEqual(res["status"], "insufficient_history_fallback")
        self.assertEqual(res["solver_status"], "fallback_hold_previous")
        self.assertEqual(res["weights"]["A"], 0.05)
        self.assertEqual(res["one_way_turnover"], 0.0)


    def test_coupled_covariance_and_objective_improvement(self):
        # 3 assets with coupled covariance
        rng = np.random.default_rng(777)
        cov = np.array([
            [0.0004, 0.0003, 0.0000],
            [0.0003, 0.0005, 0.0000],
            [0.0000, 0.0000, 0.0001]
        ])
        rets = rng.multivariate_normal([0.001, 0.0005, 0.0002], cov, size=150)
        mu = {"A": 0.0015, "B": 0.0004, "C": 0.0008}
        w_prev = {"A": 0.20, "B": 0.20, "C": 0.20}

        res = optimize_portfolio_weights_admm(
            expected_returns=mu,
            returns_history=rets,
            current_weights=w_prev,
            gamma_buy=0.0005,
            gamma_sell=0.0010,
            risk_aversion=2.0,
            max_single_weight=0.50,
            max_total_weight=0.80,
            tol=1e-4,
            max_iter=900,
        )

        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["solver_status"], "converged")
        self.assertIn("turnover", res)
        self.assertIn("optimal_weights", res)

        w_opt_vec = np.array([res["weights"]["A"], res["weights"]["B"], res["weights"]["C"]])
        w_prev_vec = np.array([w_prev["A"], w_prev["B"], w_prev["C"]])
        mu_vec = np.array([mu["A"], mu["B"], mu["C"]])

        def obj(w):
            dw = w - w_prev_vec
            buy_cost = 0.0005 * np.sum(np.maximum(0.0, dw))
            sell_cost = 0.0010 * np.sum(np.maximum(0.0, -dw))
            return 0.5 * 2.0 * float(w @ cov @ w) - float(mu_vec @ w) + buy_cost + sell_cost

        # The optimized portfolio should achieve a better (lower) objective than the initial equal weights
        self.assertLess(obj(w_opt_vec), obj(w_prev_vec))


if __name__ == "__main__":
    unittest.main()
