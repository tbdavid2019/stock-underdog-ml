"""
backtest/turnover_optimizer.py - 純 NumPy ADMM 非對稱換手率多頭部位優化器 (Turnover-Aware Optimizer)

提供零外部商業求解器 (無 cvxpy / scipy) 依賴之純 Python / NumPy 確定性 ADMM 演算法：
1. 對角線收縮共變異數正則化 (Shrinkage Covariance)
2. 買賣非對稱 L1 換手懲罰 (gamma_buy, gamma_sell 支援台股 0.3% 證交稅抑制)
3. Dykstra 凸可行域投影 + 最終確定性裁剪 (clip_to_feasible_polytope，殘差 <= 1e-8)
4. 懲罰參數下界 rho >= 1.0 徹底杜絕 risk_aversion=0 除零發散
5. 100% 通過 `json.dumps(..., allow_nan=False)`
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np


def clip_to_feasible_polytope(
    w: np.ndarray,
    max_single_weight: float = 0.10,
    max_total_weight: float = 1.0,
    sector_indices: Optional[Mapping[str, Sequence[int]]] = None,
    max_sector_weight: float = 0.30,
    max_passes: int = 10,
) -> np.ndarray:
    """將權重向量確定性投影/裁剪至非負凸多面體可行域 (殘差 <= 1e-8)

    約束條件：
    1. 0 <= w_i <= max_single_weight
    2. sum(w) <= max_total_weight
    3. sum_{i in g} w_i <= max_sector_weight
    """
    w_out = np.copy(w).astype(np.float64)
    # 基礎非負與單股上限裁剪
    w_out = np.clip(w_out, 0.0, max_single_weight)

    for _ in range(max_passes):
        changed = False

        # 產業上限檢查與等比收縮
        if sector_indices:
            for sector, indices in sector_indices.items():
                if len(indices) == 0:
                    continue
                sec_sum = float(np.sum(w_out[indices]))
                if sec_sum > max_sector_weight + 1e-10:
                    scale = max_sector_weight / sec_sum
                    w_out[indices] *= scale
                    changed = True

        # 總部位上限檢查與等比收縮
        tot_sum = float(np.sum(w_out))
        if tot_sum > max_total_weight + 1e-10:
            scale = max_total_weight / tot_sum
            w_out *= scale
            changed = True

        # 單股上限再保證
        over_mask = w_out > max_single_weight + 1e-10
        if np.any(over_mask):
            w_out[over_mask] = max_single_weight
            changed = True

        neg_mask = w_out < 0.0
        if np.any(neg_mask):
            w_out[neg_mask] = 0.0
            changed = True

        if not changed:
            break

    return np.round(w_out, 8)


def project_dykstra(
    v: np.ndarray,
    max_single_weight: float = 0.10,
    max_total_weight: float = 1.0,
    sector_indices: Optional[Mapping[str, Sequence[int]]] = None,
    max_sector_weight: float = 0.30,
    max_iter: int = 25,
) -> np.ndarray:
    """以 Dykstra 交替投影法將向量 v 歐氏投影至凸多面體可行域 C"""
    w = np.copy(v).astype(np.float64)
    p_box = np.zeros_like(w)
    p_tot = np.zeros_like(w)
    p_sec: Dict[str, np.ndarray] = {}
    if sector_indices:
        for sec, idx_list in sector_indices.items():
            if len(idx_list) > 0:
                p_sec[sec] = np.zeros(len(idx_list), dtype=np.float64)

    for _ in range(max_iter):
        # 1. 盒約束 [0, max_single_weight]
        y_box = w + p_box
        w_box = np.clip(y_box, 0.0, max_single_weight)
        p_box = y_box - w_box
        w_curr = np.copy(w_box)

        # 2. 產業約束 (若有)
        if sector_indices:
            for sec, idx_list in sector_indices.items():
                if len(idx_list) == 0:
                    continue
                idx_arr = np.array(idx_list, dtype=int)
                y_s = w_curr[idx_arr] + p_sec[sec]
                s_sec = float(np.sum(y_s))
                if s_sec > max_sector_weight:
                    w_sec = y_s - (s_sec - max_sector_weight) / len(idx_arr)
                else:
                    w_sec = y_s
                p_sec[sec] = y_s - w_sec
                w_curr[idx_arr] = w_sec

        # 3. 總部位半空間 sum(w) <= max_total_weight
        y_tot = w_curr + p_tot
        s_tot = float(np.sum(y_tot))
        if s_tot > max_total_weight:
            w_tot = y_tot - (s_tot - max_total_weight) / len(y_tot)
        else:
            w_tot = y_tot
        p_tot = y_tot - w_tot

        w = w_tot

    return w


def optimize_portfolio_weights_admm(
    expected_returns: Union[Sequence[float], Mapping[str, float], np.ndarray],
    cov_matrix: Optional[np.ndarray] = None,
    returns_history: Optional[np.ndarray] = None,
    current_weights: Optional[Union[Sequence[float], Mapping[str, float], np.ndarray]] = None,
    tickers: Optional[Sequence[str]] = None,
    gamma_buy: float = 0.0015,
    gamma_sell: float = 0.0045,
    risk_aversion: float = 1.0,
    max_single_weight: float = 0.10,
    max_total_weight: float = 1.0,
    sector_map: Optional[Mapping[str, str]] = None,
    max_sector_weight: float = 0.30,
    shrinkage: float = 0.10,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """以純 NumPy ADMM 求解非對稱換手率多頭部位權重優化 (零外部求解器，allow_nan=False)"""

    # 1. 解析預期報酬與標的代號
    ticker_list: List[str] = []
    if isinstance(expected_returns, Mapping):
        ticker_list = list(expected_returns.keys())
        mu_vec = np.array([float(expected_returns[tk]) for tk in ticker_list], dtype=np.float64)
    elif tickers is not None:
        ticker_list = [str(t) for t in tickers]
        mu_vec = np.array(expected_returns, dtype=np.float64).flatten()
    else:
        mu_vec = np.array(expected_returns, dtype=np.float64).flatten()
        ticker_list = [f"asset_{i}" for i in range(len(mu_vec))]

    n_assets = len(mu_vec)
    if n_assets == 0:
        res = {
            "status": "error_empty_assets",
            "solver_status": "error",
            "weights": {},
            "optimal_weights": {},
            "one_way_turnover": 0.0,
            "turnover": 0.0,
            "buy_turnover": 0.0,
            "sell_turnover": 0.0,
            "cash_weight": 1.0,
            "iterations": 0,
        }
        json.dumps(res, allow_nan=False)
        return res

    # 檢查輸入有限性
    if not np.all(np.isfinite(mu_vec)):
        zero_weights = {tk: 0.0 for tk in ticker_list}
        res = {
            "status": "degenerate_input_non_finite",
            "solver_status": "fallback_hold_previous",
            "weights": zero_weights,
            "optimal_weights": zero_weights,
            "one_way_turnover": 0.0,
            "turnover": 0.0,
            "buy_turnover": 0.0,
            "sell_turnover": 0.0,
            "cash_weight": 1.0,
            "iterations": 0,
        }
        json.dumps(res, allow_nan=False)
        return res

    # 2. 既有持倉權重 w_prev 解析
    if current_weights is not None:
        if isinstance(current_weights, Mapping):
            w_prev = np.array([float(current_weights.get(tk, 0.0)) for tk in ticker_list], dtype=np.float64)
        else:
            w_prev = np.array(current_weights, dtype=np.float64).flatten()
            if len(w_prev) != n_assets:
                w_prev = np.zeros(n_assets, dtype=np.float64)
        w_prev = np.nan_to_num(w_prev, nan=0.0, posinf=0.0, neginf=0.0)
        w_prev = clip_to_feasible_polytope(w_prev, max_single_weight=max_single_weight, max_total_weight=max_total_weight)
    else:
        w_prev = np.zeros(n_assets, dtype=np.float64)

    # 3. 共變異數矩陣計算與對角線收縮正則化
    if cov_matrix is not None:
        sigma = np.copy(cov_matrix).astype(np.float64)
    elif returns_history is not None:
        ret_h = np.copy(returns_history).astype(np.float64)
        if ret_h.shape[0] < 30 or ret_h.shape[1] != n_assets or not np.all(np.isfinite(ret_h)):
            # 樣本過少或非有限值降級
            prev_dict = {tk: round(float(w_prev[i]), 6) for i, tk in enumerate(ticker_list)}
            res = {
                "status": "insufficient_history_fallback",
                "solver_status": "fallback_hold_previous",
                "weights": prev_dict,
                "optimal_weights": prev_dict,
                "one_way_turnover": 0.0,
                "turnover": 0.0,
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
                "cash_weight": round(float(1.0 - np.sum(w_prev)), 6),
                "iterations": 0,
            }
            json.dumps(res, allow_nan=False)
            return res
        sigma = np.cov(ret_h, rowvar=False, ddof=1)
    else:
        # 若未提供共變異數矩陣，以單位矩陣代替
        sigma = np.eye(n_assets, dtype=np.float64) * 0.04

    if not np.all(np.isfinite(sigma)):
        sigma = np.eye(n_assets, dtype=np.float64) * 0.04

    # 對角線收縮 (Diagonal Shrinkage)
    diag_target = np.diag(np.diag(sigma))
    sigma_shrunk = (1.0 - shrinkage) * sigma + shrinkage * diag_target
    # 保證對稱
    sigma_shrunk = 0.5 * (sigma_shrunk + sigma_shrunk.T)

    # 4. 解析產業索引映射
    sector_indices: Dict[str, List[int]] = {}
    if sector_map:
        for idx, tk in enumerate(ticker_list):
            sec = str(sector_map.get(tk, "default"))
            if sec not in sector_indices:
                sector_indices[sec] = []
            sector_indices[sec].append(idx)

    # 5. 3-Block 共識 ADMM 演算法配置
    # 懲罰參數下界 rho >= 1.0 (支援 risk_aversion=0.0 絕不除以零)
    lam = max(0.0, float(risk_aversion))
    sigma_norm = float(np.linalg.norm(sigma_shrunk, 2))
    rho = max(1.0, float(lam * sigma_norm))

    # 二次項矩陣 H = lambda * Sigma + 2 * rho * I，最小特徵值恆 >= 2 * rho >= 2.0
    h_matrix = lam * sigma_shrunk + 2.0 * rho * np.eye(n_assets, dtype=np.float64)
    try:
        chol = np.linalg.cholesky(h_matrix)
    except np.linalg.LinAlgError:
        # 若數值微小非正定，額外加入微小擾動
        h_matrix += 1e-6 * np.eye(n_assets, dtype=np.float64)
        chol = np.linalg.cholesky(h_matrix)

    w = np.copy(w_prev)
    z1 = np.copy(w_prev)
    z2 = np.copy(w_prev)
    u1 = np.zeros(n_assets, dtype=np.float64)
    u2 = np.zeros(n_assets, dtype=np.float64)

    tau_buy = float(gamma_buy / rho)
    tau_sell = float(gamma_sell / rho)

    solver_status = "max_iterations_feasible"
    iters_completed = max_iter

    # 6. 3-Block 共識 ADMM 主迭代迴圈
    for it in range(max_iter):
        z1_old = np.copy(z1)
        z2_old = np.copy(z2)

        # 6.1 w-update: 無約束二次閉式線性解
        # H * w = mu + rho * (z1 - u1) + rho * (z2 - u2)
        rhs = mu_vec + rho * (z1 - u1) + rho * (z2 - u2)
        w = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))

        # 6.2 z1-update: 非對稱 L1 換手軟閾值 (Asymmetric Soft-Thresholding)
        v1 = w + u1
        delta = v1 - w_prev
        z1_new = np.copy(w_prev)
        buy_mask = delta > tau_buy
        sell_mask = delta < -tau_sell
        z1_new[buy_mask] = v1[buy_mask] - tau_buy
        z1_new[sell_mask] = v1[sell_mask] + tau_sell
        z1 = z1_new

        # 6.3 z2-update: Dykstra 歐氏投影至凸多面體可行域 C
        z2 = project_dykstra(
            w + u2,
            max_single_weight=max_single_weight,
            max_total_weight=max_total_weight,
            sector_indices=sector_indices,
            max_sector_weight=max_sector_weight,
            max_iter=25,
        )

        # 6.4 對偶變數更新 (Scaled Dual Updates)
        u1 += (w - z1)
        u2 += (w - z2)

        # 6.5 收斂判準
        r_prim = max(float(np.linalg.norm(w - z1)), float(np.linalg.norm(w - z2)))
        r_dual = float(rho * max(np.linalg.norm(z1 - z1_old), np.linalg.norm(z2 - z2_old)))

        if r_prim < tol and r_dual < tol:
            solver_status = "converged"
            iters_completed = it + 1
            break

    # 7. 最終可行域精確裁剪投影 (以 z2 投影結果為基底，保證約束殘差 <= 1e-8)
    w_final = clip_to_feasible_polytope(
        z2,
        max_single_weight=max_single_weight,
        max_total_weight=max_total_weight,
        sector_indices=sector_indices,
        max_sector_weight=max_sector_weight,
    )

    # 8. 換手率與權重字典彙整 (向下取 6 位小數確保不會因四捨五入超限)
    w_safe = np.floor(w_final * 1e6) / 1e6
    w_safe = clip_to_feasible_polytope(
        w_safe,
        max_single_weight=max_single_weight,
        max_total_weight=max_total_weight,
        sector_indices=sector_indices,
        max_sector_weight=max_sector_weight,
    )

    diff = w_safe - w_prev
    buy_turnover = float(np.sum(np.maximum(diff, 0.0)))
    sell_turnover = float(np.sum(np.maximum(-diff, 0.0)))
    one_way_turnover = float(0.5 * np.sum(np.abs(diff)))
    cash_w = float(max(0.0, 1.0 - np.sum(w_safe)))

    weights_dict = {ticker_list[i]: float(w_safe[i]) for i in range(n_assets)}

    result = {
        "status": "ok",
        "solver_status": solver_status,
        "weights": weights_dict,
        "optimal_weights": weights_dict,
        "one_way_turnover": round(one_way_turnover, 6),
        "turnover": round(one_way_turnover, 6),
        "buy_turnover": round(buy_turnover, 6),
        "sell_turnover": round(sell_turnover, 6),
        "cash_weight": round(cash_w, 6),
        "iterations": iters_completed,
    }

    json.dumps(result, allow_nan=False)
    return result
