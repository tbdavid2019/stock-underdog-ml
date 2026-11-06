"""
backtest/statistical_validation.py - 策略歷史驗證統計顯著性檢定 (Statistical Validation)

提供與單筆撮合解耦之純函式統計檢定模組：
1. Rademacher 符號翻轉對數日報酬均值檢定 (p_value_return)
2. 交易順序置換最大回撤路徑檢定 (p_value_max_drawdown, mc_95_worst_drawdown)
3. Circular Block Bootstrap 年化 Sharpe 95% 信賴區間 [sharpe_ci_low, sharpe_ci_high]
4. 零變異 (std <= 1e-12) 與小樣本守衛，全程通過 `json.dumps(..., allow_nan=False)`
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def _clean_numeric_sequence(seq: Optional[Sequence[Any]]) -> np.ndarray:
    """提取並過濾非有限數值，回傳 1D float64 numpy array"""
    if seq is None:
        return np.array([], dtype=np.float64)
    cleaned = []
    for x in seq:
        if x is None:
            continue
        try:
            f = float(x)
            if math.isfinite(f):
                cleaned.append(f)
        except (TypeError, ValueError):
            continue
    return np.array(cleaned, dtype=np.float64)


def _calc_max_drawdown(rets: np.ndarray) -> float:
    """計算複利淨值序列之最大回撤 (Drawdown >= 0.0)"""
    if len(rets) == 0:
        return 0.0
    # 限制單筆跌幅下界防止非物理負資產
    clamped = np.maximum(rets, -0.999999)
    equity = np.cumprod(1.0 + clamped)
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - (equity / peaks)
    dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    return float(np.clip(dd, 0.0, 1.0))


def validate_returns_significance(
    daily_returns: Optional[Sequence[Any]] = None,
    trade_returns: Optional[Sequence[Any]] = None,
    random_seed: int = 42,
    b_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """執行完整統計顯著性檢定 (純函式，固定隨機種子可位元級重現)"""

    d_rets = _clean_numeric_sequence(daily_returns)
    t_rets = _clean_numeric_sequence(trade_returns)

    t_count = len(d_rets)
    m_count = len(t_rets)

    rng = np.random.default_rng(random_seed)

    # 1. 小樣本守衛
    if t_count < 30 and m_count < 10:
        res = {
            "status": "insufficient_samples",
            "daily_sample_count": t_count,
            "trade_sample_count": m_count,
            "p_value_return": None,
            "p_value_max_drawdown": None,
            "mc_95_worst_drawdown": None,
            "sharpe_ci_low": None,
            "sharpe_ci_high": None,
            "observed_sharpe": None,
            "observed_return_mean": None,
            "observed_max_drawdown": None,
            "is_statistically_significant": False,
        }
        json.dumps(res, allow_nan=False)
        return res

    # 2. 零變異與常數序列守衛
    d_std = float(np.std(d_rets, ddof=1)) if t_count > 1 else 0.0
    t_std = float(np.std(t_rets, ddof=1)) if m_count > 1 else 0.0

    if (t_count >= 30 and d_std <= 1e-12) or (t_count < 30 and m_count >= 10 and t_std <= 1e-12):
        res = {
            "status": "degenerate_zero_variance",
            "daily_sample_count": t_count,
            "trade_sample_count": m_count,
            "p_value_return": 1.0,
            "p_value_max_drawdown": 1.0,
            "mc_95_worst_drawdown": 0.0,
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "observed_sharpe": 0.0,
            "observed_return_mean": round(float(np.mean(d_rets)), 6) if t_count > 0 else 0.0,
            "observed_max_drawdown": 0.0,
            "is_statistically_significant": False,
        }
        json.dumps(res, allow_nan=False)
        return res

    # 3. 日報酬檢定 (T >= 30)
    p_value_return: Optional[float] = None
    sharpe_ci_low: Optional[float] = None
    sharpe_ci_high: Optional[float] = None
    observed_sharpe: Optional[float] = None
    observed_ret_mean: Optional[float] = None

    if t_count >= 30:
        observed_ret_mean = round(float(np.mean(d_rets)), 6)
        obs_s = float(np.sqrt(252.0) * np.mean(d_rets) / (d_std + 1e-12))
        observed_sharpe = round(obs_s, 4)

        # 3.1 Rademacher 符號翻轉檢定
        # 有限對數日報酬 z_t = ln(1 + max(r_t, -0.999999))
        clamped_d = np.maximum(d_rets, -0.999999)
        z = np.log1p(clamped_d)
        z_mean = np.mean(z)
        z_std = np.std(z, ddof=1)

        if z_std <= 1e-12:
            p_value_return = 1.0
        else:
            t_obs = z_mean / z_std
            # 產生 B 次隨機符號 (-1, +1)
            signs = rng.choice([-1.0, 1.0], size=(b_bootstrap, t_count))
            z_sim = signs * z  # (B, T)
            sim_means = np.mean(z_sim, axis=1)
            sim_stds = np.std(z_sim, axis=1, ddof=1)
            t_sims = np.divide(sim_means, sim_stds + 1e-12)
            p_val = (1.0 + np.sum(t_sims >= t_obs)) / (b_bootstrap + 1.0)
            p_value_return = round(float(p_val), 4)

        # 3.2 Circular Block Bootstrap for Sharpe CI
        block_len = max(5, int(math.floor(t_count ** (1.0 / 3.0))))
        n_blocks = int(math.ceil(t_count / block_len))
        boot_sharpes = np.zeros(b_bootstrap, dtype=np.float64)

        for b_idx in range(b_bootstrap):
            start_indices = rng.integers(0, t_count, size=n_blocks)
            sample_indices = []
            for s_idx in start_indices:
                for offset in range(block_len):
                    sample_indices.append((s_idx + offset) % t_count)
            sample_rets = d_rets[sample_indices[:t_count]]

            s_std = np.std(sample_rets, ddof=1)
            if s_std <= 1e-12:
                boot_sharpes[b_idx] = 0.0
            else:
                boot_sharpes[b_idx] = np.sqrt(252.0) * np.mean(sample_rets) / s_std

        sharpe_ci_low = round(float(np.percentile(boot_sharpes, 2.5)), 4)
        sharpe_ci_high = round(float(np.percentile(boot_sharpes, 97.5)), 4)

    # 4. 交易順序置換最大回撤檢定 (M >= 10)
    p_value_max_drawdown: Optional[float] = None
    mc_95_worst_drawdown: Optional[float] = None
    observed_max_drawdown: Optional[float] = None

    if m_count >= 10:
        obs_dd = _calc_max_drawdown(t_rets)
        observed_max_drawdown = round(obs_dd, 4)

        if t_std <= 1e-12:
            p_value_max_drawdown = 1.0
            mc_95_worst_drawdown = observed_max_drawdown
        else:
            perm_dds = np.zeros(b_bootstrap, dtype=np.float64)
            for b_idx in range(b_bootstrap):
                permuted = rng.permutation(t_rets)
                perm_dds[b_idx] = _calc_max_drawdown(permuted)

            # 檢定真實路徑回撤是否顯著小於隨機打散路徑
            p_dd = (1.0 + np.sum(perm_dds <= obs_dd)) / (b_bootstrap + 1.0)
            p_value_max_drawdown = round(float(p_dd), 4)
            mc_95_worst_drawdown = round(float(np.percentile(perm_dds, 95.0)), 4)
    elif t_count >= 30:
        # 若未傳入交易報酬，則基於日報酬序列計算回撤
        observed_max_drawdown = round(_calc_max_drawdown(d_rets), 4)

    # 5. 綜合顯著性判定
    # 需滿足：樣本充足、p_value_return < 0.05 且 Sharpe 信賴區間下界 > 0
    is_sig = False
    if p_value_return is not None and p_value_return < 0.05:
        if sharpe_ci_low is not None and sharpe_ci_low > 0:
            is_sig = True
        elif sharpe_ci_low is None:
            is_sig = True

    status_str = "ok" if (t_count >= 30 or m_count >= 10) else "insufficient_samples"

    res = {
        "status": status_str,
        "daily_sample_count": t_count,
        "trade_sample_count": m_count,
        "p_value_return": p_value_return,
        "p_value_max_drawdown": p_value_max_drawdown,
        "mc_95_worst_drawdown": mc_95_worst_drawdown,
        "sharpe_ci_low": sharpe_ci_low,
        "sharpe_ci_high": sharpe_ci_high,
        "observed_sharpe": observed_sharpe,
        "observed_return_mean": observed_ret_mean,
        "observed_max_drawdown": observed_max_drawdown,
        "is_statistically_significant": is_sig,
    }

    json.dumps(res, allow_nan=False)
    return res
