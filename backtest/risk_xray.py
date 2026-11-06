"""
backtest/risk_xray.py - 投資組合風險 X 光片 (Portfolio Risk X-Ray)

提供純收盤價矩陣計算、零網路 I/O、100% 通過 `json.dumps(..., allow_nan=False)` 之機構級組合風險診斷：
1. 雙口徑波動率 (全組合含現金口徑 vs 風險槽正規化口徑)
2. 階梯式樣本門檻 (N < 30 insufficient_history; 30<=N<60 僅波動率/歐拉; 60<=N<100 算 95% VaR/CVaR; N>=100 算 99% VaR/CVaR)
3. 支援負對沖值之歐拉波動貢獻拆解 (Euler Component Volatility)
4. 台股原始收盤價企業行為異常跳空守衛 (Corporate Action Jump Guard)
5. 跨市場多幣別與交易日曆錯位雙重揭露
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def _clean_float(val: Any) -> Optional[float]:
    """確保數值為有限浮點數，非有限值 (NaN, Inf) 回傳 None"""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def calculate_portfolio_risk_xray(
    series_map: Mapping[str, Any],
    weights: Optional[Mapping[str, float]] = None,
    price_adjustment_modes: Optional[Mapping[str, str]] = None,
    currencies: Optional[Mapping[str, str]] = None,
    unavailable_tickers: Optional[Sequence[str]] = None,
    jump_threshold_tw: float = 0.25,
    jump_threshold_foreign: float = 0.60,
) -> Dict[str, Any]:
    """計算投資組合風險 X 光片 (零網路 I/O，嚴格 allow_nan=False)"""

    unavail = list(unavailable_tickers or [])
    adj_modes = dict(price_adjustment_modes or {})
    curr_map = dict(currencies or {})

    # 1. 整理各標的收盤價序列並計算日報酬 (去除非有限數值)
    returns_by_ticker: Dict[str, Dict[str, float]] = {}
    corporate_action_suspects: List[Dict[str, Any]] = []
    suspect_dates: set[str] = set()

    for ticker, raw_data in series_map.items():
        if raw_data is None:
            unavail.append(ticker)
            continue

        # 支援 pd.Series, pd.DataFrame, Sequence[Tuple[str, float]], Dict[str, float]
        dates: List[str] = []
        closes: List[float] = []

        if isinstance(raw_data, pd.Series):
            for d, c in raw_data.items():
                cf = _clean_float(c)
                if cf is not None and cf > 0:
                    dates.append(str(d)[:10])
                    closes.append(cf)
        elif isinstance(raw_data, pd.DataFrame):
            col_close = "Close" if "Close" in raw_data.columns else ("close" if "close" in raw_data.columns else None)
            if col_close:
                for idx, row in raw_data.iterrows():
                    cf = _clean_float(row[col_close])
                    if cf is not None and cf > 0:
                        d_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                        dates.append(d_str)
                        closes.append(cf)
        elif isinstance(raw_data, dict):
            for d, c in raw_data.items():
                cf = _clean_float(c)
                if cf is not None and cf > 0:
                    dates.append(str(d)[:10])
                    closes.append(cf)
        elif isinstance(raw_data, (list, tuple)):
            for item in raw_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    cf = _clean_float(item[1])
                    if cf is not None and cf > 0:
                        dates.append(str(item[0])[:10])
                        closes.append(cf)
                elif isinstance(item, dict) and "date" in item and "close" in item:
                    cf = _clean_float(item["close"])
                    if cf is not None and cf > 0:
                        dates.append(str(item["date"])[:10])
                        closes.append(cf)

        if len(closes) < 2:
            unavail.append(ticker)
            continue

        # 依日期遞增排序
        combined = sorted(zip(dates, closes), key=lambda x: x[0])
        # 去除重複日期 (保留最新)
        dedup_dict = {d: c for d, c in combined}
        sorted_pairs = sorted(dedup_dict.items(), key=lambda x: x[0])

        mode = adj_modes.get(ticker)
        if not mode:
            mode = "raw_close" if (ticker.endswith(".TW") or ticker.endswith(".TWO") or ticker.isdigit()) else "adjusted_close"
        threshold = jump_threshold_tw if mode == "raw_close" else jump_threshold_foreign

        ticker_rets: Dict[str, float] = {}
        for idx in range(1, len(sorted_pairs)):
            prev_d, prev_c = sorted_pairs[idx - 1]
            curr_d, curr_c = sorted_pairs[idx]
            ret = (curr_c - prev_c) / prev_c
            if not math.isfinite(ret):
                continue

            # 企業行為異常單日跳空檢測
            if abs(ret) > threshold:
                corporate_action_suspects.append({
                    "ticker": ticker,
                    "date": curr_d,
                    "return": round(float(ret), 4),
                    "raw_return": round(float(ret), 4),
                    "return_pct": f"{ret * 100:.2f}%",
                    "threshold": threshold,
                    "price_adjustment_mode": mode,
                })
                suspect_dates.add(curr_d)
            else:
                ticker_rets[curr_d] = float(ret)

        if ticker_rets:
            returns_by_ticker[ticker] = ticker_rets
        else:
            unavail.append(ticker)

    # 剔除企業行為跳空異常日
    if suspect_dates:
        for t_rets in returns_by_ticker.values():
            for s_date in suspect_dates:
                t_rets.pop(s_date, None)

    valid_tickers = [tk for tk in returns_by_ticker.keys() if len(returns_by_ticker[tk]) > 0]

    # 判斷多幣別模式與警告
    all_currencies = set()
    for tk in valid_tickers:
        c = curr_map.get(tk)
        if not c:
            c = "TWD" if (tk.endswith(".TW") or tk.endswith(".TWO") or tk.isdigit()) else ("JPY" if tk.endswith(".T") else "USD")
        all_currencies.add(c)

    if len(all_currencies) > 1:
        currency_mode = "local_currency_return_proxy"
        fx_unhedged_warning = "投資組合包含跨市場多幣別標的，日報酬以各標的本地計價幣別直接加權計算，未計入匯率波動風險。"
        calendar_mismatch_warning = "各市場交易日曆存在差異（國定假日不同步），部分交易日跨度可能超過單一自然交易日。"
    else:
        currency_mode = list(all_currencies)[0] if all_currencies else "TWD"
        fx_unhedged_warning = None
        calendar_mismatch_warning = None

    if not valid_tickers:
        res: Dict[str, Any] = {
            "status": "insufficient_history",
            "sample_count": 0,
            "effective_bars": 0,
            "annualized_volatility_total": None,
            "annualized_volatility_risk_sleeve": None,
            "hhi_risk_sleeve": None,
            "effective_n_risk_sleeve": None,
            "weights": {},
            "var_95": None,
            "cvar_95": None,
            "var_99": None,
            "cvar_99": None,
            "var_95_1d": None,
            "var_95_20d": None,
            "cvar_95_1d": None,
            "var_99_1d": None,
            "var_99_20d": None,
            "cvar_99_1d": None,
            "tail_95_status": "insufficient_tail_samples",
            "tail_99_status": "insufficient_tail_samples",
            "tail_95_sample_count": None,
            "tail_99_sample_count": None,
            "euler_contributions": [],
            "assets": [],
            "high_correlation_pairs": [],
            "corporate_action_suspects": corporate_action_suspects,
            "currency_mode": currency_mode,
            "fx_unhedged_warning": fx_unhedged_warning,
            "calendar_mismatch_warning": calendar_mismatch_warning,
            "available_tickers": [],
            "unavailable_tickers": sorted(list(set(unavail))),
            "cash_weight": 1.0,
        }
        json.dumps(res, allow_nan=False)
        return res

    # 2. 取所有有效標的之共同交易日交集
    common_dates = set.intersection(*(set(returns_by_ticker[tk].keys()) for tk in valid_tickers))
    sorted_common_dates = sorted(list(common_dates))
    n_samples = len(sorted_common_dates)

    # 3. 處理權重與現金口徑
    k_assets = len(valid_tickers)
    total_weights: Dict[str, float] = {}
    if weights:
        w_sum = 0.0
        for tk in valid_tickers:
            raw_w = _clean_float(weights.get(tk, 0.0)) or 0.0
            w_clamped = max(0.0, raw_w)
            total_weights[tk] = w_clamped
            w_sum += w_clamped

        if w_sum > 1.0 + 1e-6:
            # 正規化至 1.0
            total_weights = {tk: total_weights[tk] / w_sum for tk in valid_tickers}
            w_sum = 1.0
        w_cash = max(0.0, round(1.0 - w_sum, 6))
    else:
        # 預設等權重
        eq_w = 1.0 / k_assets
        total_weights = {tk: eq_w for tk in valid_tickers}
        w_cash = 0.0

    w_risk = sum(total_weights[tk] for tk in valid_tickers)
    # 風險槽正規化權重 (Risk Sleeve)
    if w_risk > 1e-12:
        sleeve_weights = {tk: total_weights[tk] / w_risk for tk in valid_tickers}
    else:
        sleeve_weights = {tk: 1.0 / k_assets for tk in valid_tickers}

    hhi_risk_sleeve = sum(sleeve_weights[tk] ** 2 for tk in valid_tickers)

    # 若交集樣本數不足階梯最低門檻 N < 30
    if n_samples < 30:
        eff_n = round(float(1.0 / hhi_risk_sleeve), 4) if (hhi_risk_sleeve and math.isfinite(hhi_risk_sleeve) and hhi_risk_sleeve > 1e-12) else None
        res = {
            "status": "insufficient_history",
            "sample_count": n_samples,
            "effective_bars": n_samples,
            "annualized_volatility_total": None,
            "annualized_volatility_risk_sleeve": None,
            "hhi_risk_sleeve": round(float(hhi_risk_sleeve), 6) if math.isfinite(hhi_risk_sleeve) else None,
            "effective_n_risk_sleeve": eff_n,
            "weights": {tk: round(float(total_weights[tk]), 6) for tk in valid_tickers},
            "var_95": None,
            "cvar_95": None,
            "var_99": None,
            "cvar_99": None,
            "var_95_1d": None,
            "var_95_20d": None,
            "cvar_95_1d": None,
            "var_99_1d": None,
            "var_99_20d": None,
            "cvar_99_1d": None,
            "tail_95_status": "insufficient_tail_samples",
            "tail_99_status": "insufficient_tail_samples",
            "tail_95_sample_count": None,
            "tail_99_sample_count": None,
            "euler_contributions": [],
            "assets": [],
            "high_correlation_pairs": [],
            "corporate_action_suspects": corporate_action_suspects,
            "currency_mode": currency_mode,
            "fx_unhedged_warning": fx_unhedged_warning,
            "calendar_mismatch_warning": calendar_mismatch_warning,
            "available_tickers": valid_tickers,
            "unavailable_tickers": sorted(list(set(unavail))),
            "cash_weight": round(float(w_cash), 6),
        }
        json.dumps(res, allow_nan=False)
        return res

    # 4. 建構共同報酬矩陣 (N x K)
    ret_matrix = np.zeros((n_samples, k_assets), dtype=np.float64)
    for col_idx, tk in enumerate(valid_tickers):
        for row_idx, d in enumerate(sorted_common_dates):
            ret_matrix[row_idx, col_idx] = returns_by_ticker[tk][d]

    # 共變異數矩陣 (K x K, ddof=1)
    if n_samples > 1:
        cov_matrix = np.cov(ret_matrix, rowvar=False, ddof=1)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
    else:
        cov_matrix = np.zeros((k_assets, k_assets), dtype=np.float64)

    w_vec = np.array([total_weights[tk] for tk in valid_tickers], dtype=np.float64)
    w_sleeve_vec = np.array([sleeve_weights[tk] for tk in valid_tickers], dtype=np.float64)

    # 組合日報酬序列 (全組合含現金報酬 R_t = sum(w_i * r_{i,t}))
    portfolio_daily_rets = ret_matrix @ w_vec
    losses = -portfolio_daily_rets

    # 雙口徑波動率
    port_var = float(w_vec.T @ cov_matrix @ w_vec)
    ann_vol_total = math.sqrt(max(0.0, 252.0 * port_var))

    sleeve_var = float(w_sleeve_vec.T @ cov_matrix @ w_sleeve_vec)
    ann_vol_sleeve = math.sqrt(max(0.0, 252.0 * sleeve_var))

    # 5. 歐拉波動率拆解 (Euler Component Volatility & Percentage Contribution)
    euler_contributions: List[Dict[str, Any]] = []
    if ann_vol_total > 1e-12:
        cov_w = cov_matrix @ w_vec
        cov_w_sleeve = cov_matrix @ w_sleeve_vec
        for idx, tk in enumerate(valid_tickers):
            ann_vol_i = math.sqrt(max(0.0, 252.0 * float(cov_matrix[idx, idx])))
            cv_i = float(252.0 * w_vec[idx] * cov_w[idx] / ann_vol_total)
            mc_i = float(252.0 * cov_w[idx] / ann_vol_total)
            if sleeve_var > 1e-12:
                pcv_i = float(w_sleeve_vec[idx] * cov_w_sleeve[idx] / sleeve_var)
            else:
                pcv_i = float(1.0 / k_assets)

            euler_contributions.append({
                "ticker": tk,
                "weight": round(float(w_vec[idx]), 6),
                "weight_total": round(float(w_vec[idx]), 6),
                "weight_sleeve": round(float(w_sleeve_vec[idx]), 6),
                "annualized_volatility": round(float(ann_vol_i), 6),
                "marginal_contribution": round(float(mc_i), 6),
                "component_volatility": round(cv_i, 6),
                "percentage_contribution": round(pcv_i, 6),
                "is_hedge": bool(pcv_i < 0),
            })
    else:
        for idx, tk in enumerate(valid_tickers):
            ann_vol_i = math.sqrt(max(0.0, 252.0 * float(cov_matrix[idx, idx])))
            euler_contributions.append({
                "ticker": tk,
                "weight": round(float(w_vec[idx]), 6),
                "weight_total": round(float(w_vec[idx]), 6),
                "weight_sleeve": round(float(w_sleeve_vec[idx]), 6),
                "annualized_volatility": round(float(ann_vol_i), 6),
                "marginal_contribution": 0.0,
                "component_volatility": 0.0,
                "percentage_contribution": round(float(1.0 / k_assets), 6),
                "is_hedge": False,
            })

    # 6. 高相關配對預警 (|corr| >= 0.75)
    high_corr_pairs: List[Dict[str, Any]] = []
    if k_assets > 1:
        stds = np.sqrt(np.diag(cov_matrix))
        std_outer = np.outer(stds, stds)
        corr_matrix = np.divide(cov_matrix, std_outer, out=np.zeros_like(cov_matrix), where=std_outer > 1e-12)

        for i in range(k_assets):
            for j in range(i + 1, k_assets):
                c_val = float(corr_matrix[i, j])
                if math.isfinite(c_val) and abs(c_val) >= 0.75:
                    high_corr_pairs.append({
                        "ticker_a": valid_tickers[i],
                        "ticker_b": valid_tickers[j],
                        "correlation": round(c_val, 4),
                    })
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # 7. 分位數階梯樣本門檻 VaR / CVaR
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_99: Optional[float] = None
    tail_95_status = "insufficient_tail_samples"
    tail_99_status = "insufficient_tail_samples"
    tail_95_samples: Optional[int] = None
    tail_99_samples: Optional[int] = None

    if n_samples >= 60:
        tail_95_status = "ok"
        tail_95_samples = n_samples
        v95 = float(np.percentile(losses, 95))
        tail_95_losses = losses[losses >= v95]
        cv95 = float(np.mean(tail_95_losses)) if len(tail_95_losses) > 0 else v95
        var_95 = round(v95, 6)
        cvar_95 = round(cv95, 6)

    if n_samples >= 100:
        tail_99_status = "ok"
        tail_99_samples = n_samples
        v99 = float(np.percentile(losses, 99))
        tail_99_losses = losses[losses >= v99]
        cv99 = float(np.mean(tail_99_losses)) if len(tail_99_losses) > 0 else v99
        var_99 = round(v99, 6)
        cvar_99 = round(cv99, 6)

    eff_n = round(float(1.0 / hhi_risk_sleeve), 4) if (hhi_risk_sleeve and math.isfinite(hhi_risk_sleeve) and hhi_risk_sleeve > 1e-12) else None
    var_95_20d = round(float(var_95 * math.sqrt(20.0)), 6) if var_95 is not None else None
    var_99_20d = round(float(var_99 * math.sqrt(20.0)), 6) if var_99 is not None else None

    result = {
        "status": "ok",
        "sample_count": n_samples,
        "effective_bars": n_samples,
        "annualized_volatility_total": round(ann_vol_total, 6),
        "annualized_volatility_risk_sleeve": round(ann_vol_sleeve, 6),
        "hhi_risk_sleeve": round(float(hhi_risk_sleeve), 6),
        "effective_n_risk_sleeve": eff_n,
        "weights": {tk: round(float(total_weights[tk]), 6) for tk in valid_tickers},
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
        "var_95_1d": var_95,
        "var_95_20d": var_95_20d,
        "cvar_95_1d": cvar_95,
        "var_99_1d": var_99,
        "var_99_20d": var_99_20d,
        "cvar_99_1d": cvar_99,
        "tail_95_status": tail_95_status,
        "tail_99_status": tail_99_status,
        "tail_95_sample_count": tail_95_samples,
        "tail_99_sample_count": tail_99_samples,
        "euler_contributions": euler_contributions,
        "assets": euler_contributions,
        "high_correlation_pairs": high_corr_pairs,
        "corporate_action_suspects": corporate_action_suspects,
        "currency_mode": currency_mode,
        "fx_unhedged_warning": fx_unhedged_warning,
        "calendar_mismatch_warning": calendar_mismatch_warning,
        "available_tickers": valid_tickers,
        "unavailable_tickers": sorted(list(set(unavail))),
        "cash_weight": round(float(w_cash), 6),
    }

    # 強制驗證 100% 通過 allow_nan=False
    json.dumps(result, allow_nan=False)
    return result
