from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from strategies.base import StrategyResult
from data.macro import MacroState


@dataclass
class EvaluationReport:
    """Standardized composite analysis report across all strategies for an index"""
    index_name: str
    strategy_results: Dict[str, List[StrategyResult]]  # strategy_key -> list of results
    overlap_candidates: List[Dict[str, Any]] = field(default_factory=list)
    ranked_stocks: List[Dict[str, Any]] = field(default_factory=list)
    xuantie_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    timestral_results: List[Dict[str, Any]] = field(default_factory=list)
    timesfm_results: List[Dict[str, Any]] = field(default_factory=list)
    moirai_results: List[Dict[str, Any]] = field(default_factory=list)
    lstm_results: List[Dict[str, Any]] = field(default_factory=list)
    overlap_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_state: Optional[MacroState] = None
    ai_summary: str = ""


class CompositeEvaluator:
    """Evaluator that combines multiple strategy signals, dual foundation models, institutional flows, and macro gates"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_overlap_count: int = 2
    ):
        self.weights = weights or {
            "xuantie": 0.30,
            "timesfm": 0.20,
            "timestral": 0.20,
            "moirai": 0.15,
            "institutional": 0.15,
            "sector": 0.10,
            "fundamental": 0.05
        }
        self.min_overlap_count = min_overlap_count

    def evaluate(
        self,
        index_name: str,
        strategy_outputs: Dict[str, List[StrategyResult]],
        fundamentals_map: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
        macro_state: Optional[MacroState] = None
    ) -> EvaluationReport:
        """
        Evaluate all strategy outputs for an index, computing composite scores, dual-FM consensus, and overlap.
        """
        fundamentals_map = fundamentals_map or {}

        # 1. Index strategy results by ticker
        ticker_strat_map: Dict[str, Dict[str, StrategyResult]] = {}
        all_tickers = set()

        for strat_name, res_list in strategy_outputs.items():
            for res in res_list:
                if res.ticker not in ticker_strat_map:
                    ticker_strat_map[res.ticker] = {}
                ticker_strat_map[res.ticker][strat_name] = res
                all_tickers.add(res.ticker)

        # 2. Build Legacy DataFrames / Lists for backward compatibility
        # XuanTie legacy df
        xuantie_hits = []
        if "xuantie" in strategy_outputs:
            for r in strategy_outputs["xuantie"]:
                if r.is_hit:
                    xuantie_hits.append({
                        "ticker": r.ticker,
                        "current_price": r.current_price,
                        "signal": True,
                        "major_trend": r.signals.get("major_trend", True),
                        "pullback": r.signals.get("pullback", True),
                        "pullback_type": r.signals.get("pullback_type", ""),
                        "ma5": r.metrics.get("ma5"),
                        "ma10": r.metrics.get("ma10"),
                        "ma60": r.metrics.get("ma60"),
                        "ma120": r.metrics.get("ma120"),
                        "ma250": r.metrics.get("ma250"),
                        "pe": r.metrics.get("pe"),
                        "pb": r.metrics.get("pb"),
                        "forward_pe": r.metrics.get("forward_pe"),
                        "ev_ebitda": r.metrics.get("ev_ebitda")
                    })
        xuantie_df = pd.DataFrame(xuantie_hits) if xuantie_hits else pd.DataFrame()

        # Timestral list (Mistral AI 209M)
        timestral_results_list = []
        if "timestral" in strategy_outputs:
            for r in strategy_outputs["timestral"]:
                if r.potential is not None:
                    timestral_results_list.append({
                        "ticker": r.ticker,
                        "potential": r.potential,
                        "current_price": r.current_price,
                        "predicted_price": r.predicted_price or r.current_price,
                        "horizon_predicted_price": r.signals.get("horizon_predicted_price"),
                        "risk_reward_ratio": r.signals.get("risk_reward_ratio"),
                        "quantiles": r.signals.get("quantiles", {}),
                        "p10": r.metrics.get("p10"),
                        "p50": r.metrics.get("p50"),
                        "p90": r.metrics.get("p90"),
                        "pe": r.metrics.get("pe"),
                        "pb": r.metrics.get("pb"),
                        "forward_pe": r.metrics.get("forward_pe"),
                        "ev_ebitda": r.metrics.get("ev_ebitda")
                    })
            timestral_results_list.sort(key=lambda x: x["potential"], reverse=True)

        # TimesFM list (Google Research 200M)
        timesfm_results_list = []
        if "timesfm" in strategy_outputs:
            for r in strategy_outputs["timesfm"]:
                if r.potential is not None:
                    timesfm_results_list.append({
                        "ticker": r.ticker,
                        "potential": r.potential,
                        "current_price": r.current_price,
                        "predicted_price": r.predicted_price or r.current_price,
                        "horizon_predicted_price": r.signals.get("horizon_predicted_price"),
                        "risk_reward_ratio": r.signals.get("risk_reward_ratio"),
                        "p10": r.metrics.get("p10"),
                        "p50": r.metrics.get("p50"),
                        "p90": r.metrics.get("p90"),
                        "pe": r.metrics.get("pe"),
                        "pb": r.metrics.get("pb"),
                        "forward_pe": r.metrics.get("forward_pe"),
                        "ev_ebitda": r.metrics.get("ev_ebitda")
                    })
            timesfm_results_list.sort(key=lambda x: x["potential"], reverse=True)

        # MOIRAI-2 list (Salesforce Foundation Model)
        moirai_results_list = []
        if "moirai" in strategy_outputs:
            for r in strategy_outputs["moirai"]:
                if r.potential is not None:
                    moirai_results_list.append({
                        "ticker": r.ticker,
                        "potential": r.potential,
                        "current_price": r.current_price,
                        "predicted_price": r.predicted_price or r.current_price,
                        "horizon_predicted_price": r.signals.get("horizon_predicted_price"),
                        "risk_reward_ratio": r.signals.get("risk_reward_ratio"),
                        "p10": r.metrics.get("p10"),
                        "p50": r.metrics.get("p50"),
                        "p90": r.metrics.get("p90"),
                        "pe": r.metrics.get("pe"),
                        "pb": r.metrics.get("pb"),
                        "forward_pe": r.metrics.get("forward_pe"),
                        "ev_ebitda": r.metrics.get("ev_ebitda")
                    })
            moirai_results_list.sort(key=lambda x: x["potential"], reverse=True)

        # LSTM legacy list (deprecated fallback)
        lstm_results_legacy = []
        if "lstm" in strategy_outputs:
            for r in strategy_outputs["lstm"]:
                if r.potential is not None:
                    lstm_results_legacy.append({
                        "ticker": r.ticker,
                        "potential": r.potential,
                        "current_price": r.current_price,
                        "predicted_price": r.predicted_price or r.current_price,
                        "pe": r.metrics.get("pe"),
                        "pb": r.metrics.get("pb"),
                        "forward_pe": r.metrics.get("forward_pe"),
                        "ev_ebitda": r.metrics.get("ev_ebitda")
                    })
            lstm_results_legacy.sort(key=lambda x: x["potential"], reverse=True)

        # Dynamic active weights matching available strategies
        active_weights = dict(self.weights)
        if "timestral" in strategy_outputs and "timestral" not in active_weights:
            active_weights["timestral"] = 0.20
        if "timesfm" in strategy_outputs and "timesfm" not in active_weights:
            active_weights["timesfm"] = 0.20
        if "moirai" in strategy_outputs and "moirai" not in active_weights:
            active_weights["moirai"] = 0.15
        if "lstm" in strategy_outputs and ("timestral" in strategy_outputs or "timesfm" in strategy_outputs or "moirai" in strategy_outputs):
            # When foundation models are present, deprecate LSTM weight to 0 or trace
            active_weights["lstm"] = 0.0

        def eligible(name: str, result: StrategyResult) -> bool:
            if name not in {"timesfm", "timestral", "moirai"}:
                return True
            meta = result.metadata or {}
            return (meta.get("role", "active") in {"active", "primary"}
                    and meta.get("status", "ready") in {"ready", "active"}
                    and meta.get("health_state") != "suspended"
                    and meta.get("effective_weight", 1.0) > 0)

        # 3. Evaluate each stock across all strategies
        overlap_candidates = []
        ranked_stocks = []

        for ticker in all_tickers:
            original_strats = ticker_strat_map[ticker]
            strats = {name: value for name, value in original_strats.items() if eligible(name, value)}
            fund = fundamentals_map.get(ticker, {})

            # Count hits and gather tags
            hits = [s for s in strats.values() if s.is_hit]
            hit_count = len(hits)

            combined_tags = []
            for s in hits:
                combined_tags.extend(s.tags)

            # Fundamental valuation bonus / tags
            pe_val = fund.get("pe")
            pb_val = fund.get("pb")
            fund_score = 0.0

            if pe_val is not None and pe_val > 0:
                if pe_val < 20.0:
                    fund_score += 10.0
                    combined_tags.append("低PE")
                elif pe_val < 30.0:
                    fund_score += 5.0

            if pb_val is not None and pb_val > 0:
                if pb_val < 3.0:
                    fund_score += 10.0
                    combined_tags.append("低PB")
                elif pb_val < 5.0:
                    fund_score += 5.0

            # Check specific strategies
            xuantie_res = strats.get("xuantie")
            timestral_res = strats.get("timestral")
            timesfm_res = strats.get("timesfm")
            moirai_res = strats.get("moirai")
            lstm_res = strats.get("lstm")
            inst_res = strats.get("institutional")
            sector_res = strats.get("sector_rotation") or strats.get("sector")

            if xuantie_res and xuantie_res.is_hit:
                combined_tags.append("季線支撐")

            # Dynamic Model Plugin Tags (零硬編碼，動態遍歷 ModelPluginRegistry)
            from models.protocol import model_registry
            active_plugins = model_registry.list_active_plugins()
            bullish_model_names = []

            for p in active_plugins:
                p_id = getattr(p, "plugin_id", None) or getattr(p, "tag_name", "").lower() or p.model_id.split("/")[-1].split("-")[0].lower()
                p_res = strats.get(p_id)
                if p_res and p_res.is_hit:
                    pot_val = float(p_res.potential) if p_res.potential is not None else 0.0
                    rr_val = float(p_res.signals.get("risk_reward_ratio", 0)) if (p_res.signals and isinstance(p_res.signals, dict)) else 0.0
                    tags = p.generate_tags(pot_val, rr_val)
                    combined_tags.extend(tags)
                    if pot_val > 0:
                        disp_name = getattr(p, "tag_name", None) or getattr(p, "display_name", p_id)
                        if "(" in disp_name:
                            disp_name = disp_name.split("(")[0].strip()
                        bullish_model_names.append(disp_name)
                    if rr_val >= 1.5:
                        combined_tags.append("高風報比")
                    elif rr_val >= 1.0:
                        combined_tags.append("高盈虧比")

            if lstm_res and lstm_res.is_hit and not timestral_res:
                combined_tags.append("LSTM看多")
                if lstm_res.potential and lstm_res.potential > 0:
                    bullish_model_names.append("LSTM")

            # Three Foundation Models Consensus (TimesFM × Timestral × MOIRAI-2)
            hit_timestral = bool(timestral_res and timestral_res.is_hit)
            hit_timesfm = bool(timesfm_res and timesfm_res.is_hit)
            hit_moirai = bool(moirai_res and moirai_res.is_hit)
            fm_hits = [h for h in (hit_timestral, hit_timesfm, hit_moirai) if h]
            hit_tri_fm = len(fm_hits) >= 3
            hit_both_fm = len(fm_hits) >= 2
            hit_any_fm = len(fm_hits) >= 1

            # Legacy ML hit indicator (for backward compatibility if only LSTM was run)
            hit_lstm = bool(lstm_res and lstm_res.is_hit)
            hit_both_ml = hit_both_fm or (hit_lstm and (hit_timesfm or hit_timestral or hit_moirai))
            hit_any_ml = hit_any_fm or hit_lstm

            if inst_res:
                meta = inst_res.metadata or {}
                if meta.get("is_sync_buy"):
                    combined_tags.append("土洋合買")
                if meta.get("is_trust_streak"):
                    combined_tags.append(f"投信連買{meta.get('trust_streak', '')}天")
                elif meta.get("trust_net_5d", 0) > 0:
                    combined_tags.append("投信買超")
            if sector_res:
                sec_meta = sector_res.metadata or {}
                if sec_meta.get("is_top_sector"):
                    combined_tags.append(f"主流板塊({sec_meta.get('sector', '')})")

            hit_xuantie = bool(xuantie_res and xuantie_res.is_hit)
            hit_inst = bool(inst_res and inst_res.is_hit)
            hit_sector = bool(sector_res and sector_res.is_hit)

            # Quadruple / Multi-factor Confluence
            is_quad_resonance = hit_xuantie and hit_inst and (hit_both_fm or hit_both_ml)
            is_triple_resonance = False
            if not is_quad_resonance:
                if hit_xuantie and hit_inst and hit_any_ml:
                    is_triple_resonance = True
                elif hit_xuantie and hit_both_ml:
                    is_triple_resonance = True
                elif hit_inst and hit_both_ml:
                    is_triple_resonance = True
                elif hit_sector and ((hit_xuantie and hit_inst) or ((hit_xuantie or hit_inst) and hit_any_ml)):
                    is_triple_resonance = True

            # Foundation Model Potentials & Positive Gate Evaluation
            ml_potentials = []
            if timestral_res and timestral_res.potential is not None and np.isfinite(timestral_res.potential):
                ml_potentials.append(float(timestral_res.potential))
            if timesfm_res and timesfm_res.potential is not None and np.isfinite(timesfm_res.potential):
                ml_potentials.append(float(timesfm_res.potential))
            if moirai_res and moirai_res.potential is not None and np.isfinite(moirai_res.potential):
                ml_potentials.append(float(moirai_res.potential))
            if not ml_potentials and lstm_res and lstm_res.potential is not None and np.isfinite(lstm_res.potential):
                ml_potentials.append(float(lstm_res.potential))

            has_ml = len(ml_potentials) > 0
            avg_ml_pot = sum(ml_potentials) / len(ml_potentials) if has_ml else 0.0

            # Bearish check (防守門檻):
            is_ml_bearish = False
            if has_ml:
                if all(p <= 0.0 for p in ml_potentials):
                    is_ml_bearish = True
                elif len(ml_potentials) >= 2:
                    if avg_ml_pot <= 0.0 or min(ml_potentials) <= -5.0:
                        is_ml_bearish = True
                elif len(ml_potentials) == 1 and ml_potentials[0] <= 0.0:
                    is_ml_bearish = True

            # Must have at least one primary directional signal (Technical buy or FM bullish hit)
            has_primary_signal = bool(
                (xuantie_res and xuantie_res.is_hit) or
                (timestral_res and timestral_res.is_hit) or
                (timesfm_res and timesfm_res.is_hit) or
                (moirai_res and moirai_res.is_hit) or
                (lstm_res and lstm_res.is_hit)
            )

            # Defensive tags for bearish stocks (純文字無 emoji)
            if is_ml_bearish:
                if any(p <= -10.0 for p in ml_potentials):
                    combined_tags.append("深度防守")
                else:
                    combined_tags.append("防守")

            # Determine Concurring Models Tier
            resonance_tier = ", ".join(bullish_model_names) if (bullish_model_names and not is_ml_bearish) else ""

            # Calculate Weighted Composite Score (0~100)
            score_total = 0.0
            weight_total = 0.0

            for s_name, s_weight in active_weights.items():
                if s_name == "fundamental":
                    score_total += fund_score * s_weight * 5.0
                    weight_total += s_weight
                elif s_name in strats:
                    score_total += strats[s_name].score * s_weight
                    weight_total += s_weight
                elif s_name == "sector" and "sector_rotation" in strats:
                    score_total += strats["sector_rotation"].score * s_weight
                    weight_total += s_weight

            composite_score = round(score_total / weight_total, 2) if weight_total > 0 else 0.0

            # Macro Exposure Multiplier Discount
            if macro_state and macro_state.exposure is not None and macro_state.exposure < 1.0:
                composite_score = round(composite_score * macro_state.exposure, 2)

            # Bearish ML Penalty on composite score
            if is_ml_bearish:
                composite_score = round(composite_score * 0.5, 2)

            # Current price resolution with defensive empty fallback
            curr_price = next((s.current_price for s in strats.values() if getattr(s, "current_price", None) is not None),
                              next((s.current_price for s in original_strats.values() if getattr(s, "current_price", None) is not None), 0.0))

            # Target Price & Potential Defensive Synthesis:
            # Weighted ensemble of foundation models based on active strategy weights
            fm_weights = {
                "timesfm": active_weights.get("timesfm", 0.20),
                "timestral": active_weights.get("timestral", 0.20),
                "moirai": active_weights.get("moirai", 0.15),
            }
            valid_weights_and_prices = []
            for name, res in [("timesfm", timesfm_res), ("timestral", timestral_res), ("moirai", moirai_res)]:
                if res and res.predicted_price is not None and np.isfinite(res.predicted_price):
                    valid_weights_and_prices.append((fm_weights.get(name, 0.20), float(res.predicted_price)))

            if valid_weights_and_prices:
                total_w = sum(w for w, _ in valid_weights_and_prices)
                synthesized_target = round(sum(w * p for w, p in valid_weights_and_prices) / total_w, 2)
            elif lstm_res and lstm_res.predicted_price is not None and np.isfinite(lstm_res.predicted_price):
                synthesized_target = round(float(lstm_res.predicted_price), 2)
            else:
                synthesized_target = None

            if curr_price > 0 and synthesized_target is not None:
                synthesized_potential = round(((synthesized_target - curr_price) / curr_price) * 100.0, 2)
            else:
                synthesized_potential = None

            # Risk-reward resolution across available foundation models
            best_rr = None
            for fm_res in (timestral_res, timesfm_res, moirai_res):
                if fm_res and fm_res.signals.get("risk_reward_ratio") is not None:
                    best_rr = max(best_rr or 0.0, float(fm_res.signals.get("risk_reward_ratio")))

            # Deduplicate tags
            cleaned_tags = [t for t in combined_tags if not any(k in t for k in ("共振", "重符合", "👑", "🏆", "🔮", "⭐", "🌟"))]
            if is_ml_bearish:
                cleaned_tags = [t for t in cleaned_tags if "看多" not in t and "強" not in t and "符合" not in t]
            final_tags = list(dict.fromkeys(cleaned_tags))

            # Confluence Overlap Condition
            is_resonance = (
                is_quad_resonance or
                is_triple_resonance or
                hit_both_fm or
                hit_both_ml or
                (hit_count >= self.min_overlap_count and has_primary_signal)
            )

            is_valid_overlap = is_resonance and (not is_ml_bearish)

            entry = {
                "ticker": ticker,
                "composite_score": composite_score,
                "current_price": curr_price,
                "predicted_price": synthesized_target,
                "potential": synthesized_potential,
                "hit_count": hit_count,
                "hit_strategies": [s.strategy_name for s in hits],
                "tags": final_tags,
                "fundamentals": fund,
                "resonance_tier": resonance_tier,
                "is_ml_bearish": is_ml_bearish,
                "avg_ml_pot": avg_ml_pot,
                "dual_model_consensus": hit_both_fm,
                "tri_model_consensus": hit_tri_fm,
                "risk_reward_ratio": best_rr,
            }

            if xuantie_res:
                entry["ma60"] = xuantie_res.metrics.get("ma60")
                entry["pullback_type"] = xuantie_res.signals.get("pullback_type", "")
            if timestral_res:
                entry["timestral_potential"] = timestral_res.potential
                entry["timestral_predicted_price"] = timestral_res.predicted_price
                entry["timestral_p10"] = timestral_res.metrics.get("p10")
                entry["timestral_p90"] = timestral_res.metrics.get("p90")
            if timesfm_res:
                entry["timesfm_potential"] = timesfm_res.potential
                entry["timesfm_predicted_price"] = timesfm_res.predicted_price
            if moirai_res:
                entry["moirai_potential"] = moirai_res.potential
                entry["moirai_predicted_price"] = moirai_res.predicted_price
                entry["moirai_p10"] = moirai_res.metrics.get("p10")
                entry["moirai_p90"] = moirai_res.metrics.get("p90")
            if lstm_res:
                entry["lstm_potential"] = lstm_res.potential
            if inst_res:
                entry["institutional"] = inst_res.metadata

            ranked_stocks.append(entry)

            if is_valid_overlap:
                overlap_candidates.append(entry)

        # Dynamic priority sort based purely on active concurring models & quantitative factors
        def candidate_sort_key(c: Dict[str, Any]):
            tags = c.get("tags", [])
            model_mult = sum(1 for t in tags if t.endswith("看多"))
            tech_bonus = 50.0 if any("季線" in t for t in tags) else 0.0
            inst_bonus = 50.0 if any("合買" in t or "法人" in t or "連買" in t for t in tags) else 0.0
            tier_val = model_mult * 100.0 + tech_bonus + inst_bonus
            score_val = c.get("composite_score", 0.0)
            pot_val = max(0.0, c.get("avg_ml_pot", 0.0))
            rr_val = c.get("risk_reward_ratio") or 1.0
            return (tier_val, score_val, pot_val, rr_val)

        overlap_candidates.sort(key=candidate_sort_key, reverse=True)
        ranked_stocks.sort(key=lambda x: x["composite_score"], reverse=True)

        # Build legacy overlap_df
        overlap_legacy_rows = []
        for o in overlap_candidates:
            overlap_legacy_rows.append({
                "ticker": o["ticker"],
                "potential": o.get("potential", 0.0),
                "predicted_price": o.get("predicted_price", o["current_price"]),
                "current_price": o["current_price"],
                "timestral_potential": o.get("timestral_potential"),
                "timestral_predicted_price": o.get("timestral_predicted_price"),
                "timestral_p10": o.get("timestral_p10"),
                "timestral_p90": o.get("timestral_p90"),
                "timesfm_potential": o.get("timesfm_potential"),
                "timesfm_predicted_price": o.get("timesfm_predicted_price"),
                "moirai_potential": o.get("moirai_potential"),
                "moirai_predicted_price": o.get("moirai_predicted_price"),
                "moirai_p10": o.get("moirai_p10"),
                "moirai_p90": o.get("moirai_p90"),
                "dual_model_consensus": o.get("dual_model_consensus", False),
                "tri_model_consensus": o.get("tri_model_consensus", False),
                "lstm_potential": o.get("lstm_potential", 0.0),
                "risk_reward_ratio": o.get("risk_reward_ratio"),
                "pullback_type": o.get("pullback_type", ""),
                "ma60": o.get("ma60"),
                "pe": o["fundamentals"].get("pe"),
                "pb": o["fundamentals"].get("pb"),
                "forward_pe": o["fundamentals"].get("forward_pe"),
                "ev_ebitda": o["fundamentals"].get("ev_ebitda"),
                "composite_score": o.get("composite_score", 0.0),
                "tags": o.get("tags", []),
                "resonance_tier": o.get("resonance_tier", ""),
                "hit_strategies": o.get("hit_strategies", [])
            })
        overlap_df = pd.DataFrame(overlap_legacy_rows) if overlap_legacy_rows else pd.DataFrame()

        return EvaluationReport(
            index_name=index_name,
            strategy_results=strategy_outputs,
            overlap_candidates=overlap_candidates,
            ranked_stocks=ranked_stocks,
            xuantie_results=xuantie_df,
            timestral_results=timestral_results_list,
            timesfm_results=timesfm_results_list,
            moirai_results=moirai_results_list,
            lstm_results=lstm_results_legacy,
            overlap_results=overlap_df,
            macro_state=macro_state
        )
