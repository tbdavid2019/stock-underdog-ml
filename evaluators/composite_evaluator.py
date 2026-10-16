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
            "xuantie": 0.35,
            "timesfm": 0.25,
            "timestral": 0.25,
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
                        "pe": r.metrics.get("pe"),
                        "pb": r.metrics.get("pb"),
                        "forward_pe": r.metrics.get("forward_pe"),
                        "ev_ebitda": r.metrics.get("ev_ebitda")
                    })
            timesfm_results_list.sort(key=lambda x: x["potential"], reverse=True)

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
            active_weights["timestral"] = 0.25
        if "timesfm" in strategy_outputs and "timesfm" not in active_weights:
            active_weights["timesfm"] = 0.25
        if "lstm" in strategy_outputs and "timestral" in strategy_outputs:
            # When both Timestral and LSTM are present, deprecate LSTM weight to 0 or trace
            active_weights["lstm"] = 0.0

        # 3. Evaluate each stock across all strategies
        overlap_candidates = []
        ranked_stocks = []

        for ticker in all_tickers:
            strats = ticker_strat_map[ticker]
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
            lstm_res = strats.get("lstm")
            inst_res = strats.get("institutional")
            sector_res = strats.get("sector_rotation") or strats.get("sector")

            if xuantie_res and xuantie_res.is_hit:
                combined_tags.append("玄鐵買點")
            if timestral_res and timestral_res.is_hit:
                combined_tags.append("Timestral看漲")
                if timestral_res.signals.get("risk_reward_ratio", 0) >= 2.0:
                    combined_tags.append("高風報比")
            if timesfm_res and timesfm_res.is_hit:
                combined_tags.append("TimesFM看漲")
                if timesfm_res.signals.get("risk_reward_ratio", 0) >= 2.0:
                    combined_tags.append("高盈虧比")
            if lstm_res and lstm_res.is_hit and not timestral_res:
                combined_tags.append("LSTM看漲")

            # Dual Foundation Model Consensus (TimesFM ∩ Timestral)
            hit_timestral = bool(timestral_res and timestral_res.is_hit)
            hit_timesfm = bool(timesfm_res and timesfm_res.is_hit)
            hit_both_fm = hit_timestral and hit_timesfm
            hit_any_fm = hit_timestral or hit_timesfm

            # Legacy ML hit indicator (for backward compatibility if only LSTM was run)
            hit_lstm = bool(lstm_res and lstm_res.is_hit)
            hit_both_ml = hit_both_fm or (hit_lstm and (hit_timesfm or hit_timestral))
            hit_any_ml = hit_any_fm or hit_lstm

            if hit_both_fm:
                combined_tags.append("🔮雙大模型共振")
            elif hit_both_ml and not hit_both_fm:
                combined_tags.append("🔮雙ML共振")

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

            # Quadruple Resonance (四重大模型/多維共振): 玄鐵 + 法人 + (雙大模型 或 雙ML)
            is_quad_resonance = hit_xuantie and hit_inst and (hit_both_fm or hit_both_ml)

            # Triple Resonance (三重共振): 嚴格符合多策略共振組合（必須具備技術面玄鐵或籌碼面法人支撐）
            # 1. 玄鐵 + 法人 + 基礎模型 (TimesFM 或 Timestral)
            # 2. 玄鐵 + 雙大模型 (TimesFM + Timestral)
            # 3. 法人 + 雙大模型 (TimesFM + Timestral)
            # 4. 板塊 + (玄鐵 或 法人) + (基礎模型 或 另一方)
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
            if not ml_potentials and lstm_res and lstm_res.potential is not None and np.isfinite(lstm_res.potential):
                ml_potentials.append(float(lstm_res.potential))

            has_ml = len(ml_potentials) > 0
            avg_ml_pot = sum(ml_potentials) / len(ml_potentials) if has_ml else 0.0

            # Bearish check (防守門檻):
            # 1. 雙基礎模型均看跌 (<= 0) 絕對為看跌/防守
            # 2. 平均潛力 <= 0 或任一模型嚴重破底 (<= -5.0%) 且無強力彌補
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
                (lstm_res and lstm_res.is_hit)
            )

            # Defensive tags for bearish stocks
            if is_ml_bearish:
                if any(p <= -10.0 for p in ml_potentials):
                    combined_tags.append("🔻深度防守")
                else:
                    combined_tags.append("🔻防守")

            # Determine Resonance Tier
            resonance_tier = ""
            if not is_ml_bearish:
                if is_quad_resonance:
                    resonance_tier = "👑四重共振"
                elif is_triple_resonance:
                    resonance_tier = "🏆三重共振"
                elif hit_both_fm:
                    resonance_tier = "🔮雙大模型共振"
                elif hit_both_ml:
                    resonance_tier = "🔮雙ML共振"
                elif hit_count >= 2:
                    resonance_tier = "🌟多維共振" if hit_count > 2 else "⭐雙重共振"

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

            # Current price resolution
            curr_price = next(iter(strats.values())).current_price if strats else 0.0

            # Target Price & Potential Defensive Synthesis:
            # Ensemble of Google TimesFM and Mistral Timestral P50
            valid_target_prices = []
            if timesfm_res and timesfm_res.predicted_price is not None and np.isfinite(timesfm_res.predicted_price):
                valid_target_prices.append(float(timesfm_res.predicted_price))
            if timestral_res and timestral_res.predicted_price is not None and np.isfinite(timestral_res.predicted_price):
                valid_target_prices.append(float(timestral_res.predicted_price))

            if len(valid_target_prices) == 2:
                synthesized_target = round(0.5 * valid_target_prices[0] + 0.5 * valid_target_prices[1], 2)
            elif len(valid_target_prices) == 1:
                synthesized_target = round(valid_target_prices[0], 2)
            elif lstm_res and lstm_res.predicted_price is not None and np.isfinite(lstm_res.predicted_price):
                synthesized_target = round(float(lstm_res.predicted_price), 2)
            else:
                synthesized_target = curr_price

            if curr_price > 0:
                synthesized_potential = round(((synthesized_target - curr_price) / curr_price) * 100.0, 2)
            else:
                synthesized_potential = 0.0

            # Risk-reward resolution
            best_rr = None
            if timestral_res and timestral_res.signals.get("risk_reward_ratio") is not None:
                best_rr = timestral_res.signals.get("risk_reward_ratio")
            elif timesfm_res and timesfm_res.signals.get("risk_reward_ratio") is not None:
                best_rr = timesfm_res.signals.get("risk_reward_ratio")

            # Deduplicate tags and prepend resonance tier
            if is_ml_bearish:
                combined_tags = [t for t in combined_tags if "共振" not in t and "符合" not in t]
            final_tags = list(dict.fromkeys(combined_tags))
            if resonance_tier and resonance_tier not in final_tags:
                final_tags.insert(0, resonance_tier)

            # Resonance Overlap Condition
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
            if lstm_res:
                entry["lstm_potential"] = lstm_res.potential
            if inst_res:
                entry["institutional"] = inst_res.metadata

            ranked_stocks.append(entry)

            if is_valid_overlap:
                overlap_candidates.append(entry)

        # Tier weights for sorting priority (四重 > 三重 > 雙大模型 > 雙ML > 多維/雙重)
        tier_weights = {
            "👑四重共振": 450.0,
            "🏆三重共振": 350.0,
            "🔮雙大模型共振": 250.0,
            "🔮雙ML共振": 200.0,
            "🌟多維共振": 150.0,
            "⭐雙重共振": 100.0
        }

        def candidate_sort_key(c: Dict[str, Any]):
            tier_val = tier_weights.get(c.get("resonance_tier", ""), 50.0)
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
                "dual_model_consensus": o.get("dual_model_consensus", False),
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
            lstm_results=lstm_results_legacy,
            overlap_results=overlap_df,
            macro_state=macro_state
        )
