from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
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
    lstm_results: List[Dict[str, Any]] = field(default_factory=list)
    timesfm_results: List[Dict[str, Any]] = field(default_factory=list)
    overlap_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_state: Optional[MacroState] = None
    ai_summary: str = ""


class CompositeEvaluator:
    """Evaluator that combines multiple strategy signals, institutional flows, and macro gates"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_overlap_count: int = 2
    ):
        self.weights = weights or {
            "xuantie": 0.35,
            "lstm": 0.35,
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
        Evaluate all strategy outputs for an index, computing composite scores and overlap.
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

        # 2. Build Legacy DataFrames for backward compatibility
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

        # LSTM legacy list
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

        # TimesFM legacy list
        timesfm_results_legacy = []
        if "timesfm" in strategy_outputs:
            for r in strategy_outputs["timesfm"]:
                if r.potential is not None:
                    timesfm_results_legacy.append({
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
            timesfm_results_legacy.sort(key=lambda x: x["potential"], reverse=True)

        # Dynamic active weights with timesfm support
        active_weights = dict(self.weights)
        if "timesfm" in strategy_outputs and "timesfm" not in active_weights:
            if "lstm" in active_weights:
                active_weights["lstm"] = 0.20
                active_weights["timesfm"] = 0.20
                active_weights["xuantie"] = 0.30
            else:
                active_weights["timesfm"] = 0.35

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
            lstm_res = strats.get("lstm")
            timesfm_res = strats.get("timesfm")
            inst_res = strats.get("institutional")
            sector_res = strats.get("sector_rotation") or strats.get("sector")

            if xuantie_res and xuantie_res.is_hit:
                combined_tags.append("玄鐵買點")
            if lstm_res and lstm_res.is_hit:
                combined_tags.append("LSTM看漲")
            if timesfm_res and timesfm_res.is_hit:
                combined_tags.append("TimesFM看漲")
                if timesfm_res.signals.get("risk_reward_ratio", 0) >= 2.0:
                    combined_tags.append("高盈虧比")

            # Dual ML Resonance (雙ML共振: LSTM ∩ TimesFM)
            is_dual_ml_resonance = bool(
                lstm_res and lstm_res.is_hit and
                timesfm_res and timesfm_res.is_hit
            )
            if is_dual_ml_resonance:
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
            hit_lstm = bool(lstm_res and lstm_res.is_hit)
            hit_timesfm = bool(timesfm_res and timesfm_res.is_hit)
            hit_sector = bool(sector_res and sector_res.is_hit)

            hit_any_ml = hit_lstm or hit_timesfm
            hit_both_ml = hit_lstm and hit_timesfm

            # Quadruple Resonance (四重共振): 玄鐵 + 法人 + LSTM + TimesFM
            is_quad_resonance = hit_xuantie and hit_inst and hit_both_ml

            # Triple Resonance (三重共振): 嚴格符合多策略共振組合（必須具備技術面玄鐵或籌碼面法人支撐）
            # 1. 玄鐵 + 法人 + (LSTM 或 TimesFM)
            # 2. 玄鐵 + 雙ML (LSTM + TimesFM)
            # 3. 法人 + 雙ML (LSTM + TimesFM)
            # 4. 板塊 + (玄鐵 或 法人) + (LSTM 或 TimesFM 或 另一方)
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

            # ML Potentials & Positive Gate Evaluation
            lstm_pot = lstm_res.potential if (lstm_res and lstm_res.potential is not None) else None
            timesfm_pot = timesfm_res.potential if (timesfm_res and timesfm_res.potential is not None) else None

            ml_potentials = []
            if lstm_pot is not None and not (isinstance(lstm_pot, float) and pd.isna(lstm_pot)):
                ml_potentials.append(float(lstm_pot))
            if timesfm_pot is not None and not (isinstance(timesfm_pot, float) and pd.isna(timesfm_pot)):
                ml_potentials.append(float(timesfm_pot))

            has_ml = len(ml_potentials) > 0
            avg_ml_pot = sum(ml_potentials) / len(ml_potentials) if has_ml else 0.0

            # Bearish check (防守門檻):
            # 1. 雙 ML 均看跌 (<= 0) 絕對為看跌/防守
            # 2. 雙 ML 平均潛力 <= 0 或任一模型嚴重破底 (<= -5.0%) 且無強力彌補
            # 3. 單 ML 潛力 <= 0
            is_ml_bearish = False
            if has_ml:
                if all(p <= 0.0 for p in ml_potentials):
                    is_ml_bearish = True
                elif len(ml_potentials) >= 2:
                    if avg_ml_pot <= 0.0 or min(ml_potentials) <= -5.0:
                        is_ml_bearish = True
                elif len(ml_potentials) == 1 and ml_potentials[0] <= 0.0:
                    is_ml_bearish = True

            # Must have at least one primary directional signal (Technical buy or ML bullish hit)
            has_primary_signal = bool(
                (xuantie_res and xuantie_res.is_hit) or
                (lstm_res and lstm_res.is_hit) or
                (timesfm_res and timesfm_res.is_hit)
            )

            # Defensive tags for bearish stocks
            if is_ml_bearish:
                if any(p <= -10.0 for p in ml_potentials):
                    combined_tags.append("🔻深度防守")
                else:
                    combined_tags.append("🔻防守")

            # Determine Resonance Tier (Strictly exclude bearish/defensive stocks from resonance tiers)
            resonance_tier = ""
            if not is_ml_bearish:
                if is_quad_resonance:
                    resonance_tier = "👑四重共振"
                elif is_triple_resonance:
                    resonance_tier = "🏆三重共振"
                elif is_dual_ml_resonance:
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
            if macro_state and macro_state.exposure < 1.0:
                composite_score = round(composite_score * macro_state.exposure, 2)

            # Bearish ML Penalty on composite score: prevent high ranking for negative stocks
            if is_ml_bearish:
                composite_score = round(composite_score * 0.5, 2)

            # Get current price from first available result
            curr_price = next(iter(strats.values())).current_price if strats else 0.0

            # Deduplicate tags and prepend resonance tier
            if is_ml_bearish:
                # Strip out any resonance / overlap tags from bearish candidates so they never pollute DB or sinks
                combined_tags = [t for t in combined_tags if "共振" not in t and "符合" not in t]
            final_tags = list(dict.fromkeys(combined_tags))
            if resonance_tier and resonance_tier not in final_tags:
                final_tags.insert(0, resonance_tier)

            # Resonance Overlap Condition:
            # 1. 符合四重、三重、雙ML共振，或 2 個以上策略命中且具備主要方向性訊號
            # 2. 嚴格過濾負值/大幅看跌標的 (not is_ml_bearish)
            is_resonance = (
                is_quad_resonance or
                is_triple_resonance or
                is_dual_ml_resonance or
                (hit_count >= self.min_overlap_count and has_primary_signal)
            )

            is_valid_overlap = is_resonance and (not is_ml_bearish)

            entry = {
                "ticker": ticker,
                "composite_score": composite_score,
                "current_price": curr_price,
                "hit_count": hit_count,
                "hit_strategies": [s.strategy_name for s in hits],
                "tags": final_tags,
                "fundamentals": fund,
                "resonance_tier": resonance_tier,
                "is_ml_bearish": is_ml_bearish,
                "avg_ml_pot": avg_ml_pot
            }

            if xuantie_res:
                entry["ma60"] = xuantie_res.metrics.get("ma60")
                entry["pullback_type"] = xuantie_res.signals.get("pullback_type", "")
            if lstm_res:
                entry["lstm_potential"] = lstm_res.potential
                entry["predicted_price"] = lstm_res.predicted_price
            if timesfm_res:
                entry["timesfm_potential"] = timesfm_res.potential
                entry["timesfm_predicted_price"] = timesfm_res.predicted_price
                entry["risk_reward_ratio"] = timesfm_res.signals.get("risk_reward_ratio")
            if inst_res:
                entry["institutional"] = inst_res.metadata

            ranked_stocks.append(entry)

            # Check overlap threshold: only valid, positive-resonance candidates enter 優先推薦
            if is_valid_overlap:
                overlap_candidates.append(entry)

        # Tier weights for sorting priority (四重 > 三重 > 雙ML > 多維/雙重)
        tier_weights = {
            "👑四重共振": 400.0,
            "🏆三重共振": 300.0,
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
                "lstm_potential": o.get("lstm_potential", 0.0),
                "timesfm_potential": o.get("timesfm_potential"),
                "current_price": o["current_price"],
                "predicted_price": o.get("predicted_price", o["current_price"]),
                "timesfm_predicted_price": o.get("timesfm_predicted_price"),
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
            lstm_results=lstm_results_legacy,
            timesfm_results=timesfm_results_legacy,
            overlap_results=overlap_df,
            macro_state=macro_state
        )
