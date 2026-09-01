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
            inst_res = strats.get("institutional")
            sector_res = strats.get("sector_rotation") or strats.get("sector")

            if xuantie_res and xuantie_res.is_hit:
                combined_tags.append("玄鐵買點")
            if lstm_res and lstm_res.is_hit:
                combined_tags.append("LSTM看漲")
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

            # Triple Resonance (三重共振): 玄鐵 + LSTM + 投信/法人
            is_triple_resonance = (
                xuantie_res and xuantie_res.is_hit and
                lstm_res and lstm_res.is_hit and
                inst_res and inst_res.is_hit
            )

            # Calculate Weighted Composite Score (0~100)
            score_total = 0.0
            weight_total = 0.0

            for s_name, s_weight in self.weights.items():
                if s_name == "fundamental":
                    score_total += fund_score * s_weight * 5.0
                    weight_total += s_weight
                elif s_name in strats:
                    score_total += strats[s_name].score * s_weight
                    weight_total += s_weight

            composite_score = round(score_total / weight_total, 2) if weight_total > 0 else 0.0

            # Macro Exposure Multiplier Discount
            if macro_state and macro_state.exposure < 1.0:
                composite_score = round(composite_score * macro_state.exposure, 2)

            # Get current price from first available result
            curr_price = next(iter(strats.values())).current_price if strats else 0.0

            # Deduplicate tags
            final_tags = list(dict.fromkeys(combined_tags))
            if is_triple_resonance:
                final_tags.insert(0, "🏆三重共振")
            elif hit_count >= self.min_overlap_count or (xuantie_res and xuantie_res.is_hit and lstm_res and lstm_res.is_hit):
                final_tags.insert(0, "雙重符合" if hit_count == 2 else f"{hit_count}重符合")

            entry = {
                "ticker": ticker,
                "composite_score": composite_score,
                "current_price": curr_price,
                "hit_count": hit_count,
                "hit_strategies": [s.strategy_name for s in hits],
                "tags": final_tags,
                "fundamentals": fund
            }

            if xuantie_res:
                entry["ma60"] = xuantie_res.metrics.get("ma60")
                entry["pullback_type"] = xuantie_res.signals.get("pullback_type", "")
            if lstm_res:
                entry["lstm_potential"] = lstm_res.potential
                entry["predicted_price"] = lstm_res.predicted_price
            if inst_res:
                entry["institutional"] = inst_res.metadata

            ranked_stocks.append(entry)

            # Check overlap threshold (2 or more hits or Triple Resonance)
            if is_triple_resonance or hit_count >= self.min_overlap_count or (xuantie_res and xuantie_res.is_hit and lstm_res and lstm_res.is_hit):
                overlap_candidates.append(entry)

        # Sort ranked stocks and overlaps
        ranked_stocks.sort(key=lambda x: x["composite_score"], reverse=True)
        overlap_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        # Build legacy overlap_df
        overlap_legacy_rows = []
        for o in overlap_candidates:
            overlap_legacy_rows.append({
                "ticker": o["ticker"],
                "lstm_potential": o.get("lstm_potential", 0.0),
                "current_price": o["current_price"],
                "predicted_price": o.get("predicted_price", o["current_price"]),
                "pullback_type": o.get("pullback_type", ""),
                "ma60": o.get("ma60"),
                "pe": o["fundamentals"].get("pe"),
                "pb": o["fundamentals"].get("pb"),
                "forward_pe": o["fundamentals"].get("forward_pe"),
                "ev_ebitda": o["fundamentals"].get("ev_ebitda")
            })
        overlap_df = pd.DataFrame(overlap_legacy_rows) if overlap_legacy_rows else pd.DataFrame()

        return EvaluationReport(
            index_name=index_name,
            strategy_results=strategy_outputs,
            overlap_candidates=overlap_candidates,
            ranked_stocks=ranked_stocks,
            xuantie_results=xuantie_df,
            lstm_results=lstm_results_legacy,
            overlap_results=overlap_df,
            macro_state=macro_state
        )
