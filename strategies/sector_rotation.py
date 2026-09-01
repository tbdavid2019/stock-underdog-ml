"""
strategies/sector_rotation.py - 7 大板塊資金輪動策略 (Sector Rotation Strategy)

追蹤市場核心板塊之 10D / 15D / 20D 加權動量資金流，
挑選排名前 3 大強勢板塊，並篩選板塊內站穩季線 MA60 之領頭個股。
"""

import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from core.config import config
from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy

logger = logging.getLogger("stock_app.strategy.sector")


@register_strategy("sector_rotation", category="sector")
class SectorRotationStrategy(BaseStrategy):
    """板塊資金輪動選股策略"""

    def __init__(self, top_sectors: int = 3, min_return_threshold: float = -3.0):
        super().__init__(name="板塊輪動", category="sector")
        self.top_sectors = top_sectors
        self.min_return_threshold = min_return_threshold

    def evaluate(self, context: StockContext) -> StrategyResult:
        """單一股票評估 (單獨調用時由 evaluate_batch 代理)"""
        # 單一股票無法計算板塊相對排名，預設回傳
        return StrategyResult(
            ticker=context.ticker,
            strategy_name=self.name,
            is_hit=False,
            score=0.0,
            reason="板塊輪動需要批次評估",
            metadata={"sector": config.sector.get_sector(context.ticker)}
        )

    def evaluate_batch(self, contexts: Dict[str, StockContext]) -> List[StrategyResult]:
        """批次計算所有成分股之板塊動量並篩選強勢板塊領頭股"""
        results: List[StrategyResult] = []
        if not contexts:
            return results

        # 1. 整理各標的動量與板塊歸屬
        sector_stocks: Dict[str, List[Dict[str, Any]]] = {}
        for ticker, ctx in contexts.items():
            df = ctx.df
            if df is None or len(df) < 20:
                continue

            close = df["Close"]
            cur_p = float(close.iloc[-1])
            p_10d = float(close.iloc[-10]) if len(close) >= 10 else cur_p
            p_15d = float(close.iloc[-15]) if len(close) >= 15 else cur_p
            p_20d = float(close.iloc[-20]) if len(close) >= 20 else cur_p
            ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else cur_p

            mom_10 = (cur_p / p_10d - 1.0) * 100
            mom_15 = (cur_p / p_15d - 1.0) * 100
            mom_20 = (cur_p / p_20d - 1.0) * 100
            
            # 綜合加權動量 (10D: 40%, 15D: 30%, 20D: 30%)
            weighted_mom = 0.4 * mom_10 + 0.3 * mom_15 + 0.3 * mom_20
            above_ma60 = cur_p >= ma60

            sector = config.sector.get_sector(ticker)
            if sector not in sector_stocks:
                sector_stocks[sector] = []

            sector_stocks[sector].append({
                "ticker": ticker,
                "context": ctx,
                "current_price": cur_p,
                "weighted_mom": weighted_mom,
                "mom_20d": mom_20,
                "above_ma60": above_ma60
            })

        # 2. 計算各板塊平均報酬並排名
        sector_scores: Dict[str, float] = {}
        for sec, s_list in sector_stocks.items():
            if s_list:
                avg_mom = float(np.mean([s["weighted_mom"] for s in s_list]))
                sector_scores[sec] = avg_mom

        # 排序板塊 (由大至小)
        ranked_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top_sector_names = set()
        
        # 篩選前 N 大強勢板塊 (且平均動量高於門檻，避免空頭下買垃圾板塊)
        for sec, score in ranked_sectors[:self.top_sectors]:
            if score >= self.min_return_threshold:
                top_sector_names.add(sec)

        # 3. 輸出 StrategyResult
        for ticker, ctx in contexts.items():
            sec = config.sector.get_sector(ticker)
            sec_score = sector_scores.get(sec, 0.0)
            is_in_top_sector = sec in top_sector_names
            
            # 個股符合條件: 處於 Top 板塊 + 站上 MA60 + 個股動量為正
            stock_info = next((s for s in sector_stocks.get(sec, []) if s["ticker"] == ticker), None)
            is_hit = False
            score_val = 0.0
            signals = []

            if is_in_top_sector and stock_info:
                if stock_info["above_ma60"] and stock_info["weighted_mom"] > 0:
                    is_hit = True
                    score_val = min(100.0, max(50.0, 50.0 + stock_info["weighted_mom"] * 2.0))
                    signals.append(f"強勢板塊({sec})")
                    signals.append("站穩MA60")
                    signals.append(f"動量{stock_info['weighted_mom']:+.1f}%")

            results.append(StrategyResult(
                ticker=ticker,
                strategy_name=self.name,
                is_hit=is_hit,
                score=score_val,
                signals=signals,
                metadata={
                    "sector": sec,
                    "sector_score": round(sec_score, 2),
                    "is_top_sector": is_in_top_sector,
                    "weighted_mom": round(stock_info["weighted_mom"], 2) if stock_info else 0.0
                }
            ))

        return results
