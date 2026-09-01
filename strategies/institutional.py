"""
strategies/institutional.py - 三大法人籌碼策略 (Institutional Flow Strategy)

評估台股三大法人籌碼動向，篩選投信連續買超 (Streak >= 3) 或土洋合買超之個股。
"""

import logging
from typing import Dict, List, Any

from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy
from data.institutional import InstitutionalProvider, InstitutionalSummary

logger = logging.getLogger("stock_app.strategy.institutional")


@register_strategy("institutional", category="institutional")
class InstitutionalStrategy(BaseStrategy):
    """三大法人籌碼策略"""

    def __init__(self, min_streak: int = 2, min_trust_5d_shares: int = 200):
        super().__init__(name="三大法人籌碼", category="institutional")
        self.min_streak = min_streak
        self.min_trust_5d_shares = min_trust_5d_shares

    def evaluate(self, context: StockContext) -> StrategyResult:
        """單一股票評估"""
        summaries = InstitutionalProvider.get_institutional_summary_batch([context.ticker], lookback_days=5)
        summary = summaries.get(context.ticker, InstitutionalSummary(ticker=context.ticker))
        return self._build_result(context.ticker, summary)

    def evaluate_batch(self, contexts: Dict[str, StockContext]) -> List[StrategyResult]:
        """批次評估多檔股票法人籌碼"""
        tickers = list(contexts.keys())
        summaries = InstitutionalProvider.get_institutional_summary_batch(tickers, lookback_days=5)
        
        results = []
        for ticker, ctx in contexts.items():
            summary = summaries.get(ticker, InstitutionalSummary(ticker=ticker))
            results.append(self._build_result(ticker, summary))
        return results

    def _build_result(self, ticker: str, summary: InstitutionalSummary) -> StrategyResult:
        is_hit = False
        score = 0.0
        signals = []

        # 1. 投信連買判斷
        if summary.trust_streak >= 3:
            is_hit = True
            score += 40.0 + min(30.0, summary.trust_streak * 10.0)
            signals.append(f"投信連買{summary.trust_streak}天")
        elif summary.trust_net_5d >= self.min_trust_5d_shares:
            is_hit = True
            score += 30.0
            signals.append(f"投信5D買超({summary.trust_net_5d}張)")

        # 2. 土洋合買加分
        if summary.is_sync_buy:
            is_hit = True
            score += 30.0
            signals.append("土洋合買")

        # 3. 外資大幅買超加分
        if summary.foreign_net_5d >= 1000:
            score += 20.0
            signals.append(f"外資大買({summary.foreign_net_5d}張)")

        score = min(100.0, score)

        return StrategyResult(
            ticker=ticker,
            strategy_name=self.name,
            is_hit=is_hit,
            score=score,
            signals=signals,
            metadata=summary.to_dict()
        )
