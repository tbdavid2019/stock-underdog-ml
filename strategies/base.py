"""
Base Strategy Interface and Standardized Data Contracts.
Defines StockContext, StrategyResult, and the BaseStrategy abstract class.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd


@dataclass
class StockContext:
    """Standardized input context for strategy evaluation"""
    ticker: str
    df: pd.DataFrame
    fundamentals: Dict[str, Optional[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Standardized output result produced by any strategy"""
    ticker: str
    strategy_name: str
    is_hit: bool
    score: float = 0.0
    current_price: float = 0.0
    predicted_price: Optional[float] = None
    potential: Optional[float] = None
    signals: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """Abstract base class for all trading and stock selection strategies"""
    
    name: str = "base_strategy"
    category: str = "technical"   # 'technical', 'ml', 'fundamental', 'chip', 'sector'
    required_lookback: int = 60   # Minimum required historical bars

    def __init__(
        self, 
        name: Optional[str] = None, 
        category: Optional[str] = None, 
        required_lookback: Optional[int] = None
    ):
        if name:
            self.name = name
        if category:
            self.category = category
        if required_lookback:
            self.required_lookback = required_lookback

    @abstractmethod
    def evaluate(self, context: StockContext) -> StrategyResult:
        """
        Evaluate a single stock and return a StrategyResult.
        """
        pass

    def evaluate_batch(self, contexts: Dict[str, StockContext]) -> List[StrategyResult]:
        """
        Evaluate a batch of stocks. Subclasses can override for vectorized or GPU batch inference.
        """
        results = []
        for ctx in contexts.values():
            try:
                res = self.evaluate(ctx)
                if res is not None:
                    results.append(res)
            except Exception:
                pass
        return results
