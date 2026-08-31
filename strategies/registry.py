"""
Strategy Registry and Factory.
Enables pluggable strategy registration via decorator and dynamic instantiation.
"""
import logging
from typing import Dict, Type, List, Optional
from strategies.base import BaseStrategy

logger = logging.getLogger("stock_app.strategies")


class StrategyRegistry:
    """Central registry for discovering and instantiating strategies"""

    _registry: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class with a unique identifier"""
        def decorator(subclass: Type[BaseStrategy]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_class(cls, name: str, subclass: Type[BaseStrategy]):
        """Directly register a strategy class"""
        cls._registry[name.lower()] = subclass

    @classmethod
    def get_strategy(cls, name: str, **kwargs) -> BaseStrategy:
        """Instantiate a registered strategy by name"""
        normalized_name = name.lower()
        if normalized_name not in cls._registry:
            raise ValueError(f"Unknown strategy: '{name}'. Available: {list(cls._registry.keys())}")
        return cls._registry[normalized_name](**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all currently registered strategy identifiers"""
        return list(cls._registry.keys())

    @classmethod
    def create_strategies(cls, names: List[str]) -> List[BaseStrategy]:
        """Instantiate multiple strategies from a list of names"""
        strategies = []
        for name in names:
            try:
                strat = cls.get_strategy(name)
                strategies.append(strat)
            except Exception as e:
                logger.error(f"❌ 初始化策略 '{name}' 失敗: {e}")
        return strategies


# Shorthand alias
register_strategy = StrategyRegistry.register
