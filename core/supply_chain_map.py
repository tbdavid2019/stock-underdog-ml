"""
core/supply_chain_map.py - 前沿技術瓶頸與台美產業鏈映射庫

載入 config/discovery_supply_chain.json 外部設定檔，
支援主題到標的、標的到主題與美股錨點的雙向查詢與緩存。
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional, Set

from core.ticker_canonicalizer import canonicalize_ticker

logger = logging.getLogger("stock_app.discovery.supply_chain")


class SupplyChainMap:
    """前沿技術瓶頸與供應鏈映射器"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized and not config_path:
            return

        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "discovery_supply_chain.json"
        )
        self._data: Dict[str, Any] = {}
        self._themes_by_id: Dict[str, Dict[str, Any]] = {}
        self._ticker_to_themes: Dict[str, List[Dict[str, Any]]] = {}
        self._ticker_to_anchors: Dict[str, Set[str]] = {}
        self._all_tickers: Set[str] = set()

        self.reload()
        self._initialized = True

    def reload(self):
        """重新讀取設定檔並建立雙向索引快取"""
        if not os.path.exists(self.config_path):
            logger.warning("供應鏈設定檔不存在: %s", self.config_path)
            self._data = {"themes": []}
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception as e:
            logger.error("讀取供應鏈設定檔失敗: %s", e)
            self._data = {"themes": []}

        self._themes_by_id.clear()
        self._ticker_to_themes.clear()
        self._ticker_to_anchors.clear()
        self._all_tickers.clear()

        for theme in self._data.get("themes", []):
            t_id = theme.get("id")
            if not t_id:
                continue
            self._themes_by_id[t_id] = theme

            anchors = [canonicalize_ticker(a) for a in theme.get("us_anchors", []) if canonicalize_ticker(a)]
            tw_chain = theme.get("tw_supply_chain", [])

            for item in tw_chain:
                raw_ticker = item.get("ticker", "")
                canon = canonicalize_ticker(raw_ticker)
                if not canon:
                    continue

                self._all_tickers.add(canon)

                if canon not in self._ticker_to_themes:
                    self._ticker_to_themes[canon] = []
                self._ticker_to_themes[canon].append({
                    "theme_id": t_id,
                    "theme_name": theme.get("name_zh", ""),
                    "theme_name_en": theme.get("name_en", ""),
                    "role": item.get("role", ""),
                    "relevance_weight": item.get("relevance_weight", 0.8),
                    "name": item.get("name", "")
                })

                if canon not in self._ticker_to_anchors:
                    self._ticker_to_anchors[canon] = set()
                self._ticker_to_anchors[canon].update(anchors)

            for anchor in anchors:
                self._all_tickers.add(anchor)

    def get_all_themes(self) -> List[Dict[str, Any]]:
        """取得所有前沿主題設定"""
        return self._data.get("themes", [])

    def get_theme(self, theme_id: str) -> Optional[Dict[str, Any]]:
        """依照主題 ID 查詢主題資訊"""
        return self._themes_by_id.get(theme_id)

    def get_supply_chain_for_theme(self, theme_id: str) -> List[Dict[str, Any]]:
        """取得指定主題之台灣供應鏈標的"""
        theme = self.get_theme(theme_id)
        if not theme:
            return []
        return theme.get("tw_supply_chain", [])

    def get_anchors_for_theme(self, theme_id: str) -> List[str]:
        """取得指定主題之美股錨點標的"""
        theme = self.get_theme(theme_id)
        if not theme:
            return []
        return [canonicalize_ticker(a) for a in theme.get("us_anchors", []) if canonicalize_ticker(a)]

    def get_themes_for_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """反向查詢：依標的代碼查詢所屬的所有前沿主題與角色"""
        canon = canonicalize_ticker(ticker)
        return self._ticker_to_themes.get(canon, [])

    def get_anchors_for_ticker(self, ticker: str) -> List[str]:
        """反向查詢：依標的代碼查詢其對應的美股錨點"""
        canon = canonicalize_ticker(ticker)
        anchors = self._ticker_to_anchors.get(canon, set())
        return sorted(list(anchors))

    def get_all_supply_chain_tickers(self) -> Set[str]:
        """取得設定檔中定義之所有標的集合 (台股+美股錨點)"""
        return set(self._all_tickers)

    def get_bottleneck_description(self, theme_id: str) -> str:
        """取得主題之技術瓶頸詳細描述"""
        theme = self.get_theme(theme_id)
        return theme.get("bottleneck_description", "") if theme else ""

    def map_paper_to_themes(self, title: str, abstract: str) -> List[str]:
        """
        利用關鍵字規則將論文標題與摘要映射至相應之技術瓶頸主題 ID
        """
        text = f"{title} {abstract}".lower()
        matched: List[str] = []

        # 關鍵字特徵定義
        rules = {
            "cpo_optical_interconnect": [
                "co-packaged", "cpo", "silicon photonics", "optical interconnect",
                "optical transceiver", "light engine", "photonic integrated", "laser diode",
                "inp", "gaas", "wdm", "fiber array", "fau", "光互連", "矽光子", "共同封裝光學"
            ],
            "hbm_kv_cache": [
                "hbm", "hbm3", "hbm3e", "hbm4", "high bandwidth memory", "kv-cache",
                "kv cache", "memory wall", "memory bandwidth", "attention cache",
                "context length", "bandwidth bottleneck", "die-to-die", "記憶體牆", "高頻寬記憶體"
            ],
            "liquid_cooling": [
                "liquid cooling", "cold plate", "immersion cooling", "cdu",
                "cooling distribution", "thermal management", "heat dissipation",
                "two-phase cooling", "manifold", "vapor chamber", "散熱", "液冷", "水冷"
            ],
            "embodied_ai_robotics": [
                "embodied", "humanoid", "robotics", "actuator", "bimanual",
                "manipulation", "vision-language-action", "vla", "harmonic drive",
                "cycloidal", "tactile sensor", "servo", "具身智能", "機器人", "減速機"
            ],
            "advanced_packaging_cowos": [
                "cowos", "2.5d", "3d packaging", "chiplet", "interposer", "soic",
                "hybrid bonding", "wafer-level packaging", "fan-out", "advanced packaging",
                "先進封裝", "小晶片", "晶圓級"
            ]
        }

        for theme_id, kw_list in rules.items():
            if any(kw in text for kw in kw_list):
                matched.append(theme_id)

        # 若未命中任何關鍵字，根據普遍 LLM / 硬體概念預設歸類
        return matched


# 全域單例
default_supply_chain_map = SupplyChainMap()
