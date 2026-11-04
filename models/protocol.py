"""
models/protocol.py - Standard Model Plugin Protocol and Registry for Elephant & Goldfish Architecture.

Defines the runtime checkable ModelPlugin protocol and thread-safe ModelPluginRegistry.
Enables dynamic registration, health probing, market-adaptive batch prediction, and graceful teardown.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Protocol, Type, Union, runtime_checkable
import numpy as np

logger = logging.getLogger("stock_app.models.protocol")


@runtime_checkable
class ModelPlugin(Protocol):
    """
    大象底座所依賴的標準化時序模型插件協議 (Elephant & Goldfish Model Plugin Protocol).
    任何時序基礎模型外掛 (TimesFM, Timestral, MOIRAI-2 等) 均需完整符合此協議。
    """
    model_id: str                # 唯一識別碼，如 "timesfm-3.0", "timestral-209m", "moirai-2.0-R-small"
    display_name: str            # 介面顯示名稱，如 "Google TimesFM 3.0"
    version: str                 # 模型版本號，如 "3.0"
    author: str                  # 機構/作者，如 "Google Research"
    supported_markets: List[str] # 支援市場清單 ["tw", "us", "jp"]
    default_role: str            # 預設角色: "active" | "shadow" | "candidate"

    # 🌟 側參數與協變量能力宣告 (Exogenous Covariate Capability)
    supports_covariates: bool    # 是否原生支援多變量/外部協變量
    required_covariates: List[str] # 偏好或必備之側參數清單，如 ["vix", "usdtwd", "sox", "tnx"]

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        [生命週期勾子 1: 初始化與權重載入]
        下載並載入模型權重至指定運算設備，配置 context 窗口與推論參數。
        若權重不存在或環境缺乏必要依賴，回傳 False 並記錄結構化 log，不可拋出未攔截異常。
        """
        ...

    def health_check(self) -> Dict[str, Any]:
        """
        [生命週期勾子 2: 健康檢查與設備探測]
        回傳設備狀態、記憶體佔用量、推論延遲基準，供管線判定是否熔斷降級。
        範例: {"status": "healthy", "device": "cpu", "memory_mb": 420.5, "p99_latency_ms": 45.2}
        """
        ...

    def predict_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        covariates_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        horizon: int = 5,
        market: str = "tw",
        market_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        [生命週期勾子 3: 批次推論運算]
        標準輸入:
          - series_dict: 標的日收盤價時序字典 (包含足額歷史天數)。
          - covariates_dict: 動態側參數特徵矩陣字典 (可選)，結構為:
            {
                "ticker": {
                    "vix": np.ndarray,       # 恐慌指數時序 (長度與 series 對齊)
                    "fx_rate": np.ndarray,   # 本地匯率時序 (USD/TWD, USD/JPY 或 DXY)
                    "yield_10y": np.ndarray, # 美債 10 年期殖利率時序 (^TNX)
                    "benchmark": np.ndarray, # 基準指數/板塊時序 (^SOX, ^TWII, ^N225)
                    "inst_flow": np.ndarray  # 法人籌碼/淨買賣動能時序 (台股特有)
                }
            }
          - horizon: 預測步長天數 (1, 5, 20)。
          - market: 目標市場代碼 ("tw", "us", "jp")。
          - market_params: 跨市場特化參數覆寫字典。

        標準輸出字典結構:
        {
            "ticker": {
                "predicted_price": float,           # P50 點預測目標價
                "potential": float,                 # 預期潛在漲跌幅 %
                "quantiles": {                      # 蒙地卡羅風控分位數
                    "p10": float, "p50": float, "p90": float
                },
                "trajectory": List[float],          # 各預測步長軌跡 (t+1 ... t+h)
                "risk_reward_ratio": float          # 預期盈虧比 (Upside / Downside)
            }
        }
        """
        ...

    def teardown(self) -> None:
        """
        [生命週期勾子 4: 資源釋放與優雅退出]
        清空 GPU/MPS 快取、釋放檔案鎖與模型推論環境，確保多進程或熱拔除時不發生記憶體洩漏。
        """
        ...

    def generate_tags(self, potential: float, risk_reward: Optional[float] = None) -> List[str]:
        """
        [生命週期勾子 5: 模型實體標籤生成]
        純粹基於模型實體名稱動態生成量化標籤（零硬編碼）：
        - potential > 0: f"{display_name}看多"
        - potential < 0: f"{display_name}看空"
        - potential >= 5.0: f"{display_name}強"
        """
        ...


class BaseModelPlugin:
    """
    可選的基礎插件骨架類別，提供預設屬性與標準勾子模板。
    """
    model_id: str = "base-model"
    plugin_id: str = "base-model"
    name: str = "base-model"
    tag_name: str = "BaseModel"
    display_name: str = "Base Time-Series Model"
    version: str = "1.0.0"
    author: str = "Unknown"
    supported_markets: List[str] = ["tw", "us", "jp"]
    default_role: str = "candidate"
    supports_covariates: bool = False
    required_covariates: List[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not getattr(self, "name", None):
            self.name = getattr(self, "plugin_id", self.model_id)
        if not hasattr(self, "supported_markets") or self.supported_markets is None:
            self.supported_markets = ["tw", "us", "jp"]
        if not hasattr(self, "required_covariates") or self.required_covariates is None:
            self.required_covariates = []

    def initialize(self, config: Dict[str, Any]) -> bool:
        return True

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "model_id": self.model_id,
            "version": self.version,
        }

    def predict_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        covariates_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        horizon: int = 5,
        market: str = "tw",
        market_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        return {}

    def teardown(self) -> None:
        pass

    def generate_tags(self, potential: float, risk_reward: Optional[float] = None) -> List[str]:
        """純粹基於模型實體名稱動態生成量化標籤（零硬編碼）"""
        tags = []
        name = getattr(self, "tag_name", None) or getattr(self, "display_name", "Model")
        if "(" in name:
            name = name.split("(")[0].strip()
        if potential > 0:
            tags.append(f"{name}看多")
            if potential >= 5.0:
                tags.append(f"{name}強")
        elif potential < 0:
            tags.append(f"{name}看空")
        return tags


class ModelPluginRegistry:
    """
    時序大模型熱插拔註冊表工廠 (Model Plugin Registry & Factory)。
    提供動態註冊、取得、列表、熱卸載與執行期健康檢查。
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, ModelPlugin] = {}
        self._classes: Dict[str, Type[ModelPlugin]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        plugin_or_cls: Union[ModelPlugin, Type[ModelPlugin]],
        override: bool = True,
    ) -> Union[ModelPlugin, Type[ModelPlugin]]:
        """
        註冊模型插件實例或類別。可作為普通方法或裝飾器使用。
        """
        with self._lock:
            if isinstance(plugin_or_cls, type):
                # Check class attribute presence
                model_id = getattr(plugin_or_cls, "model_id", None)
                if not model_id:
                    raise ValueError(f"Plugin class {plugin_or_cls.__name__} must define 'model_id'.")
                if (model_id in self._plugins or model_id in self._classes) and not override:
                    raise KeyError(f"Model ID '{model_id}' is already registered.")
                self._classes[model_id] = plugin_or_cls
                # If an instance was registered before and override is requested, clear it
                if model_id in self._plugins and override:
                    del self._plugins[model_id]
                logger.info(f"Registered model plugin class: {model_id} ({plugin_or_cls.__name__})")
            else:
                # Must satisfy ModelPlugin protocol
                if not isinstance(plugin_or_cls, ModelPlugin):
                    # Detailed diagnostic for protocol violation
                    missing = []
                    for attr in ("model_id", "display_name", "version", "author",
                                 "supported_markets", "default_role", "supports_covariates", "required_covariates"):
                        if not hasattr(plugin_or_cls, attr):
                            missing.append(attr)
                    for meth in ("initialize", "health_check", "predict_batch", "teardown", "generate_tags"):
                        if not hasattr(plugin_or_cls, meth) or not callable(getattr(plugin_or_cls, meth)):
                            missing.append(meth)
                    raise TypeError(f"Object {plugin_or_cls} does not conform to ModelPlugin protocol. Missing: {missing}")

                model_id = plugin_or_cls.model_id
                if not model_id:
                    raise ValueError("ModelPlugin instance must have non-empty 'model_id'.")
                if (model_id in self._plugins or model_id in self._classes) and not override:
                    raise KeyError(f"Model ID '{model_id}' is already registered.")
                self._plugins[model_id] = plugin_or_cls
                logger.info(f"Registered model plugin instance: {model_id} [{plugin_or_cls.display_name}]")

        return plugin_or_cls

    def register_plugin(
        self,
        plugin: Union[ModelPlugin, Type[ModelPlugin]],
        override: bool = True,
    ) -> Union[ModelPlugin, Type[ModelPlugin]]:
        """Alias for register()"""
        return self.register(plugin, override=override)

    def unregister(self, model_id: str) -> bool:
        """卸載模型插件，觸發 teardown() 資源釋放"""
        with self._lock:
            plugin = self._plugins.pop(model_id, None)
            cls_target = self._classes.pop(model_id, None)
            if plugin is not None:
                try:
                    plugin.teardown()
                except Exception as exc:
                    logger.warning(f"Error tearing down plugin '{model_id}': {exc}")
                logger.info(f"Unregistered model plugin: {model_id}")
                return True
            if cls_target is not None:
                logger.info(f"Unregistered model plugin class: {model_id}")
                return True
            return False

    def get(self, model_id: str) -> Optional[ModelPlugin]:
        """依 model_id 取得模型插件實例。若已註冊類別但未實例化，則自動延遲實例化。"""
        with self._lock:
            if model_id in self._plugins:
                return self._plugins[model_id]
            if model_id in self._classes:
                cls_type = self._classes[model_id]
                try:
                    instance = cls_type()
                    self._plugins[model_id] = instance
                    return instance
                except Exception as exc:
                    logger.error(f"Failed to instantiate model plugin '{model_id}' from class {cls_type}: {exc}")
                    return None
            return None

    def _ensure_default_plugins(self) -> None:
        if not self._plugins and not self._classes:
            try:
                import models.plugins  # noqa: F401
            except Exception:
                pass

    def list_models(self) -> List[str]:
        """列出所有已註冊之 model_id 清單"""
        self._ensure_default_plugins()
        with self._lock:
            all_ids = set(self._plugins.keys()).union(self._classes.keys())
            return sorted(all_ids)

    def get_all(self) -> Dict[str, ModelPlugin]:
        """取得所有已註冊的模型插件實例字典 (線程安全原子快照)"""
        self._ensure_default_plugins()
        with self._lock:
            result = {}
            for mid in self.list_models():
                p = self.get(mid)
                if p is not None:
                    result[mid] = p
            return result

    def list_active_plugins(self) -> List[ModelPlugin]:
        """列出所有已啟用之模型插件實例清單 (線程安全原子快照)"""
        self._ensure_default_plugins()
        with self._lock:
            active = []
            for mid in self.list_models():
                p = self.get(mid)
                if p is not None and getattr(p, "default_role", "active") == "active":
                    active.append(p)
            return active

    def clear(self) -> None:
        """清空註冊表並釋放所有外掛資源"""
        with self._lock:
            for mid, plugin in list(self._plugins.items()):
                try:
                    plugin.teardown()
                except Exception as exc:
                    logger.warning(f"Error during clear/teardown of '{mid}': {exc}")
            self._plugins.clear()
            self._classes.clear()


# Global singleton instance of ModelPluginRegistry
model_registry = ModelPluginRegistry()
