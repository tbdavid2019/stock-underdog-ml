"""
data/covariate_protocol.py - Hot-Pluggable Covariate Architecture for Elephant & Goldfish Engine.

Defines the CovariatePlugin Protocol, BaseCovariatePlugin, specialized Provider Drivers
(YahooCovariateProvider, FredCovariateProvider, TwseCovariateProvider), and the
HotPluggableCovariateRegistry with Dynamic Feature Matrix Assembly and Single-Point Fault Isolation.
"""
from __future__ import annotations

import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("stock_app.data.covariate_protocol")


@runtime_checkable
class CovariatePlugin(Protocol):
    """
    側參數金魚熱插拔插件生命週期標準協議 (Covariate Plugin Protocol).
    宣告式、秒級增刪、獨立運行的自描述特徵外掛。
    """
    covariate_id: str           # 唯一指標代碼，如 "GOLD", "BTC", "VIX", "USDTWD"
    label: str                  # 顯示標籤，如 "Comex Gold Continuous Futures"
    source_type: str            # 數據源驅動: "yahoo" | "fred" | "twse" | "custom_http"
    source_code: str            # 上游查詢符號，如 "GC=F", "BTC-USD", "^VIX", "DGS10"
    market_affinity: List[str]  # 適用市場清單，如 ["tw", "us", "jp"]
    category: str               # 類別: "macro" | "fx" | "sentiment" | "commodity" | "chips" | "crypto_liquidity"
    normalization_mode: str     # 標準化策略: "log_returns" | "rolling_zscore" | "bias_ma20"
    update_frequency: str       # 更新頻率: "daily_close" | "hourly" | "tick"

    def fetch_observations(self, start_date: str, end_date: str) -> pd.Series:
        """
        從上游抓取歷史觀測值時序，具備 Stale Cache 與降級備援機制。
        回傳以時間為索引的 pd.Series (數值型態)。
        """
        ...

    def transform(self, raw_series: pd.Series, lookback: int = 60) -> np.ndarray:
        """
        執行非線性平穩化與防前瞻截斷，輸出長度為 lookback 之標準化浮點陣列 (1D float32)。
        """
        ...

    def health_check(self) -> bool:
        """
        上游可用性探測與熔斷狀態檢查。若持續 429 或超時，回報 False 觸發平滑熱拔除。
        """
        ...


class BaseCovariatePlugin:
    """
    側參數外掛基底類別，實作非線性平穩化特徵工程、熔斷器與健康狀態機。
    """

    def __init__(
        self,
        covariate_id: str,
        label: str,
        source_type: str,
        source_code: str,
        market_affinity: List[str],
        category: str,
        normalization_mode: str = "log_returns",
        update_frequency: str = "daily_close",
        lookback: int = 60,
        fallback_policy: str = "stale_cache",
        circuit_breaker_retries: int = 3,
        custom_fetcher: Optional[Callable[[str, str], pd.Series]] = None,
        **kwargs: Any,
    ) -> None:
        self.covariate_id = covariate_id.upper()
        self.label = label
        self.source_type = source_type.lower()
        self.source_code = source_code
        self.market_affinity = [m.lower() for m in market_affinity]
        self.category = category.lower()
        self.normalization_mode = normalization_mode.lower()
        self.update_frequency = update_frequency
        self.lookback = lookback
        self.fallback_policy = fallback_policy
        self.circuit_breaker_retries = circuit_breaker_retries
        self.custom_fetcher = custom_fetcher

        # Circuit breaker and health state
        self.is_healthy: bool = True
        self.consecutive_failures: int = 0
        self.circuit_open: bool = False
        self.cached_series: Optional[pd.Series] = None

    def health_check(self) -> bool:
        """探測熔斷器狀態與健康狀態"""
        return not self.circuit_open and self.is_healthy

    def trip_circuit(self) -> None:
        """手動熔斷開關 (用於熱拔除測試或異常隔離)"""
        self.circuit_open = True
        self.is_healthy = False
        logger.warning(f"Circuit tripped for covariate: {self.covariate_id}")

    def reset_circuit(self) -> None:
        """重置熔斷狀態"""
        self.circuit_open = False
        self.consecutive_failures = 0
        self.is_healthy = True
        logger.info(f"Circuit reset for covariate: {self.covariate_id}")

    def record_success(self, series: pd.Series) -> None:
        """記錄抓取成功，刷新快取並重置失敗計數器"""
        self.consecutive_failures = 0
        self.circuit_open = False
        self.is_healthy = True
        if series is not None and len(series) > 0:
            self.cached_series = series

    def record_failure(self) -> None:
        """記錄抓取失敗，超過門檻自動開啟熔斷"""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.circuit_breaker_retries:
            self.circuit_open = True
            self.is_healthy = False
            logger.error(
                f"Covariate '{self.covariate_id}' exceeded max retries "
                f"({self.circuit_breaker_retries}). Circuit OPENED -> Hot-Unplugged."
            )

    def fetch_observations(self, start_date: str, end_date: str) -> pd.Series:
        """抓取時序觀測值，支援 custom_fetcher 與 stale cache 降級備援"""
        if self.custom_fetcher is not None:
            try:
                s = self.custom_fetcher(start_date, end_date)
                self.record_success(s)
                return s
            except Exception as exc:
                self.record_failure()
                if self.fallback_policy == "stale_cache" and self.cached_series is not None:
                    logger.warning(
                        f"Fetch failed for {self.covariate_id} ({exc}); falling back to stale cache."
                    )
                    return self.cached_series
                raise exc

        # Default fallback if no custom fetcher is attached
        if self.cached_series is not None:
            return self.cached_series

        raise NotImplementedError(
            f"No fetch implementation or cache available for covariate '{self.covariate_id}'."
        )

    def transform(self, raw_series: pd.Series, lookback: int = 60) -> np.ndarray:
        """
        執行嚴格防前瞻非線性平穩化與維度截斷。
        回傳長度恰為 lookback 的 1D float32 陣列。
        """
        effective_lookback = lookback or self.lookback or 60
        if raw_series is None or len(raw_series) == 0:
            return np.zeros(effective_lookback, dtype=np.float32)

        # 清洗無效值
        clean = (
            raw_series.astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        if self.normalization_mode == "log_returns":
            # 價格類對數收益率: log(P_t / P_{t-1})
            shifted = clean.shift(1)
            ratio = clean / shifted
            # 避免除以 0 或負數
            ratio = ratio.replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
            ratio = np.maximum(ratio, 1e-6)
            transformed = np.log(ratio).fillna(0.0)
            # 防極端離群值
            transformed = np.clip(transformed, -1.0, 1.0)

        elif self.normalization_mode == "rolling_zscore":
            # 滾動 20 日 Robust Z-Score: (X - MA20) / STD20
            window = min(20, max(3, len(clean)))
            rolling_mean = clean.rolling(window, min_periods=2).mean().bfill().ffill().fillna(clean)
            rolling_std = clean.rolling(window, min_periods=2).std().replace(0, 1e-6).bfill().ffill().fillna(1.0)
            z = (clean - rolling_mean) / rolling_std
            transformed = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            transformed = np.clip(transformed, -5.0, 5.0)

        elif self.normalization_mode == "bias_ma20":
            # 相對 20MA 乖離率百分比: (P - MA20) / MA20 * 100
            window = min(20, max(3, len(clean)))
            ma20 = clean.rolling(window, min_periods=2).mean().bfill().ffill().fillna(clean)
            bias = ((clean - ma20) / ma20.replace(0, 1e-6)) * 100.0
            transformed = bias.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            transformed = np.clip(transformed, -50.0, 50.0)

        else:
            # 預設直接標準化除以均值
            mean_val = float(clean.mean()) if float(clean.mean()) != 0 else 1.0
            transformed = ((clean - mean_val) / abs(mean_val)).fillna(0.0)

        arr = np.asarray(transformed.values, dtype=np.float32)
        # 對齊 lookback 長度 (若不足則在左側補 0，若超出則取最近 lookback 筆)
        if len(arr) >= effective_lookback:
            return arr[-effective_lookback:]
        else:
            pad = np.zeros(effective_lookback - len(arr), dtype=np.float32)
            return np.concatenate([pad, arr])


class YahooCovariateProvider(BaseCovariatePlugin):
    """
    Yahoo Finance 驅動適配器 (支援 VIX, USDTWD, USDJPY, SOX, GOLD, BTC 等)
    """

    def fetch_observations(self, start_date: str, end_date: str) -> pd.Series:
        if self.custom_fetcher is not None:
            return super().fetch_observations(start_date, end_date)

        try:
            # Try utilizing existing data.covariates or direct requests if available
            import yfinance as yf  # type: ignore
            ticker_obj = yf.Ticker(self.source_code)
            hist = ticker_obj.history(start=start_date, end=end_date)
            if hist.empty or "Close" not in hist.columns:
                raise ValueError(f"Yahoo query for '{self.source_code}' returned empty history.")
            s = hist["Close"].astype(float)
            self.record_success(s)
            return s
        except Exception as exc:
            self.record_failure()
            if self.fallback_policy == "stale_cache" and self.cached_series is not None:
                logger.warning(
                    f"Yahoo fetch failed for {self.covariate_id} ({exc}); returning stale cache."
                )
                return self.cached_series
            logger.warning(f"Yahoo fetch failed for {self.covariate_id}: {exc}")
            raise exc


class FredCovariateProvider(BaseCovariatePlugin):
    """
    FRED (Federal Reserve Economic Data) 總經數據源驅動適配器 (美債殖利率、聯準會利率)
    """

    def fetch_observations(self, start_date: str, end_date: str) -> pd.Series:
        if self.custom_fetcher is not None:
            return super().fetch_observations(start_date, end_date)

        try:
            # FRED Direct CSV endpoint
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={self.source_code}"
            import requests  # type: ignore
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"FRED HTTP {resp.status_code}")
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
            df = df.set_index("DATE").sort_index()
            val_col = [c for c in df.columns if c != "DATE"][0]
            s = pd.to_numeric(df[val_col], errors="coerce").dropna()

            # 防前瞻偏誤：依 start_date 與 end_date 嚴格截斷，禁止未來資訊洩漏
            if start_date:
                s = s[s.index >= pd.to_datetime(start_date).tz_localize(None)]
            if end_date:
                s = s[s.index <= pd.to_datetime(end_date).tz_localize(None)]

            self.record_success(s)
            return s
        except Exception as exc:
            self.record_failure()
            if self.fallback_policy == "stale_cache" and self.cached_series is not None:
                logger.warning(
                    f"FRED fetch failed for {self.covariate_id} ({exc}); returning stale cache."
                )
                return self.cached_series
            logger.warning(f"FRED fetch failed for {self.covariate_id}: {exc}")
            raise exc


class TwseCovariateProvider(BaseCovariatePlugin):
    """
    台灣證交所 / 法人籌碼特徵驅動適配器
    """

    def fetch_observations(self, start_date: str, end_date: str) -> pd.Series:
        if self.custom_fetcher is not None:
            return super().fetch_observations(start_date, end_date)

        try:
            from data.institutional import InstitutionalProvider
            # Institutional provider interface
            raise NotImplementedError("TWSE direct institutional time-series fetcher in progress; please use custom_fetcher or DuckDB store.")
        except Exception as exc:
            self.record_failure()
            if self.fallback_policy == "stale_cache" and self.cached_series is not None:
                return self.cached_series
            logger.warning(f"TWSE fetch failed for {self.covariate_id}: {exc}")
            raise exc


class HotPluggableCovariateRegistry:
    """
    側參數動態插件註冊表與特徵矩陣裝配引擎 (Hot-Pluggable Covariate Registry & Feature Assembler)。
    具備：
      1. 宣告式 YAML 熱載入與比對雜湊無重啟熱更新 (Hot-Reload without Restart)。
      2. 動態特徵矩陣裝配器 (Dynamic Feature Matrix Assembler)。
      3. 單點故障熱拔除與安全隔離降級 (Fault Isolation & Hot-Unplug)。
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, CovariatePlugin] = {}
        self._yaml_cids: set = set()
        self._yaml_hash: Optional[str] = None
        self._yaml_path: Optional[str] = None
        self._lock = threading.Lock()

    def register_plugin(self, plugin: CovariatePlugin) -> CovariatePlugin:
        """註冊側參數插件實例"""
        if not isinstance(plugin, CovariatePlugin):
            raise TypeError(f"Object {plugin} does not conform to CovariatePlugin protocol.")
        cid = plugin.covariate_id.upper()
        with self._lock:
            self._plugins[cid] = plugin
            logger.info(f"Registered covariate plugin: {cid} [{plugin.label}]")
        return plugin

    def unregister_plugin(self, covariate_id: str) -> bool:
        """熱拔除卸載側參數外掛"""
        cid = covariate_id.upper()
        with self._lock:
            if cid in self._plugins:
                del self._plugins[cid]
                logger.info(f"Unregistered covariate plugin: {cid}")
                return True
            return False

    def get_plugin(self, covariate_id: str) -> Optional[CovariatePlugin]:
        """依 ID 取得側參數外掛"""
        with self._lock:
            return self._plugins.get(covariate_id.upper())

    def list_plugins(self, market: Optional[str] = None) -> List[CovariatePlugin]:
        """列出側參數外掛。若指定 market 則過濾市場親和性 (market_affinity)。"""
        with self._lock:
            plugins = list(self._plugins.values())
        if market:
            m_lower = market.lower()
            return [p for p in plugins if m_lower in [ma.lower() for ma in p.market_affinity]]
        return plugins

    def clear(self) -> None:
        """清空註冊表"""
        with self._lock:
            self._plugins.clear()
            self._yaml_cids.clear()
            self._yaml_hash = None
            self._yaml_path = None

    def load_from_yaml(self, yaml_path: Union[str, Path]) -> int:
        """
        從 YAML 設定檔宣告式載入側參數定義並自動實例化適配驅動。
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Covariates registry YAML file not found: {path}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        data = yaml.safe_load(raw_bytes)
        items = data.get("covariates_registry", []) if isinstance(data, dict) else []

        new_yaml_cids: set = set()
        loaded_count = 0
        with self._lock:
            for item in items:
                cid = item.get("id")
                if not cid:
                    continue
                cid_upper = cid.upper()
                new_yaml_cids.add(cid_upper)
                source_type = (item.get("source_type") or "yahoo").lower()

                kwargs = {
                    "covariate_id": cid,
                    "label": item.get("label", cid),
                    "source_type": source_type,
                    "source_code": item.get("source_code", cid),
                    "market_affinity": item.get("market_affinity", ["tw", "us", "jp"]),
                    "category": item.get("category", "macro"),
                    "normalization_mode": item.get("normalization_mode", "log_returns"),
                    "update_frequency": item.get("update_frequency", "daily_close"),
                    "lookback": item.get("lookback", 60),
                    "fallback_policy": item.get("fallback_policy", "stale_cache"),
                    "circuit_breaker_retries": item.get("circuit_breaker_retries", 3),
                }

                # 依 source_type 裝配驅動
                if source_type == "yahoo":
                    plugin = YahooCovariateProvider(**kwargs)
                elif source_type == "fred":
                    plugin = FredCovariateProvider(**kwargs)
                elif source_type == "twse":
                    plugin = TwseCovariateProvider(**kwargs)
                else:
                    plugin = BaseCovariatePlugin(**kwargs)

                self._plugins[cid_upper] = plugin
                loaded_count += 1

            # 熱拔除：清理前次由 YAML 載入，但在新 YAML 中已被刪除的舊外掛
            removed_cids = self._yaml_cids - new_yaml_cids
            for old_cid in removed_cids:
                if old_cid in self._plugins:
                    del self._plugins[old_cid]
                    logger.info(f"Unloaded deleted covariate plugin: {old_cid}")

            self._yaml_cids = new_yaml_cids
            self._yaml_hash = file_hash
            self._yaml_path = str(path.resolve())

        logger.info(f"Loaded {loaded_count} covariates from {yaml_path} (MD5: {file_hash[:8]})")
        return loaded_count

    def reload_if_changed(self, yaml_path: Optional[Union[str, Path]] = None) -> bool:
        """
        比對 YAML 雜湊值；若檔案內容變更則執行無縫熱重載 (Hot-Reload without Restart)。
        """
        target_path = Path(yaml_path or self._yaml_path or "config/covariates_registry.yaml")
        if not target_path.exists():
            return False

        current_hash = hashlib.md5(target_path.read_bytes()).hexdigest()
        if current_hash != self._yaml_hash:
            logger.info(f"Covariates YAML changed ({self._yaml_hash} -> {current_hash}). Hot-reloading...")
            self.load_from_yaml(target_path)
            return True
        return False

    def assemble_feature_matrix(
        self,
        market: str,
        start_date: str,
        end_date: str,
        lookback: int = 60,
        common_dates: Optional[Union[pd.DatetimeIndex, List[Any]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        動態特徵矩陣裝配器 (Dynamic Feature Matrix Assembler)
        依目標市場自動過濾指標，並實作單點隔離熱拔除 (Fault Isolation & Hot-Unplug)。

        回傳:
          - (T, K) 浮點特徵矩陣，其中 T = lookback, K = 成功組裝之特徵數。
          - 成功啟用的 covariate_id 清單 (長度 K)。
        若所有指標均故障，平滑降級回傳 shape=(lookback, 0) 之空陣列，保證決策管線不中斷。
        """
        matching_plugins = self.list_plugins(market=market)
        if not matching_plugins:
            return np.empty((lookback, 0), dtype=np.float32), []

        feature_cols: List[np.ndarray] = []
        active_cids: List[str] = []

        # 準備基準交易日曆以對齊不同資產 (避免 24/7 加密貨幣與平日股市交易日錯位)
        common_idx = None
        if common_dates is not None and len(common_dates) > 0:
            try:
                common_idx = pd.to_datetime(common_dates).tz_localize(None).normalize()
            except Exception:
                common_idx = None
        elif start_date and end_date:
            try:
                common_idx = pd.bdate_range(start=start_date, end=end_date)
            except Exception:
                common_idx = None

        for plugin in matching_plugins:
            cid = plugin.covariate_id

            # [防護 1: 探針健康檢查與熔斷判斷]
            if not plugin.health_check():
                logger.warning(
                    f"⚠️ [Hot-Unplug] Covariate '{cid}' failed health_check or circuit open. "
                    f"Isolating feature; inference pipeline proceeds."
                )
                continue

            # [防護 2: 上游抓取隔離 - 捕捉超時、429 與網路例外]
            try:
                obs_series = plugin.fetch_observations(start_date, end_date)
            except Exception as exc:
                logger.warning(
                    f"⚠️ [Hot-Unplug] Covariate '{cid}' fetch failed: {exc}. "
                    f"Dynamically unplugging feature to prevent cascading collapse."
                )
                # Provider.fetch_observations 內部已累加失敗次數，此處不重複累加
                continue

            if obs_series is None or len(obs_series) == 0:
                logger.warning(f"⚠️ [Hot-Unplug] Covariate '{cid}' returned empty observations.")
                continue

            # [防護 3: 跨市場時間軸對齊與重採樣]
            try:
                if common_idx is not None and len(common_idx) > 0:
                    aligned_s = obs_series.copy()
                    aligned_s.index = pd.to_datetime(aligned_s.index).tz_localize(None).normalize()
                    aligned_s = aligned_s[~aligned_s.index.duplicated(keep="last")]
                    obs_series = aligned_s.reindex(common_idx).ffill().fillna(0.0)
            except Exception as e:
                logger.debug(f"Calendar alignment fallback for '{cid}': {e}")

            # [防護 4: 平穩化轉換防前瞻截斷]
            try:
                transformed_arr = plugin.transform(obs_series, lookback=lookback)
            except Exception as exc:
                logger.warning(f"⚠️ [Hot-Unplug] Covariate '{cid}' transform failed: {exc}.")
                continue

            if not isinstance(transformed_arr, np.ndarray) or transformed_arr.shape != (lookback,):
                logger.warning(f"⚠️ [Hot-Unplug] Covariate '{cid}' invalid shape: {getattr(transformed_arr, 'shape', None)}.")
                continue

            if not np.isfinite(transformed_arr).all():
                logger.warning(f"⚠️ [Hot-Unplug] Covariate '{cid}' contains non-finite values.")
                continue

            feature_cols.append(transformed_arr)
            active_cids.append(cid)

        # 組裝 (T, K) 矩陣
        if not feature_cols:
            logger.info(
                f"ℹ️ All covariates unavailable for market '{market}'. Smoothly fallback to price-only (T, 0)."
            )
            return np.empty((lookback, 0), dtype=np.float32), []

        matrix = np.column_stack(feature_cols).astype(np.float32)
        return matrix, active_cids


# Global singleton instance of HotPluggableCovariateRegistry
covariate_registry = HotPluggableCovariateRegistry()
