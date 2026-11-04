"""
test/test_hot_pluggable_covariates.py - Comprehensive Unit Tests for Hot-Pluggable Covariate Pipeline.

Validates:
1. Declarative loading from config/covariates_registry.yaml and CovariatePlugin protocol compliance.
2. Dynamic Feature Matrix Assembler across markets (TW, US, JP) with dimensional self-adaptation.
3. Single-Point Fault Isolation & Hot-Unplug (Mock 429 / network error; matrix reduces from (T, K) to (T, K-1)).
4. Full outage smooth fallback to price-only (T, 0) without pipeline disruption.
5. Hot-Reload without restart via file MD5 hash comparison.
6. Transformation modes (log_returns, rolling_zscore, bias_ma20).
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import yaml

from data.covariate_protocol import (
    BaseCovariatePlugin,
    CovariatePlugin,
    HotPluggableCovariateRegistry,
    YahooCovariateProvider,
    FredCovariateProvider,
    TwseCovariateProvider,
)


def _make_dummy_series(days: int = 100, base: float = 100.0, trend: float = 0.1) -> pd.Series:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    vals = base + np.arange(days) * trend + np.sin(np.arange(days)) * 2.0
    return pd.Series(vals, index=dates)


class TestHotPluggableCovariates(unittest.TestCase):

    def setUp(self):
        self.registry = HotPluggableCovariateRegistry()
        self.yaml_path = Path("config/covariates_registry.yaml")

    def test_declarative_yaml_loading_and_protocol_compliance(self):
        """驗證從 YAML 宣告式載入側參數，且所有實例皆符合 CovariatePlugin 協議"""
        self.assertTrue(self.yaml_path.exists())
        count = self.registry.load_from_yaml(self.yaml_path)
        self.assertEqual(count, 7)

        # 檢驗宣告的 7 大側參數
        expected_ids = {"GOLD", "BITCOIN", "VIX", "USDTWD", "USDJPY", "UST_10Y", "SOX"}
        loaded_plugins = self.registry.list_plugins()
        self.assertEqual({p.covariate_id for p in loaded_plugins}, expected_ids)

        for p in loaded_plugins:
            self.assertTrue(isinstance(p, CovariatePlugin))
            self.assertTrue(hasattr(p, "covariate_id"))
            self.assertTrue(hasattr(p, "label"))
            self.assertTrue(hasattr(p, "source_type"))
            self.assertTrue(hasattr(p, "source_code"))
            self.assertIsInstance(p.market_affinity, list)
            self.assertTrue(hasattr(p, "category"))
            self.assertIn(p.normalization_mode, ["log_returns", "rolling_zscore", "bias_ma20"])
            self.assertTrue(callable(getattr(p, "fetch_observations")))
            self.assertTrue(callable(getattr(p, "transform")))
            self.assertTrue(callable(getattr(p, "health_check")))

    def test_cross_market_dimension_adaptation(self):
        """驗證動態特徵矩陣裝配器之跨市場維度自適應 (TW: 6, US: 4, JP: 5)"""
        self.registry.load_from_yaml(self.yaml_path)

        # 注入 Mock fetcher 供所有指標回傳合法數據
        for p in self.registry.list_plugins():
            p.custom_fetcher = lambda s, e, pid=p.covariate_id: _make_dummy_series(80)

        # 1. 台股市場裝配 (GOLD, BITCOIN, VIX, USDTWD, UST_10Y, SOX) -> 6 維
        tw_mat, tw_cids = self.registry.assemble_feature_matrix("tw", "2026-01-01", "2026-03-01", lookback=60)
        self.assertEqual(tw_mat.shape, (60, 6))
        self.assertEqual(len(tw_cids), 6)
        self.assertIn("USDTWD", tw_cids)
        self.assertNotIn("USDJPY", tw_cids)

        # 2. 美股市場裝配 (GOLD, BITCOIN, VIX, UST_10Y) -> 4 維
        us_mat, us_cids = self.registry.assemble_feature_matrix("us", "2026-01-01", "2026-03-01", lookback=60)
        self.assertEqual(us_mat.shape, (60, 4))
        self.assertEqual(len(us_cids), 4)
        self.assertNotIn("USDTWD", us_cids)
        self.assertNotIn("SOX", us_cids)

        # 3. 日股市場裝配 (GOLD, VIX, USDJPY, UST_10Y, SOX) -> 5 維
        jp_mat, jp_cids = self.registry.assemble_feature_matrix("jp", "2026-01-01", "2026-03-01", lookback=60)
        self.assertEqual(jp_mat.shape, (60, 5))
        self.assertEqual(len(jp_cids), 5)
        self.assertIn("USDJPY", jp_cids)
        self.assertNotIn("USDTWD", jp_cids)

    def test_single_point_fault_isolation_and_hot_unplug(self):
        """
        驗證單點故障熱拔除 (Fault Isolation & Hot-Unplug)：
        Mock 單一 Provider (如 FRED 美債 UST_10Y) 拋出 HTTP 429 或超時，
        驗證特徵矩陣從 (60, 6) 安全縮減為 (60, 5)，其餘 5 個特徵照常注入，絕不發生級聯崩潰。
        """
        self.registry.load_from_yaml(self.yaml_path)

        for p in self.registry.list_plugins():
            if p.covariate_id == "UST_10Y":
                def failing_fetcher(s, e):
                    raise RuntimeError("HTTP 429 Too Many Requests (Rate limited)")
                p.custom_fetcher = failing_fetcher
            else:
                p.custom_fetcher = lambda s, e, pid=p.covariate_id: _make_dummy_series(80)

        # 執行台股特徵矩陣裝配
        tw_mat, tw_cids = self.registry.assemble_feature_matrix("tw", "2026-01-01", "2026-03-01", lookback=60)

        # UST_10Y 遭安全熱拔除，維度由 6 自動縮減為 5
        self.assertEqual(tw_mat.shape, (60, 5))
        self.assertEqual(len(tw_cids), 5)
        self.assertNotIn("UST_10Y", tw_cids)
        self.assertIn("GOLD", tw_cids)
        self.assertIn("VIX", tw_cids)
        self.assertIn("USDTWD", tw_cids)
        self.assertTrue(np.isfinite(tw_mat).all())

    def test_circuit_tripped_feature_isolated(self):
        """驗證當外掛主動回報 health_check=False 時，裝配器自動繞道排除該特徵"""
        self.registry.load_from_yaml(self.yaml_path)
        for p in self.registry.list_plugins():
            p.custom_fetcher = lambda s, e: _make_dummy_series(80)

        # 手動熔斷 GOLD
        gold_plugin = self.registry.get_plugin("GOLD")
        self.assertIsNotNone(gold_plugin)
        gold_plugin.trip_circuit()
        self.assertFalse(gold_plugin.health_check())

        # 裝配台股矩陣
        tw_mat, tw_cids = self.registry.assemble_feature_matrix("tw", "2026-01-01", "2026-03-01", lookback=60)
        self.assertEqual(tw_mat.shape, (60, 5))
        self.assertNotIn("GOLD", tw_cids)

        # 重置熔斷
        gold_plugin.reset_circuit()
        self.assertTrue(gold_plugin.health_check())
        tw_mat_recovered, tw_cids_recovered = self.registry.assemble_feature_matrix("tw", "2026-01-01", "2026-03-01", lookback=60)
        self.assertEqual(tw_mat_recovered.shape, (60, 6))
        self.assertIn("GOLD", tw_cids_recovered)

    def test_all_covariates_outage_smooth_fallback(self):
        """驗證當所有外部側參數全面癱瘓時，平滑降級為 Tier 3 (T, 0) 純收盤價，決策永不中斷"""
        self.registry.load_from_yaml(self.yaml_path)

        # 全部 Provider 拋出例外
        for p in self.registry.list_plugins():
            p.custom_fetcher = MagicMock(side_effect=Exception("Total upstream blackout"))

        matrix, cids = self.registry.assemble_feature_matrix("tw", "2026-01-01", "2026-03-01", lookback=60)
        self.assertEqual(matrix.shape, (60, 0))
        self.assertEqual(len(cids), 0)

    def test_hot_reload_on_yaml_change(self):
        """驗證監聽 YAML MD5 雜湊變更即時熱重載 (Hot-Reload without Restart)"""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            initial_data = {
                "covariates_registry": [
                    {
                        "id": "TEMP1",
                        "label": "Temp 1",
                        "source_type": "yahoo",
                        "source_code": "T1",
                        "market_affinity": ["tw"],
                        "category": "macro",
                    }
                ]
            }
            yaml.safe_dump(initial_data, tmp)

        try:
            # 首次載入
            c = self.registry.load_from_yaml(tmp_path)
            self.assertEqual(c, 1)
            self.assertIsNotNone(self.registry.get_plugin("TEMP1"))
            self.assertIsNone(self.registry.get_plugin("TEMP2"))

            # 檔案未更動時 reload 回傳 False
            self.assertFalse(self.registry.reload_if_changed(tmp_path))

            # 修改 YAML 新增 TEMP2
            initial_data["covariates_registry"].append({
                "id": "TEMP2",
                "label": "Temp 2",
                "source_type": "yahoo",
                "source_code": "T2",
                "market_affinity": ["tw"],
                "category": "macro",
            })
            with open(tmp_path, "w") as f:
                yaml.safe_dump(initial_data, f)

            # 比對雜湊並熱重載
            reloaded = self.registry.reload_if_changed(tmp_path)
            self.assertTrue(reloaded)
            self.assertIsNotNone(self.registry.get_plugin("TEMP2"))
            self.assertEqual(len(self.registry.list_plugins()), 2)

            # 驗證拔除：從 YAML 刪除 TEMP1，熱重載後應自動自註冊表中移除
            initial_data["covariates_registry"] = [
                c for c in initial_data["covariates_registry"] if c["id"] != "TEMP1"
            ]
            with open(tmp_path, "w") as f:
                yaml.safe_dump(initial_data, f)
            reloaded_del = self.registry.reload_if_changed(tmp_path)
            self.assertTrue(reloaded_del)
            self.assertIsNone(self.registry.get_plugin("TEMP1"))
            self.assertIsNotNone(self.registry.get_plugin("TEMP2"))
            self.assertEqual(len(self.registry.list_plugins()), 1)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_calendar_alignment_crypto_vs_stocks(self):
        """驗證 24/7 加密貨幣 (365天) 與平日股票交易日 (252天) 裝配時嚴格對齊基準時間軸"""
        b_dates = pd.bdate_range("2026-01-01", "2026-03-31")
        all_dates = pd.date_range("2026-01-01", "2026-03-31", freq="D")

        s_stock = pd.Series(np.linspace(100, 200, len(b_dates)), index=b_dates)
        s_crypto = pd.Series(np.linspace(1000, 2000, len(all_dates)), index=all_dates)

        p_stock = BaseCovariatePlugin(
            covariate_id="STOCK", label="Stock Proxy", source_type="yahoo",
            source_code="STK", market_affinity=["tw"], category="macro",
            custom_fetcher=lambda s, e: s_stock
        )
        p_crypto = BaseCovariatePlugin(
            covariate_id="CRYPTO", label="Crypto Proxy", source_type="yahoo",
            source_code="BTC", market_affinity=["tw"], category="macro",
            custom_fetcher=lambda s, e: s_crypto
        )

        reg = HotPluggableCovariateRegistry()
        reg.register_plugin(p_stock)
        reg.register_plugin(p_crypto)

        # 傳入工作日時間索引，確認兩者組裝後輸出 shape=(60, 2) 且皆無 NaN
        mat, cids = reg.assemble_feature_matrix("tw", "2026-01-01", "2026-03-31", lookback=60, common_dates=b_dates)
        self.assertEqual(mat.shape, (60, 2))
        self.assertEqual(cids, ["STOCK", "CRYPTO"])
        self.assertTrue(np.isfinite(mat).all())

    def test_fred_date_filtering_no_lookahead(self):
        """驗證 FRED Provider 嚴格依 start_date 與 end_date 截斷，杜絕歷史前瞻偏誤"""
        from data.covariate_protocol import FredCovariateProvider
        provider = FredCovariateProvider(
            covariate_id="TNX", label="10Y Yield", source_type="fred",
            source_code="DGS10", market_affinity=["us"], category="macro"
        )
        # Mock FRED HTTP response
        csv_content = (
            "DATE,DGS10\n"
            "2026-01-02,4.10\n"
            "2026-01-05,4.15\n"
            "2026-01-06,4.18\n"
            "2026-01-07,4.25\n"
            "2026-01-08,4.30\n"
        )
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = csv_content
            mock_get.return_value = mock_resp

            s = provider.fetch_observations("2026-01-02", "2026-01-06")
            self.assertEqual(len(s), 3)
            self.assertEqual(s.index.max(), pd.Timestamp("2026-01-06"))

    def test_transformation_modes(self):
        """驗證三種平穩化數學轉換 (log_returns, rolling_zscore, bias_ma20) 正確性與防前瞻截斷"""
        series = _make_dummy_series(100, base=100.0, trend=0.5)

        # 1. log_returns
        p_log = BaseCovariatePlugin(
            covariate_id="LOG", label="Log Returns", source_type="yahoo",
            source_code="L", market_affinity=["tw"], category="macro",
            normalization_mode="log_returns"
        )
        arr_log = p_log.transform(series, lookback=50)
        self.assertEqual(arr_log.shape, (50,))
        self.assertTrue(np.isfinite(arr_log).all())
        self.assertTrue(np.all(arr_log >= -1.0) and np.all(arr_log <= 1.0))

        # 2. rolling_zscore
        p_z = BaseCovariatePlugin(
            covariate_id="Z", label="Z-score", source_type="yahoo",
            source_code="Z", market_affinity=["tw"], category="macro",
            normalization_mode="rolling_zscore"
        )
        arr_z = p_z.transform(series, lookback=50)
        self.assertEqual(arr_z.shape, (50,))
        self.assertTrue(np.isfinite(arr_z).all())
        self.assertTrue(np.all(arr_z >= -5.0) and np.all(arr_z <= 5.0))

        # 3. bias_ma20
        p_bias = BaseCovariatePlugin(
            covariate_id="BIAS", label="Bias MA20", source_type="yahoo",
            source_code="B", market_affinity=["tw"], category="macro",
            normalization_mode="bias_ma20"
        )
        arr_bias = p_bias.transform(series, lookback=50)
        self.assertEqual(arr_bias.shape, (50,))
        self.assertTrue(np.isfinite(arr_bias).all())
        self.assertTrue(np.all(arr_bias >= -50.0) and np.all(arr_bias <= 50.0))

        # 4. Short series padding
        short_series = _make_dummy_series(10)
        arr_padded = p_log.transform(short_series, lookback=60)
        self.assertEqual(arr_padded.shape, (60,))
        # Leading 50 entries should be 0.0 padding
        np.testing.assert_array_equal(arr_padded[:50], np.zeros(50, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
