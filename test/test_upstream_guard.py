"""
test/test_upstream_guard.py - 測試多金鑰輪詢池、併發限制與失敗熔斷守衛
"""
import time
import unittest
from core.upstream_guard import KeyPoolGuard, RecentFailureCooldownError, mask_key


class TestUpstreamGuard(unittest.TestCase):
    def test_multi_key_round_robin_rotation(self):
        """測試以逗號分隔多金鑰之輪詢分配"""
        guard = KeyPoolGuard(keys="key_alpha, key_beta, key_gamma", name="test_pool")
        self.assertEqual(guard.key_count, 3)

        keys_drawn = [guard.get_next_key() for _ in range(6)]
        expected = ["key_alpha", "key_beta", "key_gamma", "key_alpha", "key_beta", "key_gamma"]
        self.assertEqual(keys_drawn, expected)

    def test_in_flight_concurrency_throttle(self):
        """測試單金鑰併發上限閘門"""
        guard = KeyPoolGuard(keys="single_key", max_in_flight_per_key=1, acquire_timeout=0.05)

        with guard.acquire() as k1:
            self.assertEqual(k1, "single_key")
            # 在同一金鑰持有鎖期間，再次請求應於 acquire_timeout 後拋出 TimeoutError
            with self.assertRaises(TimeoutError):
                with guard.acquire():
                    pass

        # 釋放後應可再次成功獲取
        with guard.acquire() as k2:
            self.assertEqual(k2, "single_key")

    def test_failure_cooldown_circuit_breaker(self):
        """測試 60 秒失敗熔斷冷卻期與快速失敗機制"""
        guard = KeyPoolGuard(keys="key1", cooldown_seconds=0.2, name="cooldown_test")
        self.assertFalse(guard.is_in_cooldown())

        guard.record_failure()
        self.assertTrue(guard.is_in_cooldown())

        # 冷卻期間應立即拋出 RecentFailureCooldownError
        with self.assertRaises(RecentFailureCooldownError):
            with guard.acquire():
                pass

        # 等待冷卻結束後自動恢復
        time.sleep(0.25)
        self.assertFalse(guard.is_in_cooldown())
        with guard.acquire() as key:
            self.assertEqual(key, "key1")

    def test_record_success_resets_cooldown(self):
        """測試成功請求可立即清除冷卻"""
        guard = KeyPoolGuard(keys="key1", cooldown_seconds=10.0)
        guard.record_failure()
        self.assertTrue(guard.is_in_cooldown())

        guard.record_success()
        self.assertFalse(guard.is_in_cooldown())

    def test_mask_key_security(self):
        """測試金鑰遮罩絕不輸出明文"""
        self.assertEqual(mask_key(""), "[EMPTY]")
        self.assertEqual(mask_key(None), "[EMPTY]")
        self.assertEqual(mask_key("12345"), "***")
        masked = mask_key("sk-proj-1234567890abcdef")
        self.assertTrue(masked.startswith("sk-"))
        self.assertTrue(masked.endswith("def"))
        self.assertNotIn("1234567890", masked)

        # URL with query/token masking
        url_with_token = "https://example.com/api?token=secret123456789"
        masked_url = mask_key(url_with_token)
        self.assertNotIn("secret123456789", masked_url)
        self.assertEqual(masked_url, "example.com")

    def test_cooldown_error_masks_target_and_does_not_leak_keys(self):
        """測試熔斷冷卻例外字串遮罩敏感金鑰"""
        guard = KeyPoolGuard(keys="super_sensitive_api_key_xyz", cooldown_seconds=10.0, name="cooldown_mask_test")
        guard.record_failure()
        with self.assertRaises(RecentFailureCooldownError) as ctx:
            with guard.acquire():
                pass
        err_msg = str(ctx.exception)
        self.assertNotIn("super_sensitive_api_key_xyz", err_msg)


if __name__ == "__main__":
    unittest.main()
