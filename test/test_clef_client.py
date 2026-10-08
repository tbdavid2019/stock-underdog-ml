"""
test/test_clef_client.py - Unit tests for Clef-Flash System One Decision Client
"""

import unittest
from unittest.mock import patch, MagicMock
from data.clef_client import ClefDecisionClient, ClefSlot, ClefDecisionVerdict


class TestClefDecisionClient(unittest.TestCase):

    def setUp(self):
        self.client = ClefDecisionClient()

    def test_format_stock_state(self):
        cand = {
            "ticker": "2330.TW",
            "current_price": 1020.0,
            "composite_score": 85.0,
            "hit_strategies": ["xuantie", "lstm"],
            "tags": ["🏆三重共振", "玄鐵買點"],
            "fundamentals": {"pe": 21.5, "pb": 4.5},
            "ma60": 1000.0,
            "pullback_type": "MA60",
            "lstm_potential": 5.2,
            "timesfm_potential": 3.1,
            "risk_reward_ratio": 2.2,
            "institutional": {
                "foreign_net_5d": 8000,
                "trust_streak": 4,
                "is_sync_buy": True
            }
        }
        state = ClefDecisionClient.format_stock_state(cand, macro_regime="bull")
        self.assertEqual(state["ticker"], "2330.TW")
        self.assertEqual(state["macro_regime"], "bull")
        self.assertEqual(state["valuation"]["pe"], 21.5)
        self.assertEqual(state["models"]["lstm_potential"], 5.2)
        self.assertTrue(state["institutional"]["is_sync_buy"])

    @patch("requests.post")
    def test_evaluate_stock_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {
                    "type": "choice",
                    "choice": "strong_buy",
                    "probabilities": {"strong_buy": 0.85, "buy": 0.10, "hold_watch": 0.04, "avoid": 0.01},
                    "confidence": 0.85
                },
                "conviction_score": {
                    "type": "score",
                    "score": 4.2,
                    "confidence": 0.8
                },
                "is_favorable_entry": {
                    "type": "noul",
                    "noul": 0.78
                }
            }
        }
        mock_post.return_value = mock_response

        cand = {"ticker": "2330.TW", "current_price": 1000.0}
        verdict = self.client.evaluate_stock(cand)

        self.assertTrue(verdict.success)
        self.assertEqual(verdict.ticker, "2330.TW")
        self.assertEqual(verdict.action_decision, "strong_buy")
        self.assertEqual(verdict.conviction_score, 4.2)
        self.assertIn("🤖AI強買", verdict.tag)
        self.assertEqual(verdict.source_slot, "Clef-Create360")

    @patch("requests.post")
    def test_evaluate_stock_fallback(self, mock_post):
        # First slot fails, second slot succeeds
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.text = "Service Unavailable"

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {
                    "choice": "buy",
                    "probabilities": {"buy": 0.70, "avoid": 0.30},
                    "confidence": 0.70
                },
                "conviction_score": {"score": 3.5},
                "is_favorable_entry": {"noul": 0.65}
            }
        }
        mock_post.side_effect = [fail_resp, success_resp]

        cand = {"ticker": "2454.TW"}
        verdict = self.client.evaluate_stock(cand)

        self.assertTrue(verdict.success)
        self.assertEqual(verdict.action_decision, "buy")
        self.assertEqual(verdict.source_slot, "Clef-Aiurl")

    @patch("requests.post")
    def test_evaluate_stock_all_fail_graceful(self, mock_post):
        # All slots fail -> gracefully degrades to local rules fallback
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.text = "Internal Server Error"
        mock_post.return_value = fail_resp

        cand = {
            "ticker": "2330.TW",
            "composite_score": 88.0,
            "lstm_potential": 4.5,
            "timesfm_potential": 3.2,
            "risk_reward_ratio": 2.1,
            "institutional": {"is_sync_buy": True}
        }
        verdict = self.client.evaluate_stock(cand)

        # 降級至本地量化規則推論，仍可提供可用決策
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.source_slot, "local_rules")
        self.assertEqual(verdict.action_decision, "strong_buy")
        self.assertIn("Degraded to local rules fallback", verdict.error)
        self.assertIn("本地規則", verdict.tag)

    def test_local_rules_bearish_avoid(self):
        """驗證當模型大幅看跌時，本地規則降級輸出 avoid 決策"""
        bearish_cand = {
            "ticker": "1303.TW",
            "composite_score": 40.0,
            "lstm_potential": -12.0,
            "timesfm_potential": -2.0,
            "risk_reward_ratio": 0.5
        }
        client_disabled = ClefDecisionClient(enabled=False)
        verdict = client_disabled.evaluate_stock(bearish_cand)
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.action_decision, "avoid")
        self.assertEqual(verdict.source_slot, "local_rules")
        self.assertIn("🔻避開", verdict.tag)

    def test_slot_deduplication_and_empty_filtering(self):
        """驗證空 URL 或與前幾級重複的端點會被自動過濾，避免向同一個掛掉的 Host 重複發送導致額外超時"""
        custom_slots = [
            ClefSlot(name="Primary", url="https://primary.ai/v1/systemone"),
            ClefSlot(name="Fallback1", url="https://fallback.ai/v1/systemone"),
            ClefSlot(name="DuplicatePrimary", url="https://primary.ai/v1/systemone"),  # 重複端點
            ClefSlot(name="EmptySlot", url="")  # 空端點
        ]
        client = ClefDecisionClient(slots=custom_slots)
        self.assertEqual(len(client.slots), 2)
        self.assertEqual(client.slots[0].name, "Primary")
        self.assertEqual(client.slots[1].name, "Fallback1")

    def test_evaluate_candidates_batch_disabled_returns_local_decisions(self):
        """驗證當 Clef 遠端推論停用時，批次評估 API 仍能保留並回傳每支候選標的的本地規則決策"""
        candidates = [
            {
                "ticker": "2330.TW",
                "composite_score": 88.0,
                "lstm_potential": 4.5,
                "timesfm_potential": 3.0,
                "risk_reward_ratio": 2.0,
                "institutional": {"is_sync_buy": True}
            },
            {
                "ticker": "1303.TW",
                "composite_score": 35.0,
                "lstm_potential": -10.0,
                "timesfm_potential": -1.5,
                "risk_reward_ratio": 0.5
            }
        ]
        client_disabled = ClefDecisionClient(enabled=False)
        batch_results = client_disabled.evaluate_candidates_batch(candidates)

        self.assertEqual(len(batch_results), 2)
        self.assertIn("2330.TW", batch_results)
        self.assertIn("1303.TW", batch_results)
        self.assertEqual(batch_results["2330.TW"].action_decision, "strong_buy")
        self.assertEqual(batch_results["2330.TW"].source_slot, "local_rules")
        self.assertEqual(batch_results["1303.TW"].action_decision, "avoid")
        self.assertEqual(batch_results["1303.TW"].source_slot, "local_rules")

    def test_evaluate_local_rules_non_numeric_metrics(self):
        """驗證當自定義 state 中的數值欄位為非數字字串、NaN 或 None 時，本地規則評估不會拋出 ValueError"""
        client = ClefDecisionClient(enabled=False)
        malformed_state = {
            "ticker": "2330.TW",
            "composite_score": "not_a_number",
            "models": {
                "lstm_potential": "N/A",
                "timesfm_potential": None,
                "risk_reward_ratio": "invalid"
            }
        }
        verdict = client._evaluate_local_rules(malformed_state)
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.ticker, "2330.TW")
        self.assertEqual(verdict.source_slot, "local_rules")
        self.assertEqual(verdict.action_decision, "hold_watch")

    def test_format_stock_state_duckdb_flat_record(self):
        """驗證從 DuckDB 讀出的頂層扁平字典（含字串 tags、頂層 pe/pb/potential/score）能被完整轉換成決策特徵"""
        duckdb_record = {
            "ticker": "2317.TW",
            "current_price": 205.0,
            "score": 82.5,
            "potential": 4.8,
            "pe": 16.2,
            "pb": 2.1,
            "forward_pe": 14.5,
            "ev_ebitda": 9.8,
            "ma60": 198.0,
            "pullback_type": "MA60 (+3.5%)",
            "foreign_net_5d": 12000.0,
            "trust_net_5d": 4500.0,
            "trust_streak": 5,
            "tags": "🏆三重共振,玄鐵買點,土洋合買,主流板塊"
        }
        state = ClefDecisionClient.format_stock_state(duckdb_record, macro_regime="bull")
        self.assertEqual(state["ticker"], "2317.TW")
        self.assertEqual(state["composite_score"], 82.5)
        self.assertEqual(state["valuation"]["pe"], 16.2)
        self.assertEqual(state["valuation"]["pb"], 2.1)
        self.assertEqual(state["models"]["lstm_potential"], 4.8)
        self.assertEqual(state["institutional"]["trust_streak"], 5)
        self.assertTrue(state["institutional"]["is_sync_buy"])
        self.assertIn("土洋合買", state["tags"])
        self.assertIn("🏆三重共振", state["tags"])

    def test_format_stock_state_pipe_separated_tags(self):
        """驗證 DuckDB 實際使用的 pipe (' | ') 分隔標籤格式能正確解析並觸發土洋合買與玄鐵訊號"""
        record_with_pipes = {
            "ticker": "2330.TW",
            "current_price": 1050.0,
            "potential": 4.5,
            "tags": "玄鐵買點 | 土洋合買 | 👑四重共振"
        }
        state = ClefDecisionClient.format_stock_state(record_with_pipes)
        self.assertEqual(len(state["tags"]), 3)
        self.assertIn("玄鐵買點", state["tags"])
        self.assertIn("土洋合買", state["tags"])
        self.assertIn("👑四重共振", state["tags"])
        self.assertTrue(state["institutional"]["is_sync_buy"])

    @patch("requests.post")
    def test_evaluate_stock_malformed_confidence_null_fallback(self, mock_post):
        """驗證上游回傳異常資料（如 confidence: null 或缺少必要欄位）時不會崩潰，而是觸發降級或 fallback"""
        # 第一個節點回傳 HTTP 200 但 action_decision.confidence 為 null
        malformed_resp = MagicMock()
        malformed_resp.status_code = 200
        malformed_resp.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {
                    "choice": "buy",
                    "confidence": "INVALID_NOT_A_FLOAT"  # 格式異常
                }
            }
        }

        # 第二個節點回傳正常答案
        valid_resp = MagicMock()
        valid_resp.status_code = 200
        valid_resp.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "buy", "confidence": 0.8},
                "conviction_score": {"score": 4.0},
                "is_favorable_entry": {"noul": 0.7}
            }
        }

        mock_post.side_effect = [malformed_resp, valid_resp]

        cand = {"ticker": "2330.TW", "current_price": 1000.0}
        verdict = self.client.evaluate_stock(cand)

        # 應成功跳過第一個節點的異常答案，由第二節點完成推論
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.source_slot, "Clef-Aiurl")
        self.assertEqual(verdict.action_decision, "buy")

    @patch("requests.post")
    def test_evaluate_stock_conviction_score_string_fallback(self, mock_post):
        """驗證當主節點回傳 conviction_score 為非 dict 字串時，slot 驗證失敗並繼續嘗試備援節點"""
        resp_invalid_sc = MagicMock()
        resp_invalid_sc.status_code = 200
        resp_invalid_sc.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "buy", "confidence": 0.85},
                "conviction_score": "invalid_string_not_dict",
                "is_favorable_entry": {"noul": 0.65}
            }
        }

        resp_valid = MagicMock()
        resp_valid.status_code = 200
        resp_valid.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "strong_buy", "confidence": 0.9},
                "conviction_score": {"score": 4.5},
                "is_favorable_entry": {"noul": 0.8}
            }
        }

        mock_post.side_effect = [resp_invalid_sc, resp_valid]

        cand = {"ticker": "2330.TW", "current_price": 1000.0}
        verdict = self.client.evaluate_stock(cand)

        self.assertTrue(verdict.success)
        self.assertEqual(verdict.source_slot, "Clef-Aiurl")
        self.assertEqual(verdict.action_decision, "strong_buy")
        self.assertEqual(verdict.conviction_score, 4.5)

    def test_format_stock_state_timesfm_only_does_not_pollute_lstm(self):
        """驗證當候選只有 TimesFM 預測時，不會把 TimesFM 潛力誤拷貝給 LSTM 造成虛假雙模型共振"""
        cand = {
            "ticker": "2330.TW",
            "current_price": 1000.0,
            "timesfm_potential": 4.0,
            "potential": 4.0,
            "lstm_potential": None
        }
        state = ClefDecisionClient.format_stock_state(cand)
        self.assertIsNone(state["models"]["lstm_potential"])
        self.assertEqual(state["models"]["timesfm_potential"], 4.0)

    @patch("requests.post")
    def test_evaluate_stock_non_finite_infinity_triggers_fallback(self, mock_post):
        """驗證上游回傳 Infinity 或非有限數值時，slot 判定無效並切換備援節點，避免 JSONResponse 序列化崩潰"""
        resp_inf = MagicMock()
        resp_inf.status_code = 200
        resp_inf.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "buy", "confidence": 0.8},
                "conviction_score": {"score": "Infinity"},  # 非有限數值
                "is_favorable_entry": {"noul": 0.5}
            }
        }

        resp_valid = MagicMock()
        resp_valid.status_code = 200
        resp_valid.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "buy", "confidence": 0.8},
                "conviction_score": {"score": 4.0},
                "is_favorable_entry": {"noul": 0.6}
            }
        }

        mock_post.side_effect = [resp_inf, resp_valid]

        cand = {"ticker": "2330.TW", "current_price": 1000.0}
        verdict = self.client.evaluate_stock(cand)

        # 應跳過包含 Infinity 的節點，由備援節點成功推論
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.source_slot, "Clef-Aiurl")
        self.assertEqual(verdict.conviction_score, 4.0)

        # 驗證 to_dict 可安全通過 JSON 序列化 (無 inf)
        from starlette.responses import JSONResponse
        res = JSONResponse(verdict.to_dict())
        self.assertEqual(res.status_code, 200)

    @patch("requests.post")
    def test_evaluate_stock_unsupported_choice_triggers_fallback(self, mock_post):
        """驗證當上游回傳非契約動作 (如 'sell') 或超出契約範圍數值時，slot 拒絕並切換備援節點"""
        resp_unsupported = MagicMock()
        resp_unsupported.status_code = 200
        resp_unsupported.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "sell", "confidence": 0.8},  # 'sell' 不在支援動作中
                "conviction_score": {"score": 4.0},
                "is_favorable_entry": {"noul": 0.5}
            }
        }

        resp_valid = MagicMock()
        resp_valid.status_code = 200
        resp_valid.json.return_value = {
            "model": "clef-flash",
            "answers": {
                "action_decision": {"choice": "avoid", "confidence": 0.85},
                "conviction_score": {"score": 4.5},
                "is_favorable_entry": {"noul": 0.2}
            }
        }

        mock_post.side_effect = [resp_unsupported, resp_valid]

        cand = {"ticker": "1303.TW", "current_price": 50.0}
        verdict = self.client.evaluate_stock(cand)

        self.assertTrue(verdict.success)
        self.assertEqual(verdict.source_slot, "Clef-Aiurl")
        self.assertEqual(verdict.action_decision, "avoid")


if __name__ == "__main__":
    unittest.main()
