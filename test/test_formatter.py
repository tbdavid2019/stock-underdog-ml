"""
test/test_formatter.py - 測試 formatter 排版與格式化引擎
驗證 Telegram 高級卡片排版、Discord Markdown、Email 與 Terminal 輸出
"""

import unittest
import pandas as pd
from evaluators.formatter import (
    format_value,
    format_ticker_label,
    format_telegram_message,
    format_discord_message,
    format_email_message,
    format_dual_strategy_message,
    print_evaluation_report,
)
from evaluators.composite_evaluator import EvaluationReport
from data.macro import MacroState


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.name_map = {
            "2330.TW": "台積電",
            "5880.TW": "合庫金",
            "2603.TW": "長榮",
            "3045.TW": "台灣大"
        }
        self.macro = MacroState(
            regime_name="全面多頭 (Bullish)",
            exposure=1.0,
            vix=14.5,
            twii_above_ma60=True,
            sox_above_ma60=True
        )

        self.candidates = [
            {
                "ticker": "2330.TW",
                "current_price": 1005.0,
                "composite_score": 92.5,
                "resonance_tier": "👑四重共振",
                "tags": ["👑四重共振", "玄鐵買點", "LSTM看漲", "TimesFM看漲", "高盈虧比", "土洋合買", "主流板塊(半導體)", "低PE"],
                "lstm_potential": 5.2,
                "timesfm_potential": 4.1,
                "risk_reward_ratio": 2.5,
                "pullback_type": "MA60 (+2.0%)",
                "fundamentals": {"pe": 22.0, "pb": 5.0},
                "institutional": {"is_sync_buy": True, "trust_streak": 4}
            },
            {
                "ticker": "5880.TW",
                "current_price": 26.8,
                "composite_score": 84.0,
                "resonance_tier": "🏆三重共振",
                "tags": ["🏆三重共振", "玄鐵買點", "LSTM看漲", "投信連買3天", "低PE", "低PB"],
                "lstm_potential": 4.1,
                "timesfm_potential": 0.7,
                "risk_reward_ratio": 1.2,
                "pullback_type": "MA60 (+4.1%)",
                "fundamentals": {"pe": 19.5, "pb": 1.5},
                "institutional": {"is_sync_buy": False, "trust_streak": 3}
            },
            {
                "ticker": "3045.TW",
                "current_price": 112.5,
                "composite_score": 78.0,
                "resonance_tier": "🔮雙ML共振",
                "tags": ["🔮雙ML共振", "LSTM看漲", "TimesFM看漲", "投信買超"],
                "lstm_potential": 2.9,
                "timesfm_potential": 1.8,
                "risk_reward_ratio": 1.5,
                "pullback_type": "",
                "fundamentals": {"pe": 23.5, "pb": 4.4},
                "institutional": {"trust_net_5d": 500}
            }
        ]

        self.xuantie_df = pd.DataFrame([
            {"ticker": "2330.TW", "current_price": 1005.0, "pullback_type": "MA60 (+2.0%)", "pe": 22.0, "pb": 5.0},
            {"ticker": "2603.TW", "current_price": 195.0, "pullback_type": "MA60 (+3.4%)", "pe": 9.4, "pb": 0.9}
        ])

        self.lstm_results = [
            {"ticker": "2330.TW", "potential": 5.2, "current_price": 1005.0, "predicted_price": 1057.26},
            {"ticker": "5880.TW", "potential": 4.1, "current_price": 26.8, "predicted_price": 27.90}
        ]

        self.timesfm_results = [
            {"ticker": "2330.TW", "potential": 4.1, "horizon_predicted_price": 1046.2, "risk_reward_ratio": 2.5},
            {"ticker": "3045.TW", "potential": 1.8, "horizon_predicted_price": 114.5, "risk_reward_ratio": 1.5}
        ]

        self.report_dict = {
            "overlap_candidates": self.candidates,
            "xuantie_results": self.xuantie_df,
            "lstm_results": self.lstm_results,
            "timesfm_results": self.timesfm_results,
            "macro_state": self.macro,
            "ai_summary": "今日盤勢處於全面多頭，優先聚焦 2330.TW 與 5880.TW。"
        }

    def test_format_value(self):
        self.assertEqual(format_value(None), "N/A")
        self.assertEqual(format_value(float("nan")), "N/A")
        self.assertEqual(format_value(12.3456, decimal=2), "12.35")
        self.assertEqual(format_value("hello"), "hello")

    def test_format_ticker_label(self):
        self.assertEqual(format_ticker_label("2330.TW", self.name_map), "2330.TW 台積電")
        self.assertEqual(format_ticker_label("9999.TW", self.name_map), "9999.TW")

    def test_telegram_message_structure_and_no_pre_block_for_recommendations(self):
        """
        驗證 Telegram 訊息：
        1. 不再使用巨大 <pre> 包含優先推薦
        2. 採用清晰簡潔的層次標籤（【四重共振】、【三重共振】、【雙ML共振】）
        3. 去除雜亂無章的 emoji 堆疊，維持專業金融資訊閱讀體驗
        4. 包含 blockquote AI 解讀
        """
        msg = format_telegram_message(
            "台灣50",
            self.report_dict,
            "2026-10-08 09:00:00",
            name_map=self.name_map,
            macro_state=self.macro,
            ai_summary=self.report_dict["ai_summary"]
        )

        self.assertIn("<b>多維量化投資日報</b>", msg)
        self.assertIn("<b>台灣50</b>", msg)
        self.assertIn("<b>全面多頭 (Bullish)</b>", msg)
        self.assertIn("<blockquote>今日盤勢處於全面多頭", msg)

        # 優先推薦區塊
        self.assertIn("<b>【優先推薦 (多維共振)】</b>", msg)
        self.assertIn("<b>【四重共振】</b>", msg)
        self.assertIn("<b>【三重共振】</b>", msg)
        self.assertIn("<b>【雙ML共振】</b>", msg)

        # 驗證個股卡片內容
        self.assertIn("2330.TW 台積電", msg)
        self.assertIn("<code>1,005.00</code>", msg)
        self.assertIn("LSTM <b>+5.2%</b>", msg)
        self.assertIn("TFM <b>+4.1%</b> (2.5x)", msg)
        self.assertIn("土洋合買", msg)
        self.assertIn("PE:22.0 · PB:5.0", msg)

        # 驗證沒有使用 <pre> 包裹優先推薦
        cand_section = msg[msg.find("優先推薦"):msg.find("波段操作")]
        self.assertNotIn("<pre>", cand_section, "優先推薦不應使用 <pre> 包裹以免手機版中英破版")

    def test_discord_message_formatting(self):
        """驗證 Discord 訊息採用 Markdown 語法"""
        msg = format_discord_message(
            "台灣50",
            self.report_dict,
            "2026-10-08 09:00:00",
            name_map=self.name_map,
            macro_state=self.macro,
            ai_summary=self.report_dict["ai_summary"]
        )

        self.assertIn("**多維量化投資日報**", msg)
        self.assertIn("**【四重共振】**", msg)
        self.assertIn("**【三重共振】**", msg)
        self.assertIn("• **2330.TW 台積電** `1,005.00`", msg)
        self.assertIn("> 今日盤勢處於全面多頭", msg)

    def test_email_message_formatting(self):
        """驗證 Email 純文字訊息"""
        msg = format_email_message(
            "台灣50",
            self.report_dict,
            "2026-10-08 09:00:00",
            name_map=self.name_map,
            macro_state=self.macro,
            ai_summary=self.report_dict["ai_summary"]
        )
        self.assertIn("多維量化策略投資日報", msg)
        self.assertIn("2330.TW 台積電", msg)
        self.assertIn("[👑四重共振]", msg)

    def test_format_dual_strategy_message_dict(self):
        """驗證整合輸出字典包含 telegram, discord, email 三種通道"""
        messages = format_dual_strategy_message(
            "台灣50",
            self.report_dict,
            "2026-10-08 09:00:00",
            name_map=self.name_map,
            macro_state=self.macro,
            ai_summary=self.report_dict["ai_summary"]
        )
        self.assertIn("telegram", messages)
        self.assertIn("discord", messages)
        self.assertIn("email", messages)
        self.assertIsInstance(messages["telegram"], str)

    def test_empty_candidates_graceful_handling(self):
        """驗證當無任何推薦股票時的防守提示"""
        empty_results = {
            "overlap_candidates": [],
            "xuantie_results": pd.DataFrame(),
            "lstm_results": [],
            "timesfm_results": [],
            "macro_state": self.macro
        }
        msg = format_telegram_message("台灣50", empty_results, "2026-10-08 09:00:00")
        self.assertIn("本期無符合多維正向共振條件之標的", msg)

    def test_telegram_html_escaping(self):
        """驗證動態文字與 AI Summary 中的 HTML 保留字元 (<, >, &) 會被自動轉義，避免 Telegram 解析失敗"""
        dirty_summary = "市場處於 <震盪整理> 階段 & 潛在風險高，若 P < 10 則考慮買入。"
        dirty_regime = "警戒 <High Risk> & Defensive"
        dirty_macro = MacroState(
            regime_name=dirty_regime,
            exposure=0.6,
            vix=21.5,
            twii_above_ma60=True,
            sox_above_ma60=False
        )
        msg = format_telegram_message(
            "指數 <Alpha & Beta>",
            self.report_dict,
            "2026-10-08 09:00:00",
            name_map={"2330.TW": "台積電 <TSMC & Foundry>"},
            macro_state=dirty_macro,
            ai_summary=dirty_summary
        )
        # 確保 raw < > 沒有外洩在標籤外造成 Telegram HTML parse error
        self.assertNotIn("<震盪整理>", msg)
        self.assertIn("&lt;震盪整理&gt;", msg)
        self.assertIn("指數 &lt;Alpha &amp; Beta&gt;", msg)
        self.assertIn("警戒 &lt;High Risk&gt; &amp; Defensive", msg)
        self.assertIn("台積電 &lt;TSMC &amp; Foundry&gt;", msg)

    def test_split_telegram_message(self):
        """驗證長訊息自動切分功能，確保不超過 4000 字元限制且保持段落完整性"""
        from evaluators.formatter import split_telegram_message

        # 短訊息不切分
        short_msg = "Hello World"
        self.assertEqual(split_telegram_message(short_msg, max_length=100), ["Hello World"])

        # 超長訊息依段落切分
        para1 = "A" * 60
        para2 = "B" * 60
        para3 = "C" * 60
        long_msg = f"{para1}\n\n{para2}\n\n{para3}"

        chunks = split_telegram_message(long_msg, max_length=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_split_telegram_message_html_tag_rebalancing(self):
        """驗證當訊息跨區塊切分時，HTML 標籤（如 blockquote, b）在每個 chunk 內均保持正確閉合與重新開啟"""
        from evaluators.formatter import split_telegram_message

        html_msg = (
            "<b>Header Title</b>\n\n"
            "<blockquote>" + ("Long quote sentence that needs to be cleanly split across chunks. " * 5) + "</blockquote>"
        )
        chunks = split_telegram_message(html_msg, max_length=150)
        self.assertGreater(len(chunks), 1)

        # 驗證每個 chunk 的 <blockquote> 與 </b> 標籤皆成對平衡，不產生懸空標籤
        for idx, chunk in enumerate(chunks):
            open_bq = chunk.count("<blockquote>")
            close_bq = chunk.count("</blockquote>")
            self.assertEqual(open_bq, close_bq, f"Chunk {idx} 內 <blockquote> 與 </blockquote> 數量不平衡: {chunk}")
            open_b = chunk.count("<b>")
            close_b = chunk.count("</b>")
            self.assertEqual(open_b, close_b, f"Chunk {idx} 內 <b> 與 </b> 數量不平衡: {chunk}")


if __name__ == "__main__":
    unittest.main()
