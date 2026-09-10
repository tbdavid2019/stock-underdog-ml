"""
evaluators/formatter.py - 終端機與推播訊息排版格式化工具
包含宏觀風控卡片、AI 操盤解讀、三重共振交集與分項策略表格。
"""

import logging
from typing import Any, Optional
import pandas as pd
from evaluators.composite_evaluator import EvaluationReport

default_logger = logging.getLogger("stock_app.report")


def format_value(val: Any, decimal: int = 2) -> str:
    """Format numeric values safely, handling None, NaN, and Series"""
    if val is None:
        return "N/A"
    if isinstance(val, pd.Series):
        if val.empty:
            return "N/A"
        val = val.iloc[0]
    if isinstance(val, float) and pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.{decimal}f}"
    return str(val)


def print_evaluation_report(report: EvaluationReport, log=None):
    """
    輸出美化之終端機與日誌量化報告
    """
    logger = log or default_logger
    index_name = report.index_name
    xuantie_df = report.xuantie_results
    lstm_results = report.lstm_results
    timesfm_results = getattr(report, "timesfm_results", [])
    overlap_df = report.overlap_results
    macro = report.macro_state

    logger.info(f"\n{'='*100}")
    logger.info(f"📊 投資建議報告 - {index_name}")
    logger.info(f"{'='*100}\n")

    # ====== 頂層 1: 大盤與宏觀風控 ======
    if macro:
        is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")
        if is_tw:
            logger.info("🇹🇼 【台股大盤與籌碼風向】")
            logger.info(f"   • 大盤狀態: {macro.regime_name} (建議曝險: {int(macro.exposure*100)}%)")
            twii_str = '站穩MA60' if getattr(macro, 'twii_above_ma60', True) else '跌破MA60'
            sox_str = '費半站穩季線' if macro.sox_above_ma60 else '費半破季線'
            logger.info(f"   • 關鍵指標: 加權指數={twii_str} | 國際連動={sox_str} (VIX={macro.vix:.1f})")
        else:
            logger.info("🇺🇸 【美股宏觀環境與風控門檻】")
            logger.info(f"   • 市場狀態: {macro.regime_name} (建議曝險: {int(macro.exposure*100)}%)")
            logger.info(f"   • 關鍵指標: VIX={macro.vix:.1f} | SPY={'站穩MA60' if macro.spy_above_ma60 else '破季線'} | SOX={'站穩MA60' if macro.sox_above_ma60 else '破季線'}")
        if macro.warnings:
            for w in macro.warnings:
                logger.info(f"   • ⚠️  {w}")
        logger.info("")

    # ====== 頂層 2: AI 量化操盤解讀 ======
    if report.ai_summary:
        logger.info("🧠 【AI 量化操盤解讀】")
        for line in report.ai_summary.split("\n"):
            logger.info(f"   {line}")
        logger.info("")

    # ====== 多策略交集 / 三重共振 ======
    logger.info("⭐ 【優先推薦】三重共振 / 多策略交集 (技術面 + ML + 籌碼 + 估值)")
    logger.info(f"   重點交集符合: {len(overlap_df)} 支\n")

    if not overlap_df.empty:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'LSTM':>8} {'TimesFM':>8} {'盈虧比':>7} {'回調':>6} {'MA60':>8} {'PE':>6} {'PB':>6} {'綜合標籤'}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*30}")
        for idx, row in overlap_df.iterrows():
            cand = next((c for c in report.overlap_candidates if c["ticker"] == row["ticker"]), None)
            tags_str = " | ".join(cand["tags"]) if cand and cand.get("tags") else "觀察"

            lstm_str = f"{row['lstm_potential']:>+7.2f}%" if pd.notna(row.get('lstm_potential')) and row.get('lstm_potential') is not None else "    N/A"
            tfm_str = f"{row['timesfm_potential']:>+7.2f}%" if pd.notna(row.get('timesfm_potential')) and row.get('timesfm_potential') is not None else "    N/A"
            rr_val = row.get('risk_reward_ratio')
            rr_str = f"{rr_val:>6.2f}x" if pd.notna(rr_val) and rr_val is not None else "   N/A"

            logger.info(
                f"   {idx+1:<4} {row['ticker']:<10} "
                f"{lstm_str} "
                f"{tfm_str} "
                f"{rr_str} "
                f"{str(row['pullback_type'])[:6]:>6} "
                f"{format_value(row.get('ma60')):>8} "
                f"{format_value(row.get('pe')):>6} "
                f"{format_value(row.get('pb')):>6} "
                f"{tags_str}"
            )
    else:
        logger.info("   (本期無多策略共振股票)")

    logger.info("")

    # ====== 軌道 1: 玄鐵重劍 ======
    logger.info("🗡️  【波段操作】玄鐵重劍策略 (持有 2-4 週) - 技術面買點")
    logger.info(f"   符合條件: {len(xuantie_df)} 支\n")

    if not xuantie_df.empty:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'價格':>8} {'MA60':>8} {'回調類型':<18} {'PE':>6} {'PB':>6}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*8} {'-'*8} {'-'*18} {'-'*6} {'-'*6}")
        for idx, row in xuantie_df.head(10).iterrows():
            logger.info(
                f"   {idx+1:<4} {row['ticker']:<10} "
                f"{row['current_price']:>8.2f} "
                f"{format_value(row.get('ma60')):>8} "
                f"{row['pullback_type']:<18} "
                f"{format_value(row.get('pe')):>6} "
                f"{format_value(row.get('pb')):>6}"
            )
    else:
        logger.info("   (本期無符合條件的股票)")

    logger.info("")

    # ====== 軌道 2: LSTM 預測 ======
    logger.info("🤖 【短線操作】LSTM 預測 (持有 1-7 天) - 預測漲幅排行")
    logger.info(f"   預測完成: {len(lstm_results)} 支\n")

    if lstm_results:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'預測漲幅':>10} {'現價':>8} {'→':^3} {'預測價':>8} {'PE':>6} {'PB':>6}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*3} {'-'*8} {'-'*6} {'-'*6}")
        for i, result in enumerate(lstm_results[:10], 1):
            logger.info(
                f"   {i:<4} {result['ticker']:<10} "
                f"{result['potential']:>+9.2f}% "
                f"{result['current_price']:>8.2f} {'→':^3} "
                f"{result['predicted_price']:>8.2f} "
                f"{format_value(result.get('pe')):>6} "
                f"{format_value(result.get('pb')):>6}"
            )
    else:
        logger.info("   (本期無 LSTM 預測結果)")

    logger.info("")

    # ====== 軌道 3: TimesFM 預測 ======
    logger.info("🔮 【時序大模型】TimesFM 預測 (持有 1-5 天) - 預測漲幅排行與盈虧比")
    logger.info(f"   預測完成: {len(timesfm_results)} 支\n")

    if timesfm_results:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'預測漲幅':>10} {'現價':>8} {'→':^3} {'5日目標':>8} {'盈虧比':>8} {'PE':>6} {'PB':>6}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*3} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
        for i, result in enumerate(timesfm_results[:10], 1):
            h_price = result.get('horizon_predicted_price') or result.get('predicted_price') or result.get('current_price', 0.0)
            rr_val = result.get('risk_reward_ratio')
            rr_str = f"{rr_val:>7.2f}x" if pd.notna(rr_val) and rr_val is not None else "     N/A"
            logger.info(
                f"   {i:<4} {result['ticker']:<10} "
                f"{result['potential']:>+9.2f}% "
                f"{result['current_price']:>8.2f} {'→':^3} "
                f"{h_price:>8.2f} "
                f"{rr_str:>8} "
                f"{format_value(result.get('pe')):>6} "
                f"{format_value(result.get('pb')):>6}"
            )
    else:
        logger.info("   (本期無 TimesFM 預測結果)")

    logger.info(f"\n{'='*100}\n")
