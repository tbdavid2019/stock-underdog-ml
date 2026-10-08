"""
evaluators/formatter.py - 終端機與多管道推播訊息排版格式化工具
包含宏觀風控卡片、AI 操盤解讀、多維共振 (四重/三重/雙ML) 優先推薦與分項策略卡片。
全面支援 Telegram (HTML 高級卡片)、Discord (Markdown 卡片)、Email (結構化純文字) 與 Terminal Log。
"""

import html
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
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


def format_ticker_label(ticker: str, name_map: Optional[Dict[str, str]] = None) -> str:
    """Format ticker with stock name from map if available."""
    lookup = name_map or {}
    name = lookup.get(ticker, "")
    return f"{ticker} {name}" if name else ticker


def _extract_candidates(results: dict, overlap_df: pd.DataFrame) -> List[dict]:
    """Extract list of candidate dictionaries from results or overlap_df."""
    candidates = results.get("overlap_candidates")
    if candidates and isinstance(candidates, list) and len(candidates) > 0:
        return candidates

    cand_list = []
    if isinstance(overlap_df, pd.DataFrame) and not overlap_df.empty:
        for _, row in overlap_df.iterrows():
            d = row.to_dict()
            cand_list.append({
                k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                for k, v in d.items()
            })
    elif isinstance(overlap_df, list):
        cand_list = overlap_df
    return cand_list


# ==============================================================================
# Telegram 訊息排版 (HTML 高級結構化卡片)
# ==============================================================================

def _format_telegram_candidate_card(cand: dict, lookup: Dict[str, str]) -> str:
    """單一候選標的高級卡片排版 (適合手機閱讀，簡潔專業無雜亂表情符號)"""
    ticker = cand.get("ticker", "")
    ticker_label = html.escape(format_ticker_label(ticker, lookup))
    curr_price = cand.get("current_price") or 0.0
    price_str = f"{curr_price:,.2f}" if curr_price > 0 else "N/A"

    lines = [f"• <b>{ticker_label}</b> <code>{price_str}</code>"]

    details = []

    # 1. ML 模型預測
    lstm_pot = cand.get("lstm_potential")
    if lstm_pot is not None and pd.notna(lstm_pot):
        pot_val = float(lstm_pot)
        details.append(f"LSTM <b>{pot_val:+.1f}%</b>")

    tfm_pot = cand.get("timesfm_potential")
    if tfm_pot is not None and pd.notna(tfm_pot):
        pot_val = float(tfm_pot)
        rr = cand.get("risk_reward_ratio")
        rr_str = f" ({float(rr):.1f}x)" if rr and pd.notna(rr) and float(rr) > 0 else ""
        details.append(f"TFM <b>{pot_val:+.1f}%</b>{rr_str}")

    # 2. 均線技術買點
    pb_type = cand.get("pullback_type")
    if pb_type:
        details.append(str(pb_type).strip())

    # 3. 籌碼鎖碼訊號
    tags = cand.get("tags") or []
    inst = cand.get("institutional") or {}
    if inst.get("is_sync_buy") or "土洋合買" in tags:
        details.append("土洋合買")

    streak_tag = next((t for t in tags if "投信連買" in t), None)
    if streak_tag:
        details.append(streak_tag)
    elif inst.get("trust_streak", 0) >= 3:
        details.append(f"投信連買{inst.get('trust_streak')}天")
    elif inst.get("trust_net_5d", 0) > 0 or "投信買超" in tags:
        details.append("投信買超")

    for t in tags:
        if "主流板塊" in t:
            details.append(t)
        elif t in ("高盈虧比", "低PE", "低PB"):
            details.append(t)

    unique_details = list(dict.fromkeys(details))

    # 4. 估值 (PE / PB)
    val_parts = []
    pe_val = cand.get("pe") or (cand.get("fundamentals") or {}).get("pe")
    pb_val = cand.get("pb") or (cand.get("fundamentals") or {}).get("pb")
    if pe_val and pd.notna(pe_val) and float(pe_val) > 0:
        val_parts.append(f"PE:{float(pe_val):.1f}")
    if pb_val and pd.notna(pb_val) and float(pb_val) > 0:
        val_parts.append(f"PB:{float(pb_val):.1f}")

    detail_str = " · ".join(unique_details[:4])
    val_str = " · ".join(val_parts)

    if detail_str and val_str:
        lines.append(f"  {detail_str} | {val_str}")
    elif detail_str:
        lines.append(f"  {detail_str}")
    elif val_str:
        lines.append(f"  {val_str}")

    return "\n".join(lines)


def format_telegram_message(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
) -> str:
    """格式化量化結果為適合手機 Telegram 閱讀的高級 HTML 訊息 (去雜亂表情符號，強化層次)"""
    lookup = name_map or {}
    xuantie_df = results.get("xuantie_results", pd.DataFrame())
    lstm_results = results.get("lstm_results", [])
    timesfm_results = results.get("timesfm_results", [])
    overlap_df = results.get("overlap_results", pd.DataFrame())
    macro = macro_state or results.get("macro_state")
    summary_text = ai_summary or results.get("ai_summary", "")

    candidates = _extract_candidates(results, overlap_df)

    msg = "<b>多維量化投資日報</b>\n"
    msg += f"⏰ {calculation_time} | 指數: <b>{html.escape(index_name)}</b>\n\n"

    is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")

    # 宏觀風控卡片
    if macro:
        if is_tw:
            twii_str = '站穩MA60' if getattr(macro, 'twii_above_ma60', True) else '跌破MA60'
            sox_str = '費半站穩季線' if getattr(macro, 'sox_above_ma60', True) else '費半破季線'
            vix_val = getattr(macro, 'vix', 0.0)
            msg += "<b>【台股大盤與風控】</b>\n"
            msg += f"• 大盤狀態: <b>{html.escape(str(macro.regime_name))}</b> (建議曝險 {int(macro.exposure*100)}%)\n"
            msg += f"• 加權指數: {twii_str} | 國際連動: {sox_str} (VIX {vix_val:.1f})\n\n"
        else:
            spy_str = '站穩MA60' if getattr(macro, 'spy_above_ma60', True) else '破季線'
            sox_str = '站穩MA60' if getattr(macro, 'sox_above_ma60', True) else '破季線'
            vix_val = getattr(macro, 'vix', 0.0)
            msg += "<b>【美股宏觀風控】</b>\n"
            msg += f"• 狀態: <b>{html.escape(str(macro.regime_name))}</b> (建議曝險 {int(macro.exposure*100)}%)\n"
            msg += f"• VIX: {vix_val:.1f} | SPY: {spy_str} | SOX: {sox_str}\n\n"

    # AI 操盤總評 (採用 Telegram Blockquote 美化引言，嚴格跳脫 HTML 避免訊息解析失敗)
    if summary_text:
        clean_summary = html.escape(summary_text.strip())
        msg += f"<b>【AI 操盤解讀】</b>\n<blockquote>{clean_summary}</blockquote>\n\n"

    # 優先推薦 (多維共振)
    msg += "<b>【優先推薦 (多維共振)】</b>\n"
    if candidates:
        msg += f"符合共振條件: <b>{len(candidates)}</b> 支\n\n"

        tier_display_map = {
            "👑四重共振": "【四重共振】",
            "🏆三重共振": "【三重共振】",
            "🔮雙ML共振": "【雙ML共振】",
            "🌟多維共振": "【多維共振】",
            "⭐雙重共振": "【雙重共振】",
            "四重共振": "【四重共振】",
            "三重共振": "【三重共振】",
            "雙ML共振": "【雙ML共振】",
            "多維共振": "【多維共振】",
            "雙重共振": "【雙重共振】"
        }
        tier_order = ["👑四重共振", "🏆三重共振", "🔮雙ML共振", "🌟多維共振", "⭐雙重共振", "四重共振", "三重共振", "雙ML共振", "多維共振", "雙重共振"]
        grouped: Dict[str, List[dict]] = {}
        for c in candidates:
            tier = c.get("resonance_tier")
            if not tier:
                tags = c.get("tags") or []
                tier = next((t for t in tier_order if t in tags), "多維共振")
            grouped.setdefault(tier, []).append(c)

        shown_count = 0
        max_candidates_display = 15
        for tier_key in tier_order:
            if tier_key in grouped and shown_count < max_candidates_display:
                label = tier_display_map.get(tier_key, f"【{tier_key}】")
                msg += f"<b>{label}</b>\n"
                for cand in grouped[tier_key]:
                    if shown_count >= max_candidates_display:
                        break
                    msg += _format_telegram_candidate_card(cand, lookup) + "\n"
                    shown_count += 1
                msg += "\n"
                del grouped[tier_key]

        for tier_key, c_list in grouped.items():
            if shown_count < max_candidates_display:
                label = tier_display_map.get(tier_key, f"【{tier_key}】")
                msg += f"<b>{label}</b>\n"
                for cand in c_list:
                    if shown_count >= max_candidates_display:
                        break
                    msg += _format_telegram_candidate_card(cand, lookup) + "\n"
                    shown_count += 1
                msg += "\n"

        if len(candidates) > shown_count:
            remaining = len(candidates) - shown_count
            msg += f"<i>(其餘 {remaining} 支共振標的請參見 Web 儀表板完整清單)</i>\n\n"
    else:
        msg += "<i>(本期無符合多維正向共振條件之標的，建議防守觀望)</i>\n\n"

    # 波段操作 (玄鐵重劍)
    msg += "<b>【波段操作 (玄鐵重劍)】</b>\n"
    if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
        msg += f"符合買點: <b>{len(xuantie_df)}</b> 支 (顯示前5名)\n"
        for idx, row in xuantie_df.head(5).iterrows():
            ticker_label = html.escape(format_ticker_label(row['ticker'], lookup))
            price_str = f"{row['current_price']:,.2f}"
            pe_val = row.get('pe')
            pe_str = f"PE:{pe_val:.1f}" if pd.notna(pe_val) and pe_val else "PE:N/A"
            pb_val = row.get('pb')
            pb_str = f"PB:{pb_val:.1f}" if pd.notna(pb_val) and pb_val else "PB:N/A"
            pb_type = html.escape(str(row.get('pullback_type', '')).strip())
            msg += f"{idx+1}. <b>{ticker_label}</b> <code>{price_str}</code> | {pb_type} | {pe_str} · {pb_str}\n"
        msg += "\n"
    else:
        msg += "<i>(本期無符合波段買點標的)</i>\n\n"

    # 短線操作 (LSTM)
    msg += "<b>【短線操作 (LSTM 預測 TOP 5)】</b>\n"
    if lstm_results:
        for idx, result in enumerate(lstm_results[:5], 1):
            ticker_label = html.escape(format_ticker_label(result['ticker'], lookup))
            pot = result['potential']
            curr_p = result['current_price']
            pred_p = result.get('predicted_price', curr_p)
            msg += f"{idx}. <b>{ticker_label}</b> <b>{pot:+.1f}%</b> (<code>{curr_p:,.1f}</code> → <code>{pred_p:,.1f}</code>)\n"
        msg += "\n"
    else:
        msg += "<i>(本期無 LSTM 預測結果)</i>\n\n"

    # 時序大模型 (TimesFM)
    if timesfm_results:
        msg += "<b>【時序大模型 (TimesFM 預測 TOP 5)】</b>\n"
        for idx, result in enumerate(timesfm_results[:5], 1):
            ticker_label = html.escape(format_ticker_label(result['ticker'], lookup))
            pot = result['potential']
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: <b>{rr:.1f}x</b>" if rr and pd.notna(rr) else ""
            h_price = result.get('horizon_predicted_price')
            h_str = f"5日目標 <code>{h_price:,.1f}</code>" if h_price and pd.notna(h_price) else ""
            info_str = f" ({h_str}{rr_str})" if (h_str or rr_str) else ""
            msg += f"{idx}. <b>{ticker_label}</b> <b>{pot:+.1f}%</b>{info_str}\n"
        msg += "\n"

    return msg


def _find_markup_safe_split_index(text: str, target_limit: int) -> int:
    """
    Finds the highest split index <= target_limit that does NOT cut inside:
    1. An HTML tag like `<b...>`
    2. An HTML entity like `&amp;`
    Prefers double newline, single newline, space, or safe char offset.
    """
    if len(text) <= target_limit:
        return len(text)

    window = text[:target_limit]

    # Try split at paragraph boundary (\n\n)
    last_para = window.rfind("\n\n")
    if last_para > target_limit // 3:
        split_idx = last_para + 2
    else:
        # Try split at newline (\n)
        last_nl = window.rfind("\n")
        if last_nl > target_limit // 3:
            split_idx = last_nl + 1
        else:
            # Try split at space
            last_space = window.rfind(" ")
            if last_space > target_limit // 3:
                split_idx = last_space + 1
            else:
                split_idx = target_limit

    # Ensure split_idx does not cut an HTML tag <...>
    last_open = text.rfind("<", 0, split_idx)
    last_close = text.rfind(">", 0, split_idx)
    if last_open > last_close:
        split_idx = last_open
        if split_idx == 0:
            next_close = text.find(">", target_limit)
            split_idx = next_close + 1 if next_close != -1 else target_limit

    # Ensure split_idx does not cut an HTML entity &...;
    last_amp = text.rfind("&", 0, split_idx)
    last_semi = text.rfind(";", 0, split_idx)
    if last_amp > last_semi and (split_idx - last_amp) < 12:
        split_idx = last_amp

    return max(1, split_idx)


def split_telegram_message(message: str, max_length: int = 4000) -> List[str]:
    """
    Split an HTML formatted Telegram message into chunks <= max_length (Telegram limit is 4096).
    Splits at markup-safe boundaries and automatically rebalances opened/closed HTML tags
    so that every chunk is guaranteed to be syntactically valid HTML for Telegram.
    """
    if not message:
        return []
    if len(message) <= max_length:
        return [message]

    # Buffer for potential reopening and closing tags added during rebalancing
    target_limit = max(100, max_length - 120)

    raw_chunks: List[str] = []
    remaining = message

    while remaining:
        if len(remaining) <= target_limit:
            raw_chunks.append(remaining)
            break

        split_idx = _find_markup_safe_split_index(remaining, target_limit)
        chunk = remaining[:split_idx]
        raw_chunks.append(chunk)
        remaining = remaining[split_idx:].lstrip("\r\n")

    # Rebalance HTML tags across chunks
    tag_regex = re.compile(r'<\s*(/)?\s*([a-zA-Z0-9]+)(?:\s+[^>]*)?>')
    balanced_chunks = []
    active_stack: List[str] = []

    for raw in raw_chunks:
        prefix = "".join(f"<{t}>" for t in active_stack)
        # Scan raw chunk to see which tags are opened / closed
        for m in tag_regex.finditer(raw):
            is_closing = bool(m.group(1))
            tag_name = m.group(2).lower()
            if is_closing:
                if active_stack and active_stack[-1] == tag_name:
                    active_stack.pop()
                elif tag_name in active_stack:
                    for idx in range(len(active_stack) - 1, -1, -1):
                        if active_stack[idx] == tag_name:
                            active_stack.pop(idx)
                            break
            else:
                active_stack.append(tag_name)

        suffix = "".join(f"</{t}>" for t in reversed(active_stack))
        balanced = prefix + raw + suffix
        if balanced.strip():
            balanced_chunks.append(balanced)

    return balanced_chunks


# ==============================================================================
# Discord 訊息排版 (Markdown 卡片)
# ==============================================================================

def _format_discord_candidate_card(cand: dict, lookup: Dict[str, str]) -> str:
    """Discord Markdown 卡片 (簡潔專業無雜亂表情符號)"""
    ticker = cand.get("ticker", "")
    ticker_label = format_ticker_label(ticker, lookup)
    curr_price = cand.get("current_price") or 0.0
    price_str = f"{curr_price:,.2f}" if curr_price > 0 else "N/A"

    lines = [f"• **{ticker_label}** `{price_str}`"]

    details = []

    # 1. ML 模型預測
    lstm_pot = cand.get("lstm_potential")
    if lstm_pot is not None and pd.notna(lstm_pot):
        pot_val = float(lstm_pot)
        details.append(f"LSTM **{pot_val:+.1f}%**")

    tfm_pot = cand.get("timesfm_potential")
    if tfm_pot is not None and pd.notna(tfm_pot):
        pot_val = float(tfm_pot)
        rr = cand.get("risk_reward_ratio")
        rr_str = f" ({float(rr):.1f}x)" if rr and pd.notna(rr) and float(rr) > 0 else ""
        details.append(f"TFM **{pot_val:+.1f}%**{rr_str}")

    # 2. 均線技術買點
    pb_type = cand.get("pullback_type")
    if pb_type:
        details.append(str(pb_type).strip())

    # 3. 籌碼鎖碼訊號
    tags = cand.get("tags") or []
    inst = cand.get("institutional") or {}
    if inst.get("is_sync_buy") or "土洋合買" in tags:
        details.append("土洋合買")

    streak_tag = next((t for t in tags if "投信連買" in t), None)
    if streak_tag:
        details.append(streak_tag)
    elif inst.get("trust_streak", 0) >= 3:
        details.append(f"投信連買{inst.get('trust_streak')}天")
    elif inst.get("trust_net_5d", 0) > 0 or "投信買超" in tags:
        details.append("投信買超")

    for t in tags:
        if "主流板塊" in t:
            details.append(t)
        elif t in ("高盈虧比", "低PE", "低PB"):
            details.append(t)

    unique_details = list(dict.fromkeys(details))

    # 4. 估值 (PE / PB)
    val_parts = []
    pe_val = cand.get("pe") or (cand.get("fundamentals") or {}).get("pe")
    pb_val = cand.get("pb") or (cand.get("fundamentals") or {}).get("pb")
    if pe_val and pd.notna(pe_val) and float(pe_val) > 0:
        val_parts.append(f"PE:{float(pe_val):.1f}")
    if pb_val and pd.notna(pb_val) and float(pb_val) > 0:
        val_parts.append(f"PB:{float(pb_val):.1f}")

    detail_str = " · ".join(unique_details[:4])
    val_str = " · ".join(val_parts)

    if detail_str and val_str:
        lines.append(f"  {detail_str} | {val_str}")
    elif detail_str:
        lines.append(f"  {detail_str}")
    elif val_str:
        lines.append(f"  {val_str}")

    return "\n".join(lines)


def format_discord_message(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
) -> str:
    """格式化量化結果為 Discord Markdown 訊息"""
    lookup = name_map or {}
    xuantie_df = results.get("xuantie_results", pd.DataFrame())
    lstm_results = results.get("lstm_results", [])
    timesfm_results = results.get("timesfm_results", [])
    overlap_df = results.get("overlap_results", pd.DataFrame())
    macro = macro_state or results.get("macro_state")
    summary_text = ai_summary or results.get("ai_summary", "")

    candidates = _extract_candidates(results, overlap_df)

    msg = "**多維量化投資日報**\n"
    msg += f"⏰ {calculation_time} | 指數: **{index_name}**\n\n"

    is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")

    if macro:
        if is_tw:
            twii_str = '站穩MA60' if getattr(macro, 'twii_above_ma60', True) else '跌破MA60'
            sox_str = '費半站穩季線' if getattr(macro, 'sox_above_ma60', True) else '費半破季線'
            msg += f"**【台股大盤與風控】**: {macro.regime_name} (建議曝險 {int(macro.exposure*100)}%) | 加權: {twii_str} | 國際連動: {sox_str} (VIX {macro.vix:.1f})\n\n"
        else:
            spy_str = '站穩MA60' if getattr(macro, 'spy_above_ma60', True) else '破季線'
            sox_str = '站穩MA60' if getattr(macro, 'sox_above_ma60', True) else '破季線'
            msg += f"**【美股宏觀風控】**: {macro.regime_name} (建議曝險 {int(macro.exposure*100)}%) | VIX: {macro.vix:.1f} | SPY: {spy_str} | SOX: {sox_str}\n\n"

    if summary_text:
        quoted = summary_text.replace("\n", "\n> ")
        msg += f"**【AI 操盤解讀】**:\n> {quoted}\n\n"

    # 優先推薦 (多維共振)
    msg += "**【優先推薦 (多維共振)】**\n"
    if candidates:
        msg += f"符合共振條件: **{len(candidates)}** 支\n\n"
        tier_display_map = {
            "👑四重共振": "【四重共振】",
            "🏆三重共振": "【三重共振】",
            "🔮雙ML共振": "【雙ML共振】",
            "🌟多維共振": "【多維共振】",
            "⭐雙重共振": "【雙重共振】",
            "四重共振": "【四重共振】",
            "三重共振": "【三重共振】",
            "雙ML共振": "【雙ML共振】",
            "多維共振": "【多維共振】",
            "雙重共振": "【雙重共振】"
        }
        tier_order = ["👑四重共振", "🏆三重共振", "🔮雙ML共振", "🌟多維共振", "⭐雙重共振", "四重共振", "三重共振", "雙ML共振", "多維共振", "雙重共振"]
        grouped: Dict[str, List[dict]] = {}
        for c in candidates:
            tier = c.get("resonance_tier") or "多維共振"
            grouped.setdefault(tier, []).append(c)

        for tier_key in tier_order:
            if tier_key in grouped:
                label = tier_display_map.get(tier_key, f"【{tier_key}】")
                msg += f"**{label}**\n"
                for cand in grouped[tier_key]:
                    msg += _format_discord_candidate_card(cand, lookup) + "\n"
                msg += "\n"
                del grouped[tier_key]

        for tier_key, c_list in grouped.items():
            label = tier_display_map.get(tier_key, f"【{tier_key}】")
            msg += f"**{label}**\n"
            for cand in c_list:
                msg += _format_discord_candidate_card(cand, lookup) + "\n"
            msg += "\n"
    else:
        msg += "*(本期無符合多維正向共振條件之標的，建議防守觀望)*\n\n"

    # 波段操作
    msg += "**【波段操作 (玄鐵重劍)】**\n"
    if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
        msg += f"符合買點: **{len(xuantie_df)}** 支 (顯示前5名)\n"
        for idx, row in xuantie_df.head(5).iterrows():
            ticker_label = format_ticker_label(row['ticker'], lookup)
            price_str = f"{row['current_price']:,.2f}"
            pe_val = row.get('pe')
            pe_str = f"PE:{pe_val:.1f}" if pd.notna(pe_val) and pe_val else "PE:N/A"
            pb_val = row.get('pb')
            pb_str = f"PB:{pb_val:.1f}" if pd.notna(pb_val) and pb_val else "PB:N/A"
            pb_type = str(row.get('pullback_type', '')).strip()
            msg += f"{idx+1}. **{ticker_label}** `{price_str}` | {pb_type} | {pe_str} · {pb_str}\n"
        msg += "\n"
    else:
        msg += "*(本期無符合波段買點標的)*\n\n"

    # 短線操作
    msg += "**【短線操作 (LSTM 預測 TOP 5)】**\n"
    if lstm_results:
        for idx, result in enumerate(lstm_results[:5], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            pot = result['potential']
            curr_p = result['current_price']
            pred_p = result.get('predicted_price', curr_p)
            msg += f"{idx}. **{ticker_label}** **{pot:+.1f}%** (`{curr_p:,.1f}` → `{pred_p:,.1f}`)\n"
        msg += "\n"

    # TimesFM
    if timesfm_results:
        msg += "**【時序大模型 (TimesFM 預測 TOP 5)】**\n"
        for idx, result in enumerate(timesfm_results[:5], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            pot = result['potential']
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: **{rr:.1f}x**" if rr and pd.notna(rr) else ""
            h_price = result.get('horizon_predicted_price')
            h_str = f"5日目標 `{h_price:,.1f}`" if h_price and pd.notna(h_price) else ""
            info_str = f" ({h_str}{rr_str})" if (h_str or rr_str) else ""
            msg += f"{idx}. **{ticker_label}** **{pot:+.1f}%**{info_str}\n"
        msg += "\n"

    return msg


# ==============================================================================
# Email 訊息排版 (純文字結構化報告)
# ==============================================================================

def format_email_message(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
) -> str:
    """格式化量化結果為 Email 純文字格式"""
    lookup = name_map or {}
    xuantie_df = results.get("xuantie_results", pd.DataFrame())
    lstm_results = results.get("lstm_results", [])
    timesfm_results = results.get("timesfm_results", [])
    overlap_df = results.get("overlap_results", pd.DataFrame())
    macro = macro_state or results.get("macro_state")
    summary_text = ai_summary or results.get("ai_summary", "")

    candidates = _extract_candidates(results, overlap_df)

    msg = "多維量化策略投資日報\n"
    msg += f"運算時間: {calculation_time}\n"
    msg += f"指數: {index_name}\n\n"

    is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")

    if macro:
        if is_tw:
            twii_str = '站穩MA60' if getattr(macro, 'twii_above_ma60', True) else '跌破MA60'
            sox_str = '費半站穩季線' if getattr(macro, 'sox_above_ma60', True) else '費半破季線'
            msg += f"🇹🇼 台股大盤與風控: {macro.regime_name} (建議曝險 {int(macro.exposure*100)}%) | 加權: {twii_str} | 國際連動: {sox_str} (VIX {macro.vix:.1f})\n\n"
        else:
            spy_str = '站穩MA60' if getattr(macro, 'spy_above_ma60', True) else '破季線'
            sox_str = '站穩MA60' if getattr(macro, 'sox_above_ma60', True) else '破季線'
            msg += f"🇺🇸 美股宏觀風控: {macro.regime_name} (建議曝險 {int(macro.exposure*100)}%) | VIX: {macro.vix:.1f} | SPY: {spy_str} | SOX: {sox_str}\n\n"

    if summary_text:
        msg += f"AI 操盤解讀:\n{summary_text}\n\n"

    msg += "=" * 75 + "\n\n"

    # ⭐ 優先推薦
    msg += f"⭐ 優先推薦 (多維共振) - 符合條件: {len(candidates)} 支\n\n"
    if candidates:
        for idx, cand in enumerate(candidates, 1):
            ticker_label = format_ticker_label(cand.get("ticker", ""), lookup)
            price_val = cand.get("current_price", 0.0)
            tier_str = cand.get("resonance_tier", "多維共振")
            lstm_p = cand.get("lstm_potential")
            tfm_p = cand.get("timesfm_potential")
            rr = cand.get("risk_reward_ratio")
            ml_p_str = ""
            if lstm_p is not None:
                ml_p_str += f"LSTM: {float(lstm_p):+.1f}% "
            if tfm_p is not None:
                rr_info = f"({float(rr):.1f}x)" if rr and pd.notna(rr) else ""
                ml_p_str += f"TimesFM: {float(tfm_p):+.1f}%{rr_info}"

            tags_str = " · ".join(cand.get("tags", []))
            pe_val = cand.get("pe") or (cand.get("fundamentals") or {}).get("pe")
            pb_val = cand.get("pb") or (cand.get("fundamentals") or {}).get("pb")
            pe_str = f"PE:{float(pe_val):.1f}" if pe_val and pd.notna(pe_val) else ""
            pb_str = f"PB:{float(pb_val):.1f}" if pb_val and pd.notna(pb_val) else ""
            val_str = f" | {pe_str} {pb_str}".strip() if (pe_str or pb_str) else ""

            msg += f"{idx}. [{tier_str}] {ticker_label} 現價: {price_val:,.2f}\n"
            if ml_p_str:
                msg += f"   預測: {ml_p_str}\n"
            if tags_str:
                msg += f"   標籤: {tags_str}{val_str}\n"
            msg += "\n"
    else:
        msg += "(本期無符合多維正向共振條件之標的)\n\n"

    # 🗡️ 波段操作
    msg += f"🗡️  波段操作 (玄鐵重劍) - 符合條件: {len(xuantie_df)} 支 (顯示前10名)\n\n"
    if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
        for idx, row in xuantie_df.head(10).iterrows():
            ticker_label = format_ticker_label(row['ticker'], lookup)
            pe_val = row.get('pe')
            pe_str = f"PE:{pe_val:.1f}" if pd.notna(pe_val) and pe_val else "PE:N/A"
            pb_val = row.get('pb')
            pb_str = f"PB:{pb_val:.1f}" if pd.notna(pb_val) and pb_val else "PB:N/A"
            msg += f"{idx+1}. {ticker_label} 現價: {row['current_price']:,.2f} | 回調: {row.get('pullback_type', '')} | {pe_str} · {pb_str}\n"
        msg += "\n"

    # 🤖 LSTM
    msg += f"🤖 短線操作 (LSTM) - 預測完成: {len(lstm_results)} 支\n\n"
    if lstm_results:
        msg += "📈 預測上漲 TOP 10\n"
        for idx, result in enumerate(lstm_results[:10], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            curr_p = result['current_price']
            pred_p = result.get('predicted_price', curr_p)
            msg += f"{idx}. {ticker_label} 漲幅: {result['potential']:>+6.2f}% ({curr_p:,.1f} → {pred_p:,.1f})\n"
        msg += "\n"

    # 🔮 TimesFM
    if timesfm_results:
        msg += f"🔮 時序大模型 (TimesFM) - 預測完成: {len(timesfm_results)} 支\n\n"
        msg += "📈 TimesFM 看漲 TOP 10 (含 5 日目標價與盈虧比)\n"
        for idx, result in enumerate(timesfm_results[:10], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: {rr:.2f}x" if rr and pd.notna(rr) else ""
            h_price = result.get('horizon_predicted_price')
            h_str = f" | 5日目標: {h_price:,.2f}" if h_price and pd.notna(h_price) else ""
            msg += f"{idx}. {ticker_label} 漲幅: {result['potential']:>+6.2f}%{h_str}{rr_str}\n"
        msg += "\n"

    return msg


def format_dual_strategy_message(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
) -> dict:
    """提供 Telegram, Discord, Email 之多渠道整合字典輸出"""
    return {
        "telegram": format_telegram_message(
            index_name, results, calculation_time, name_map=name_map, macro_state=macro_state, ai_summary=ai_summary
        ),
        "discord": format_discord_message(
            index_name, results, calculation_time, name_map=name_map, macro_state=macro_state, ai_summary=ai_summary
        ),
        "email": format_email_message(
            index_name, results, calculation_time, name_map=name_map, macro_state=macro_state, ai_summary=ai_summary
        )
    }


# ==============================================================================
# 終端機與日誌輸出排版
# ==============================================================================

def print_evaluation_report(report: EvaluationReport, log=None):
    """輸出美化之終端機與日誌量化報告"""
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
            sox_str = '費半站穩季線' if getattr(macro, 'sox_above_ma60', True) else '費半破季線'
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

    # ====== 多策略交集 / 多維共振 ======
    logger.info("⭐ 【優先推薦】多維共振 (四重共振 / 三重共振 / 雙ML共振)")
    logger.info(f"   重點交集符合: {len(overlap_df)} 支\n")

    if not overlap_df.empty:
        logger.info(f"   {'排名':<4} {'等級':<8} {'代碼':<10} {'LSTM':>8} {'TimesFM':>8} {'盈虧比':>7} {'回調':>6} {'MA60':>8} {'PE':>6} {'PB':>6} {'綜合標籤'}")
        logger.info(f"   {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*30}")
        for idx, row in overlap_df.iterrows():
            cand = next((c for c in report.overlap_candidates if c["ticker"] == row["ticker"]), None)
            tier_str = (cand.get("resonance_tier") or row.get("resonance_tier") or "共振") if cand else (row.get("resonance_tier") or "共振")
            tags_str = " | ".join(cand["tags"]) if cand and cand.get("tags") else (" | ".join(row.get("tags", [])) if row.get("tags") else "觀察")

            lstm_str = f"{row['lstm_potential']:>+7.2f}%" if pd.notna(row.get('lstm_potential')) and row.get('lstm_potential') is not None else "    N/A"
            tfm_str = f"{row['timesfm_potential']:>+7.2f}%" if pd.notna(row.get('timesfm_potential')) and row.get('timesfm_potential') is not None else "    N/A"
            rr_val = row.get('risk_reward_ratio')
            rr_str = f"{rr_val:>6.2f}x" if pd.notna(rr_val) and rr_val is not None else "   N/A"

            logger.info(
                f"   {idx+1:<4} {tier_str:<8} {row['ticker']:<10} "
                f"{lstm_str} "
                f"{tfm_str} "
                f"{rr_str} "
                f"{str(row.get('pullback_type', ''))[:6]:>6} "
                f"{format_value(row.get('ma60')):>8} "
                f"{format_value(row.get('pe')):>6} "
                f"{format_value(row.get('pb')):>6} "
                f"{tags_str}"
            )
    else:
        logger.info("   (本期無多維共振推薦股票)")

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
                f"{row.get('pullback_type', ''):<18} "
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
