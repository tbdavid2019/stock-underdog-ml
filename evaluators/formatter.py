"""
evaluators/formatter.py - 終端機與多管道推播訊息排版格式化工具
包含宏觀風控卡片、AI 操盤解讀、多維共振 (四重/三重/雙ML) 優先推薦與分項策略卡片。
全面支援 Telegram (HTML 高級卡片)、Discord (Markdown 卡片)、Email (結構化純文字) 與 Terminal Log。
"""

import html
import re
import logging
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from core.config import config
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


def _format_exposure(macro: Any) -> str:
    exposure = getattr(macro, "exposure", None)
    return f"{int(exposure * 100)}%" if exposure is not None else "資料不足"


def _format_vix(macro: Any) -> str:
    vix = getattr(macro, "vix", None)
    return f"{vix:.1f}" if vix is not None else "資料不足"


def _format_trend(flag: Optional[bool], positive: str, negative: str) -> str:
    return positive if flag is True else negative if flag is False else "資料不足"


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

def _format_telegram_candidate_card(
    cand: dict,
    lookup: Dict[str, str],
    base_url: str = "https://stockdata.david888.com"
) -> str:
    """單一候選標的高級極簡卡片排版 (報明牌、附帶完整連結至網站查圖表與法人)"""
    ticker = str(cand.get("ticker", "")).strip()
    stock_url = f"{base_url}/?tab=history&ticker={quote(ticker)}"
    ticker_label = html.escape(format_ticker_label(ticker, lookup))
    curr_price = cand.get("current_price") or 0.0
    price_str = f"{curr_price:,.2f}" if curr_price > 0 else "N/A"

    # Clef badge or Tier badge
    clef_verdict = cand.get("clef_verdict") or {}
    clef_tag = cand.get("clef_tag") or clef_verdict.get("tag")
    tier = cand.get("resonance_tier")
    if not tier:
        tags = cand.get("tags") or []
        tier_order = [
            "綜合優選", "雙AI看漲", "👑四重共振", "🏆三重共振", "🔮雙大模型共振",
            "👑雙時序共振", "🔮雙ML共振", "🌟多維共振", "⭐雙重共振"
        ]
        tier = next((t for t in tier_order if t in tags), "")

    badge = ""
    if clef_tag:
        clean_clef_tag = str(clef_tag).replace("(本地規則)", "").strip()
        badge = f"[{html.escape(clean_clef_tag)}]"
    elif tier:
        badge = f"[{html.escape(tier)}]"

    line1 = f"• <a href=\"{stock_url}\"><b>{ticker_label}</b></a> <code>{price_str}</code>"
    if badge:
        line1 += f" {badge}"

    # Line 2: key model potential & core highlights
    highlights = []

    # TimesFM
    tfm_pot = cand.get("timesfm_potential")
    if tfm_pot is not None and pd.notna(tfm_pot):
        rr = cand.get("risk_reward_ratio")
        rr_str = f" ({float(rr):.1f}x)" if rr and pd.notna(rr) and float(rr) > 0 else ""
        highlights.append(f"TFM <b>{float(tfm_pot):+.1f}%</b>{rr_str}")

    # Timestral / LSTM
    tml_pot = cand.get("timestral_potential")
    if tml_pot is not None and pd.notna(tml_pot):
        highlights.append(f"Timestral <b>{float(tml_pot):+.1f}%</b>")
    elif cand.get("lstm_potential") is not None and pd.notna(cand.get("lstm_potential")):
        highlights.append(f"LSTM <b>{float(cand.get('lstm_potential')):+0.1f}%</b>")

    # Pullback / MA support: auto-replace "玄鐵買點" with "季線支撐"
    pb_type = cand.get("pullback_type")
    tags = cand.get("tags") or []
    if pb_type:
        clean_pb = str(pb_type).replace("玄鐵買點", "季線支撐").replace("玄鐵", "季線支撐").strip()
        highlights.append(html.escape(clean_pb))
    elif any("玄鐵" in t or "季線" in t for t in tags):
        highlights.append("季線支撐")

    # Institutional flows
    inst = cand.get("institutional") or {}
    if inst.get("is_sync_buy") or "土洋合買" in tags:
        highlights.append("土洋合買")
    elif any("投信連買" in t for t in tags):
        streak_tag = next(t for t in tags if "投信連買" in t)
        highlights.append(html.escape(streak_tag))
    elif inst.get("trust_streak", 0) >= 3:
        highlights.append(f"投信連買{inst.get('trust_streak')}天")
    elif inst.get("trust_net_5d", 0) > 0 or "投信買超" in tags:
        highlights.append("投信買超")

    # Always provide detailed URL link
    highlights.append(f"<a href=\"{stock_url}\">詳情↗</a>")

    line2 = "  " + " · ".join(highlights)
    return f"{line1}\n{line2}"


def format_telegram_message(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
) -> str:
    """格式化量化結果為極簡報明牌 Telegram 訊息 (報明牌+網站連結，導流至 Web 互動儀表板)"""
    lookup = name_map or {}
    overlap_df = results.get("overlap_results", pd.DataFrame())
    macro = macro_state or results.get("macro_state")
    summary_text = ai_summary or results.get("ai_summary", "")

    candidates = _extract_candidates(results, overlap_df)
    base_url = getattr(config.api, "PUBLIC_WEB_URL", "https://stockdata.david888.com").rstrip("/")

    msg = f"<b>📊 多維量化投資日報｜{html.escape(index_name)}</b>\n"
    msg += f"⏰ {calculation_time}\n\n"

    # 1 行極簡大盤風控
    if macro:
        msg += f"🛡️ 大盤風控: <b>{html.escape(str(macro.regime_name))}</b> (建議曝險 {_format_exposure(macro)} · VIX {_format_vix(macro)})\n\n"

    # 1 段極簡 AI 操盤解讀
    if summary_text:
        raw_summary = summary_text.strip()
        if len(raw_summary) > 300:
            raw_summary = raw_summary[:297] + "..."
        clean_summary = html.escape(raw_summary)
        msg += f"<b>【AI 操盤解讀】</b>\n<blockquote>{clean_summary}</blockquote>\n\n"

    # 🎯【今日精選明牌 (優先推薦)】
    msg += "<b>🎯【今日精選明牌 (優先推薦)】</b>\n"
    if candidates:
        for cand in candidates[:10]:
            msg += _format_telegram_candidate_card(cand, lookup, base_url=base_url) + "\n\n"
    else:
        msg += "<i>(本期無符合多維正向共振條件之標的，建議防守觀望)</i>\n\n"

    # 🔗 網站完整查詢引導 Footer
    index_url = f"{base_url}/?index={quote(str(index_name))}"
    msg += "🔗 <b>詳細圖表、雙模型歷史勝率與法人籌碼請至網站查詢：</b>\n"
    msg += f"👉 <a href=\"{index_url}\"><b>開啟【{html.escape(index_name)}】完整互動儀表板</b></a>\n"

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
    fm_details = []
    tfm_pot = cand.get("timesfm_potential")
    if tfm_pot is not None and pd.notna(tfm_pot):
        pot_val = float(tfm_pot)
        rr = cand.get("risk_reward_ratio")
        rr_str = f" ({float(rr):.1f}x)" if rr and pd.notna(rr) and float(rr) > 0 else ""
        fm_details.append(f"TFM **{pot_val:+.1f}%**{rr_str}")

    tml_pot = cand.get("timestral_potential")
    if tml_pot is not None and pd.notna(tml_pot):
        pot_val = float(tml_pot)
        p10 = cand.get("timestral_p10")
        p90 = cand.get("timestral_p90")
        q_str = f" [{p10:.0f}~{p90:.0f}]" if p10 is not None and p90 is not None and pd.notna(p10) and pd.notna(p90) else ""
        fm_details.append(f"Timestral **{pot_val:+.1f}%**{q_str}")
    elif cand.get("lstm_potential") is not None and pd.notna(cand.get("lstm_potential")):
        pot_val = float(cand.get("lstm_potential"))
        fm_details.append(f"LSTM **{pot_val:+.1f}%**")

    details.extend(fm_details)

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
    timestral_results = results.get("timestral_results", [])
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
            twii_str = _format_trend(getattr(macro, 'twii_above_ma60', None), '站穩MA60', '跌破MA60')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '費半站穩季線', '費半破季線')
            msg += f"**【台股大盤與風控】**: {macro.regime_name} (建議曝險 {_format_exposure(macro)}) | 加權: {twii_str} | 國際連動: {sox_str} (VIX {_format_vix(macro)})\n\n"
        else:
            spy_str = _format_trend(getattr(macro, 'spy_above_ma60', None), '站穩MA60', '破季線')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '站穩MA60', '破季線')
            msg += f"**【美股宏觀風控】**: {macro.regime_name} (建議曝險 {_format_exposure(macro)}) | VIX: {_format_vix(macro)} | SPY: {spy_str} | SOX: {sox_str}\n\n"

    if summary_text:
        quoted = summary_text.replace("\n", "\n> ")
        msg += f"**【AI 操盤解讀】**:\n> {quoted}\n\n"

    # 優先推薦 (多維共振)
    msg += "**【優先推薦 (多維共振)】**\n"
    if candidates:
        msg += f"符合共振條件: **{len(candidates)}** 支\n\n"
        tier_display_map = {
            "綜合優選": "【綜合優選】",
            "雙AI看漲": "【雙AI看漲】",
            "👑四重共振": "【四重共振】",
            "🏆三重共振": "【三重共振】",
            "🔮雙大模型共振": "【雙AI看漲】",
            "👑雙時序共振": "【雙AI看漲】",
            "🔮雙ML共振": "【雙ML共振】",
            "🌟多維共振": "【多維共振】",
            "⭐雙重共振": "【雙重共振】",
            "四重共振": "【四重共振】",
            "三重共振": "【三重共振】",
            "雙大模型共振": "【雙AI看漲】",
            "雙ML共振": "【雙ML共振】",
            "多維共振": "【多維共振】",
            "雙重共振": "【雙重共振】"
        }
        tier_order = [
            "綜合優選", "雙AI看漲",
            "👑四重共振", "🏆三重共振", "🔮雙大模型共振", "👑雙時序共振", "🔮雙ML共振", "🌟多維共振", "⭐雙重共振",
            "四重共振", "三重共振", "雙大模型共振", "雙ML共振", "多維共振", "雙重共振"
        ]
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

    # 波段操作 (季線支撐)
    msg += "**【波段操作 (季線支撐)】**\n"
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

    # 時序大模型 (Timestral-209M)
    if timestral_results:
        msg += "**【時序大模型 (Timestral-209M 預測 TOP 5)】**\n"
        for idx, result in enumerate(timestral_results[:5], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            pot = result['potential']
            curr_p = result['current_price']
            pred_p = result.get('predicted_price', curr_p)
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: **{rr:.1f}x**" if rr and pd.notna(rr) else ""
            p10 = result.get('p10')
            p90 = result.get('p90')
            band_str = f" | 區間 `{p10:.1f}~{p90:.1f}`" if p10 is not None and p90 is not None and pd.notna(p10) and pd.notna(p90) else ""
            msg += f"{idx}. **{ticker_label}** **{pot:+.1f}%** (`{curr_p:,.1f}` → `{pred_p:,.1f}`{band_str}{rr_str})\n"
        msg += "\n"
    elif lstm_results:
        msg += "**【短線操作 (LSTM 預測 TOP 5)】**\n"
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
            curr_p = result.get('current_price')
            pred_p = result.get('predicted_price') or result.get('horizon_predicted_price')
            rr = result.get('risk_reward_ratio')
            rr_str = f"盈虧比: **{rr:.1f}x**" if rr and pd.notna(rr) else ""
            p10 = result.get('p10')
            p90 = result.get('p90')
            band_str = f"區間 `{p10:.1f}~{p90:.1f}`" if p10 is not None and p90 is not None and pd.notna(p10) and pd.notna(p90) else ""

            if curr_p is not None and pred_p is not None:
                p_str = f"`{curr_p:,.1f}` → `{pred_p:,.1f}`"
            elif pred_p is not None:
                p_str = f"隔日目標 `{pred_p:,.1f}`"
            else:
                p_str = ""

            info_parts = [p for p in [p_str, band_str, rr_str] if p]
            info_str = f" ({' | '.join(info_parts)})" if info_parts else ""
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
    timestral_results = results.get("timestral_results", [])
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
            twii_str = _format_trend(getattr(macro, 'twii_above_ma60', None), '站穩MA60', '跌破MA60')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '費半站穩季線', '費半破季線')
            msg += f"🇹🇼 台股大盤與風控: {macro.regime_name} (建議曝險 {_format_exposure(macro)}) | 加權: {twii_str} | 國際連動: {sox_str} (VIX {_format_vix(macro)})\n\n"
        else:
            spy_str = _format_trend(getattr(macro, 'spy_above_ma60', None), '站穩MA60', '破季線')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '站穩MA60', '破季線')
            msg += f"🇺🇸 美股宏觀風控: {macro.regime_name} (建議曝險 {_format_exposure(macro)}) | VIX: {_format_vix(macro)} | SPY: {spy_str} | SOX: {sox_str}\n\n"

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
            tml_p = cand.get("timestral_potential")
            tfm_p = cand.get("timesfm_potential")
            lstm_p = cand.get("lstm_potential")
            rr = cand.get("risk_reward_ratio")
            ml_p_parts = []
            if tfm_p is not None:
                rr_info = f"({float(rr):.1f}x)" if rr and pd.notna(rr) else ""
                ml_p_parts.append(f"TimesFM: {float(tfm_p):+.1f}%{rr_info}")
            if tml_p is not None:
                p10 = cand.get("timestral_p10")
                p90 = cand.get("timestral_p90")
                q_info = f"[{p10:.0f}~{p90:.0f}]" if p10 is not None and p90 is not None else ""
                ml_p_parts.append(f"Timestral: {float(tml_p):+.1f}%{q_info}")
            elif lstm_p is not None:
                ml_p_parts.append(f"LSTM: {float(lstm_p):+.1f}%")

            ml_p_str = " | ".join(ml_p_parts)

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
    msg += f"🗡️  波段操作 (季線支撐) - 符合條件: {len(xuantie_df)} 支 (顯示前10名)\n\n"
    if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
        for idx, row in xuantie_df.head(10).iterrows():
            ticker_label = format_ticker_label(row['ticker'], lookup)
            pe_val = row.get('pe')
            pe_str = f"PE:{pe_val:.1f}" if pd.notna(pe_val) and pe_val else "PE:N/A"
            pb_val = row.get('pb')
            pb_str = f"PB:{pb_val:.1f}" if pd.notna(pb_val) and pb_val else "PB:N/A"
            msg += f"{idx+1}. {ticker_label} 現價: {row['current_price']:,.2f} | 回調: {row.get('pullback_type', '')} | {pe_str} · {pb_str}\n"
        msg += "\n"

    # 🔮 Timestral-209M
    if timestral_results:
        msg += f"🔮 時序大模型 (Timestral-209M) - 預測完成: {len(timestral_results)} 支\n\n"
        msg += "📈 Timestral 看漲 TOP 10 (含 隔日目標價、P10/P90 分位數與盈虧比)\n"
        for idx, result in enumerate(timestral_results[:10], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: {rr:.2f}x" if rr and pd.notna(rr) else ""
            p10 = result.get('p10')
            p90 = result.get('p90')
            q_str = f" | 區間: [{p10:.1f}~{p90:.1f}]" if p10 is not None and p90 is not None and pd.notna(p10) and pd.notna(p90) else ""
            curr_p = result['current_price']
            pred_p = result.get('predicted_price', curr_p)
            msg += f"{idx}. {ticker_label} 漲幅: {result['potential']:>+6.2f}% ({curr_p:,.1f} → {pred_p:,.1f}{q_str}{rr_str})\n"
        msg += "\n"
    elif lstm_results:
        msg += f"🤖 短線操作 (LSTM) - 預測完成: {len(lstm_results)} 支\n\n"
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
        msg += "📈 TimesFM 看漲 TOP 10 (含 隔日目標價與盈虧比)\n"
        for idx, result in enumerate(timesfm_results[:10], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: {rr:.2f}x" if rr and pd.notna(rr) else ""
            curr_p = result.get('current_price')
            pred_p = result.get('predicted_price') or result.get('horizon_predicted_price')
            if curr_p is not None and pred_p is not None:
                p_str = f" ({curr_p:,.1f} → {pred_p:,.1f})"
            elif pred_p is not None:
                p_str = f" | 隔日目標: {pred_p:,.2f}"
            else:
                p_str = ""
            msg += f"{idx}. {ticker_label} 漲幅: {result['potential']:>+6.2f}%{p_str}{rr_str}\n"
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
    timestral_results = getattr(report, "timestral_results", [])
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
            logger.info(f"   • 大盤狀態: {macro.regime_name} (建議曝險: {_format_exposure(macro)})")
            twii_str = _format_trend(getattr(macro, 'twii_above_ma60', None), '站穩MA60', '跌破MA60')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '費半站穩季線', '費半破季線')
            logger.info(f"   • 關鍵指標: 加權指數={twii_str} | 國際連動={sox_str} (VIX={_format_vix(macro)})")
        else:
            logger.info("🇺🇸 【美股宏觀環境與風控門檻】")
            logger.info(f"   • 市場狀態: {macro.regime_name} (建議曝險: {_format_exposure(macro)})")
            logger.info(f"   • 關鍵指標: VIX={_format_vix(macro)} | SPY={_format_trend(macro.spy_above_ma60, '站穩MA60', '破季線')} | SOX={_format_trend(macro.sox_above_ma60, '站穩MA60', '破季線')}")
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

    # ====== 多策略交集 / 綜合優選 ======
    logger.info("⭐ 【優先推薦】綜合優選 (TimesFM × Timestral ∩ 季線支撐 ∩ 法人鎖碼)")
    logger.info(f"   重點交集符合: {len(overlap_df)} 支\n")

    if not overlap_df.empty:
        logger.info(f"   {'排名':<4} {'等級':<8} {'代碼':<10} {'TimesFM':>8} {'Timestral':>10} {'盈虧比':>7} {'回調':>6} {'MA60':>8} {'PE':>6} {'PB':>6} {'綜合標籤'}")
        logger.info(f"   {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*30}")
        for idx, row in overlap_df.iterrows():
            cand = next((c for c in report.overlap_candidates if c["ticker"] == row["ticker"]), None)
            tier_str = (cand.get("resonance_tier") or row.get("resonance_tier") or "共振") if cand else (row.get("resonance_tier") or "共振")
            tags_str = " | ".join(cand["tags"]) if cand and cand.get("tags") else (" | ".join(row.get("tags", [])) if row.get("tags") else "觀察")

            tfm_str = f"{row['timesfm_potential']:>+7.2f}%" if pd.notna(row.get('timesfm_potential')) and row.get('timesfm_potential') is not None else "    N/A"
            tml_val = row.get('timestral_potential')
            if pd.notna(tml_val) and tml_val is not None:
                tml_str = f"{tml_val:>+9.2f}%"
            elif pd.notna(row.get('lstm_potential')) and row.get('lstm_potential') is not None:
                tml_str = f"{row['lstm_potential']:>+9.2f}%"
            else:
                tml_str = "      N/A"

            rr_val = row.get('risk_reward_ratio')
            rr_str = f"{rr_val:>6.2f}x" if pd.notna(rr_val) and rr_val is not None else "   N/A"

            logger.info(
                f"   {idx+1:<4} {tier_str:<8} {row['ticker']:<10} "
                f"{tfm_str} "
                f"{tml_str} "
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

    # ====== 軌道 1: 季線支撐 ======
    logger.info("🗡️  【波段操作】季線支撐策略 (持有 2-4 週) - 技術面買點")
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

    # ====== 軌道 2: Timestral-209M / LSTM ======
    if timestral_results:
        logger.info("🔮 【時序大模型】Mistral Timestral-209M (隔日 T+1) - 預測漲幅排行與風控分位數")
        logger.info(f"   預測完成: {len(timestral_results)} 支\n")
        logger.info(f"   {'排名':<4} {'代碼':<10} {'預測漲幅':>10} {'現價':>8} {'→':^3} {'目標價':>8} {'P10~P90區間':>16} {'盈虧比':>8}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*3} {'-'*8} {'-'*16} {'-'*8}")
        for i, result in enumerate(timestral_results[:10], 1):
            p10 = result.get('p10')
            p90 = result.get('p90')
            band_str = f"{p10:.1f}~{p90:.1f}" if p10 is not None and p90 is not None else "N/A"
            rr_val = result.get('risk_reward_ratio')
            rr_str = f"{rr_val:>7.2f}x" if pd.notna(rr_val) and rr_val is not None else "    N/A"
            logger.info(
                f"   {i:<4} {result['ticker']:<10} "
                f"{result['potential']:>+9.2f}% "
                f"{result['current_price']:>8.2f} {'→':^3} "
                f"{result['predicted_price']:>8.2f} "
                f"{band_str:>16} "
                f"{rr_str:>8}"
            )
    elif lstm_results:
        logger.info("🤖 【短線操作】LSTM 預測 (持有 1-7 天) - 預測漲幅排行")
        logger.info(f"   預測完成: {len(lstm_results)} 支\n")
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
        logger.info("   (本期無時序大模型預測結果)")

    logger.info("")

    # ====== 軌道 3: TimesFM 預測 ======
    logger.info("🔮 【時序大模型】TimesFM 預測 (隔日 T+1) - 預測漲幅排行與盈虧比")
    logger.info(f"   預測完成: {len(timesfm_results)} 支\n")

    if timesfm_results:
        logger.info(f"   {'排名':<4} {'代碼':<10} {'預測漲幅':>10} {'現價':>8} {'→':^3} {'目標價':>8} {'盈虧比':>8} {'PE':>6} {'PB':>6}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*3} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
        for i, result in enumerate(timesfm_results[:10], 1):
            pred_p = result.get('predicted_price') or result.get('horizon_predicted_price') or result.get('current_price', 0.0)
            rr_val = result.get('risk_reward_ratio')
            rr_str = f"{rr_val:>7.2f}x" if pd.notna(rr_val) and rr_val is not None else "     N/A"
            logger.info(
                f"   {i:<4} {result['ticker']:<10} "
                f"{result['potential']:>+9.2f}% "
                f"{result.get('current_price', 0.0):>8.2f} {'→':^3} "
                f"{pred_p:>8.2f} "
                f"{rr_str:>8} "
                f"{format_value(result.get('pe')):>6} "
                f"{format_value(result.get('pb')):>6}"
            )
    else:
        logger.info("   (本期無 TimesFM 預測結果)")

    logger.info(f"\n{'='*100}\n")
