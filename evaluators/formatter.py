"""
evaluators/formatter.py - 終端機與多管道推播訊息排版格式化工具
包含宏觀風控卡片、盤勢重點解讀、多維共振 (四重/三重/雙模型) 優先推薦與分項策略卡片。
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
# 動態模型時序預測萃取工具 (零硬編碼，熱插拔模型架構)
# ==============================================================================

def get_candidate_model_potentials(cand: dict) -> List[Dict[str, Any]]:
    """
    動態抓取 candidate 中所有模型的時序預測，絕不寫死單一模型名稱。
    未來熱插拔任何新時序大模型，自動識別並輸出。
    """
    results = []
    seen_names = set()
    
    # 1. 優先檢查動態外掛註冊表
    plugin_name_map = {}
    try:
        from models.protocol import model_registry
        for p in model_registry.list_active_plugins():
            p_id = getattr(p, "plugin_id", None) or getattr(p, "tag_name", "").lower() or p.model_id.split("/")[-1].split("-")[0].lower()
            disp_name = getattr(p, "tag_name", None) or getattr(p, "display_name", p_id)
            if "(" in disp_name:
                disp_name = disp_name.split("(")[0].strip()
            plugin_name_map[p_id] = disp_name
    except Exception:
        pass
        
    std_map = {
        "timesfm": "TimesFM",
        "timestral": "Timestral",
        "moirai": "MOIRAI-2",
    }
    for k, v in std_map.items():
        if k not in plugin_name_map:
            plugin_name_map[k] = v

    for k, v in cand.items():
        if k.endswith("_potential") and v is not None and pd.notna(v):
            model_key = k[:-10].lower()
            if model_key in ("composite", "synthetic", "ensemble", "avg_ml"):
                continue
            display_name = plugin_name_map.get(model_key, model_key.capitalize())
            pred_price = cand.get(f"{model_key}_predicted_price")
            p10 = cand.get(f"{model_key}_p10")
            p90 = cand.get(f"{model_key}_p90")
            rr = cand.get(f"{model_key}_risk_reward_ratio") or cand.get("risk_reward_ratio")
            results.append({
                "key": model_key,
                "name": display_name,
                "potential": float(v),
                "predicted_price": float(pred_price) if pred_price is not None and pd.notna(pred_price) else None,
                "p10": float(p10) if p10 is not None and pd.notna(p10) else None,
                "p90": float(p90) if p90 is not None and pd.notna(p90) else None,
                "risk_reward_ratio": float(rr) if rr is not None and pd.notna(rr) else None,
            })
            seen_names.add(display_name.lower())

    # 2. 如果 cand 是一筆獨立的單模型紀錄
    single_model = cand.get("model_name")
    single_pot = cand.get("potential")
    if single_model and single_pot is not None and pd.notna(single_pot):
        sm_clean = str(single_model).strip()
        if sm_clean.lower() not in seen_names and sm_clean not in ("多維共振", "綜合優選"):
            results.append({
                "key": sm_clean.lower(),
                "name": sm_clean,
                "potential": float(single_pot),
                "predicted_price": float(cand.get("predicted_price")) if cand.get("predicted_price") is not None and pd.notna(cand.get("predicted_price")) else None,
                "p10": None,
                "p90": None,
                "risk_reward_ratio": float(cand.get("risk_reward_ratio")) if cand.get("risk_reward_ratio") is not None and pd.notna(cand.get("risk_reward_ratio")) else None,
            })
            
    return results


def get_candidate_bullish_models(cand: dict) -> List[str]:
    """取得特定標的之所有看多模型名稱清單"""
    pots = get_candidate_model_potentials(cand)
    return [p["name"] for p in pots if p["potential"] > 0]


def get_report_model_sections(results: dict) -> List[Tuple[str, list]]:
    """動態提取 results 字典中包含的所有時序模型預測結果清單"""
    sections = []
    seen = set()
    try:
        from models.protocol import model_registry
        for p in model_registry.list_active_plugins():
            p_id = getattr(p, "plugin_id", None) or getattr(p, "tag_name", "").lower() or p.model_id.split("/")[-1].split("-")[0].lower()
            key = f"{p_id}_results"
            res_list = results.get(key)
            if res_list and isinstance(res_list, list):
                disp_name = getattr(p, "tag_name", None) or getattr(p, "display_name", p_id)
                if "(" in disp_name:
                    disp_name = disp_name.split("(")[0].strip()
                sections.append((disp_name, res_list))
                seen.add(disp_name.lower())
    except Exception:
        pass
    
    # 檢查 results 中任何其他 *_results
    for k, v in results.items():
        if k.endswith("_results") and isinstance(v, list) and v:
            key_name = k[:-8]
            if key_name in ("xuantie", "overlap", "sector_rotation", "institutional"):
                continue
            if key_name.lower() not in seen:
                name_disp = "TimesFM" if key_name == "timesfm" else ("Timestral" if key_name == "timestral" else ("MOIRAI-2" if key_name == "moirai" else key_name.capitalize()))
                sections.append((name_disp, v))
                seen.add(name_disp.lower())
    return sections


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
    if ticker.endswith(".T"):
        price_str = f"¥{curr_price:,.0f}" if curr_price > 0 else "N/A"
    else:
        price_str = f"{curr_price:,.2f}" if curr_price > 0 else "N/A"

    # Clef badge or Concurring models badge
    clef_verdict = cand.get("clef_verdict") or {}
    clef_tag = cand.get("clef_tag") or clef_verdict.get("tag")
    bullish_models = get_candidate_bullish_models(cand)

    badge = ""
    if clef_tag:
        clean_clef_tag = str(clef_tag).replace("(本地規則)", "").strip()
        badge = f"[{html.escape(clean_clef_tag)}]"
    elif bullish_models:
        badge = f"[{html.escape(', '.join(bullish_models))}]"

    line1 = f"• <a href=\"{stock_url}\"><b>{ticker_label}</b></a> <code>{price_str}</code>"
    if badge:
        line1 += f" {badge}"

    # Line 2: dynamic model potential & core highlights (零硬編碼)
    highlights = []
    pots = get_candidate_model_potentials(cand)
    for p in pots:
        rr = p.get("risk_reward_ratio")
        rr_str = f" ({float(rr):.1f}x)" if rr and pd.notna(rr) and float(rr) > 0 else ""
        highlights.append(f"{p['name']} <b>{p['potential']:+.1f}%</b>{rr_str}")

    # Pullback / MA support
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

    msg = f"<b>多維量化投資日報｜{html.escape(index_name)}</b>\n"
    msg += f"計算時間: {calculation_time}\n\n"

    # 1 行極簡大盤風控 (純文字無 emoji)
    if macro:
        msg += f"大盤風控: <b>{html.escape(str(macro.regime_name))}</b> (建議曝險 {_format_exposure(macro)} · VIX {_format_vix(macro)})\n\n"

    # 1 段極簡 AI 操盤解讀
    if summary_text:
        raw_summary = summary_text.strip()
        if len(raw_summary) > 300:
            raw_summary = raw_summary[:297] + "..."
        clean_summary = html.escape(raw_summary)
        msg += f"<b>【盤勢重點解讀】</b>\n<blockquote>{clean_summary}</blockquote>\n\n"

    # 【多因子優選清單】
    msg += "<b>【多因子優選清單】</b>\n"
    if candidates:
        for cand in candidates[:10]:
            msg += _format_telegram_candidate_card(cand, lookup, base_url=base_url) + "\n\n"
    else:
        msg += "<i>(本期無符合多因子正向條件之標的，建議防守觀望)</i>\n\n"

    # 網站完整查詢引導 Footer
    index_url = f"{base_url}/?index={quote(str(index_name))}"
    msg += "<b>詳細圖表、時序模型歷史勝率與法人籌碼請至網站查詢：</b>\n"
    msg += f"<a href=\"{index_url}\"><b>開啟【{html.escape(index_name)}】完整互動儀表板 ↗</b></a>\n"

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
    if ticker.endswith(".T"):
        price_str = f"¥{curr_price:,.0f}" if curr_price > 0 else "N/A"
    else:
        price_str = f"{curr_price:,.2f}" if curr_price > 0 else "N/A"

    lines = [f"• **{ticker_label}** `{price_str}`"]

    details = []

    # 1. 動態 ML 模型預測 (零硬編碼)
    fm_details = []
    pots = get_candidate_model_potentials(cand)
    for p in pots:
        pot_val = p["potential"]
        rr = p.get("risk_reward_ratio")
        rr_str = f" ({rr:.1f}x)" if rr and rr > 0 else ""
        p10 = p.get("p10")
        p90 = p.get("p90")
        q_str = f" [{p10:.0f}~{p90:.0f}]" if p10 is not None and p90 is not None else ""
        fm_details.append(f"{p['name']} **{pot_val:+.1f}%**{rr_str}{q_str}")

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
    overlap_df = results.get("overlap_results", pd.DataFrame())
    macro = macro_state or results.get("macro_state")
    summary_text = ai_summary or results.get("ai_summary", "")

    candidates = _extract_candidates(results, overlap_df)

    msg = "**多維量化投資日報**\n"
    msg += f"計算時間: {calculation_time} | 指數: **{index_name}**\n\n"

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
        msg += f"**【盤勢重點解讀】**:\n> {quoted}\n\n"

    # 【多因子優選清單】
    msg += "**【多因子優選清單】**\n"
    if candidates:
        msg += f"符合條件標的: **{len(candidates)}** 支\n\n"
        for cand in candidates[:10]:
            msg += _format_discord_candidate_card(cand, lookup) + "\n\n"
    else:
        msg += "*(本期無符合多因子正向條件之標的，建議防守觀望)*\n\n"

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

    # 動態遍歷時序大模型預測清單 (零硬編碼)
    model_sections = get_report_model_sections(results)
    for model_name, m_results in model_sections:
        msg += f"**【時序大模型 ({model_name} 預測 TOP 5)】**\n"
        for idx, result in enumerate(m_results[:5], 1):
            ticker_label = format_ticker_label(result['ticker'], lookup)
            pot = result.get('potential', 0.0)
            curr_p = result.get('current_price')
            pred_p = result.get('predicted_price') or result.get('horizon_predicted_price') or curr_p
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: **{rr:.1f}x**" if rr and pd.notna(rr) else ""
            p10 = result.get('p10')
            p90 = result.get('p90')
            band_str = f" | 區間 `{p10:.1f}~{p90:.1f}`" if p10 is not None and p90 is not None and pd.notna(p10) and pd.notna(p90) else ""
            curr_str = f"`{curr_p:,.1f}` → " if curr_p is not None else ""
            msg += f"{idx}. **{ticker_label}** **{pot:+.1f}%** ({curr_str}`{pred_p:,.1f}`{band_str}{rr_str})\n"
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
    ai_summary: str = "",
    base_url: str = "https://stockdata.david888.com"
) -> str:
    """格式化量化結果為 Email 純文字格式 (包含完整個股查圖連結)"""
    lookup = name_map or {}
    xuantie_df = results.get("xuantie_results", pd.DataFrame())
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
            msg += f"台股大盤與風控: {macro.regime_name} (建議曝險 {_format_exposure(macro)}) | 加權: {twii_str} | 國際連動: {sox_str} (VIX {_format_vix(macro)})\n\n"
        else:
            spy_str = _format_trend(getattr(macro, 'spy_above_ma60', None), '站穩MA60', '破季線')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '站穩MA60', '破季線')
            msg += f"美股宏觀風控: {macro.regime_name} (建議曝險 {_format_exposure(macro)}) | VIX: {_format_vix(macro)} | SPY: {spy_str} | SOX: {sox_str}\n\n"

    if summary_text:
        msg += f"盤勢重點解讀:\n{summary_text}\n\n"

    msg += "=" * 75 + "\n\n"

    # 【多因子優選清單】
    msg += f"【多因子優選清單】 - 符合條件: {len(candidates)} 支\n\n"
    if candidates:
        for idx, cand in enumerate(candidates, 1):
            ticker = str(cand.get("ticker", "")).strip()
            stock_url = f"{base_url}/?tab=history&ticker={quote(ticker)}"
            ticker_label = format_ticker_label(ticker, lookup)
            price_val = cand.get("current_price", 0.0)

            bullish_models = get_candidate_bullish_models(cand)
            badge_str = f"[{', '.join(bullish_models)}]" if bullish_models else "[多因子選股]"

            pots = get_candidate_model_potentials(cand)
            ml_p_parts = []
            for p in pots:
                rr_info = f"({p['risk_reward_ratio']:.1f}x)" if p.get('risk_reward_ratio') else ""
                q_info = f"[{p['p10']:.0f}~{p['p90']:.0f}]" if p.get('p10') is not None and p.get('p90') is not None else ""
                ml_p_parts.append(f"{p['name']}: {p['potential']:+.1f}%{rr_info}{q_info}")

            ml_p_str = " | ".join(ml_p_parts)

            tags_str = " · ".join(cand.get("tags", []))
            pe_val = cand.get("pe") or (cand.get("fundamentals") or {}).get("pe")
            pb_val = cand.get("pb") or (cand.get("fundamentals") or {}).get("pb")
            pe_str = f"PE:{float(pe_val):.1f}" if pe_val and pd.notna(pe_val) else ""
            pb_str = f"PB:{float(pb_val):.1f}" if pb_val and pd.notna(pb_val) else ""
            val_str = f" | {pe_str} {pb_str}".strip() if (pe_str or pb_str) else ""

            msg += f"{idx}. {badge_str} {ticker_label} 現價: {price_val:,.2f} ({stock_url})\n"
            if ml_p_str:
                msg += f"   預測: {ml_p_str}\n"
            if tags_str:
                msg += f"   標籤: {tags_str}{val_str}\n"
            msg += "\n"
    else:
        msg += "(本期無符合多因子正向條件之標的)\n\n"

    # 波段操作 (季線支撐)
    msg += f"波段操作 (季線支撐) - 符合條件: {len(xuantie_df)} 支 (顯示前10名)\n\n"
    if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
        for idx, row in xuantie_df.head(10).iterrows():
            r_ticker = str(row['ticker']).strip()
            stock_url = f"{base_url}/?tab=history&ticker={quote(r_ticker)}"
            ticker_label = format_ticker_label(r_ticker, lookup)
            pe_val = row.get('pe')
            pe_str = f"PE:{pe_val:.1f}" if pd.notna(pe_val) and pe_val else "PE:N/A"
            pb_val = row.get('pb')
            pb_str = f"PB:{pb_val:.1f}" if pd.notna(pb_val) and pb_val else "PB:N/A"
            msg += f"{idx+1}. {ticker_label} 現價: {row['current_price']:,.2f} ({stock_url}) | 回調: {row.get('pullback_type', '')} | {pe_str} · {pb_str}\n"
        msg += "\n"

    # 動態遍歷時序大模型預測清單 (零硬編碼)
    model_sections = get_report_model_sections(results)
    for model_name, m_results in model_sections:
        msg += f"時序大模型 ({model_name}) - 預測完成: {len(m_results)} 支\n\n"
        msg += f"{model_name} 看漲 TOP 10 (含 隔日目標價、分位數與盈虧比)\n"
        for idx, result in enumerate(m_results[:10], 1):
            r_ticker = str(result['ticker']).strip()
            stock_url = f"{base_url}/?tab=history&ticker={quote(r_ticker)}"
            ticker_label = format_ticker_label(r_ticker, lookup)
            rr = result.get('risk_reward_ratio')
            rr_str = f" | 盈虧比: {rr:.2f}x" if rr and pd.notna(rr) else ""
            p10 = result.get('p10')
            p90 = result.get('p90')
            q_str = f" [{p10:.1f}~{p90:.1f}]" if p10 is not None and p90 is not None else ""
            curr_p = result.get('current_price') or 0.0
            pred_p = result.get('predicted_price') or result.get('horizon_predicted_price') or curr_p
            msg += f"{idx}. {ticker_label} 漲幅: {result.get('potential', 0.0):>+6.2f}% ({curr_p:,.1f} → {pred_p:,.1f}{q_str}{rr_str}) - {stock_url}\n"
        msg += "\n"

    msg += f"進入 888 StockQuant 操盤看板: {base_url}/?index={quote(index_name)}\n"
    return msg


def format_email_html(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = "",
    base_url: str = "https://stockdata.david888.com"
) -> str:
    """產出結構優雅、相容各大郵件客戶端（Gmail, Apple Mail, Outlook）的響應式 HTML 郵件"""
    lookup = name_map or {}
    xuantie_df = results.get("xuantie_results", pd.DataFrame())
    overlap_df = results.get("overlap_results", pd.DataFrame())
    macro = macro_state or results.get("macro_state")
    summary_text = ai_summary or results.get("ai_summary", "")

    candidates = _extract_candidates(results, overlap_df)

    is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")
    index_dashboard_url = f"{base_url}/?index={quote(index_name)}"

    # 宏觀大盤指標列
    macro_html = ""
    if macro:
        exposure_badge = f'<span style="background-color: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 13px;">建議曝險 {_format_exposure(macro)}</span>'
        regime_title = html.escape(str(macro.regime_name))
        vix_str = _format_vix(macro)
        if is_tw:
            twii_str = _format_trend(getattr(macro, 'twii_above_ma60', None), '加權站穩MA60', '加權跌破MA60')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '費半站穩季線', '費半破季線')
            details_line = f"加權指數: <strong>{twii_str}</strong> · 國際連動: <strong>{sox_str}</strong> · VIX: <strong>{vix_str}</strong>"
        else:
            spy_str = _format_trend(getattr(macro, 'spy_above_ma60', None), 'SPY站穩MA60', 'SPY破季線')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), 'SOX站穩MA60', 'SOX破季線')
            details_line = f"SPY: <strong>{spy_str}</strong> · SOX: <strong>{sox_str}</strong> · VIX: <strong>{vix_str}</strong>"

        macro_html = f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px; margin: 16px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 15px; font-weight: 800; color: #0F172A;">
                    宏觀風控狀態: {regime_title}
                </div>
                <div>{exposure_badge}</div>
            </div>
            <div style="font-size: 13px; color: #475569; margin-top: 6px; line-height: 1.5;">
                {details_line}
            </div>
        </div>
        """

    # AI 盤勢重點解讀卡片
    ai_html = ""
    if summary_text:
        safe_summary = html.escape(summary_text).replace("\n", "<br/>")
        ai_html = f"""
        <div style="background-color: #FFFDF7; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 8px; padding: 14px 16px; margin: 16px 0; line-height: 1.65; color: #451A03; font-size: 14px;">
            <div style="font-size: 14px; font-weight: 800; color: #B45309; margin-bottom: 6px;">
                AI 盤勢重點解讀
            </div>
            <div>{safe_summary}</div>
        </div>
        """

    # 優先推薦 / 多因子優選 Cards
    candidates_cards_html = ""
    if candidates:
        for idx, cand in enumerate(candidates, 1):
            ticker = str(cand.get("ticker", "")).strip()
            stock_url = f"{base_url}/?tab=history&ticker={quote(ticker)}"
            ticker_name = lookup.get(ticker, "")
            ticker_label = f"{ticker} {ticker_name}".strip() if ticker_name else ticker
            safe_label = html.escape(ticker_label)

            curr_p = cand.get("current_price") or 0.0
            if ticker.endswith(".T"):
                price_str = f"¥{curr_p:,.0f}" if curr_p > 0 else "N/A"
            else:
                price_str = f"${curr_p:,.2f}" if curr_p > 0 else "N/A"

            pred_p = cand.get("predicted_price")
            pred_p_str = ""
            if pred_p and pd.notna(pred_p) and float(pred_p) > 0:
                if ticker.endswith(".T"):
                    pred_p_str = f"¥{float(pred_p):,.0f}"
                else:
                    pred_p_str = f"${float(pred_p):,.2f}"

            pot = cand.get("potential")
            pot_html = ""
            if pot is not None and pd.notna(pot):
                pot_val = float(pot)
                if pot_val >= 0:
                    pot_html = f'<span style="background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; font-size: 12px; font-weight: 800; padding: 2px 8px; border-radius: 9999px;">+{pot_val:.2f}%</span>'
                else:
                    pot_html = f'<span style="background-color: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; font-size: 12px; font-weight: 800; padding: 2px 8px; border-radius: 9999px;">{pot_val:.2f}%</span>'

            tier_str = cand.get("resonance_tier") or ""
            safe_tier = html.escape(tier_str)
            tier_badge = f'<span style="background-color: #FEF3C7; color: #92400E; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 4px; margin-left: 6px;">{safe_tier}</span>' if safe_tier else ""

            # 動態遍歷各時序模型預測
            model_items = []
            for mp in get_candidate_model_potentials(cand):
                m_name = html.escape(mp["name"])
                m_pot = mp["potential"]
                m_pred = mp.get("predicted_price")
                m_target = f" · 目標 <code>${float(m_pred):,.1f}</code>" if m_pred and pd.notna(m_pred) and float(m_pred) > 0 else ""
                p10 = mp.get("p10")
                p90 = mp.get("p90")
                q_str = f" · 區間 <code>[{p10:.0f}~{p90:.0f}]</code>" if p10 is not None and p90 is not None and pd.notna(p10) and pd.notna(p90) else ""
                rr = mp.get("risk_reward_ratio")
                rr_str = f" · 盈虧比 <strong>{float(rr):.1f}x</strong>" if rr and pd.notna(rr) and float(rr) > 0 else ""
                color = "#059669" if m_pot >= 0 else "#DC2626"
                model_items.append(f"<div>• <strong>{m_name}:</strong> <span style='color: {color}; font-weight: 700;'>{m_pot:+.1f}%</span>{m_target}{q_str}{rr_str}</div>")

            models_box = ""
            if model_items:
                models_box = f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; margin: 10px 0; font-size: 13px; line-height: 1.6; color: #334155;">
                    {''.join(model_items)}
                </div>
                """

            # 估值與籌碼
            fund_parts = []
            pe_val = cand.get("pe") or (cand.get("fundamentals") or {}).get("pe")
            pb_val = cand.get("pb") or (cand.get("fundamentals") or {}).get("pb")
            if pe_val and pd.notna(pe_val) and float(pe_val) > 0:
                fund_parts.append(f"PE: <strong>{float(pe_val):.1f}</strong>")
            if pb_val and pd.notna(pb_val) and float(pb_val) > 0:
                fund_parts.append(f"PB: <strong>{float(pb_val):.1f}</strong>")

            pb_type = cand.get("pullback_type")
            if pb_type:
                fund_parts.append(f"技術面: <strong>{html.escape(str(pb_type).strip())}</strong>")

            inst = cand.get("institutional") or {}
            if inst.get("is_sync_buy"):
                fund_parts.append("<span style='color: #B45309; font-weight: 700;'>土洋合買</span>")
            elif inst.get("trust_streak", 0) >= 3:
                fund_parts.append(f"<span style='color: #B45309; font-weight: 700;'>投信連買{inst.get('trust_streak')}天</span>")
            elif inst.get("trust_net_5d", 0) > 0:
                fund_parts.append("<span style='color: #B45309; font-weight: 700;'>投信買超</span>")

            fund_line = f"<div style='font-size: 12px; color: #64748B; margin-top: 6px;'>{' · '.join(fund_parts)}</div>" if fund_parts else ""

            # 標籤
            tags = cand.get("tags") or []
            tags_html = ""
            if tags:
                tag_pills = [f"<span style='display: inline-block; background-color: #F1F5F9; color: #475569; font-size: 11px; padding: 2px 7px; border-radius: 4px; margin-right: 4px; margin-top: 4px;'>{html.escape(str(t))}</span>" for t in tags[:5]]
                tags_html = f"<div style='margin-top: 8px;'>{''.join(tag_pills)}</div>"

            target_display = f" → 目標: <strong style='color: #059669;'>{pred_p_str}</strong>" if pred_p_str else ""

            candidates_cards_html += f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 12px; font-weight: 800; color: #94A3B8; margin-right: 6px;">#{idx}</span>
                        <a href="{stock_url}" target="_blank" style="color: #2563EB; font-weight: 800; font-size: 16px; text-decoration: none;">
                            {safe_label} ↗
                        </a>
                        {tier_badge}
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 14px; color: #1E293B;">現價: <strong>{price_str}</strong>{target_display}</span>
                        {pot_html}
                    </div>
                </div>
                {models_box}
                {fund_line}
                {tags_html}
            </div>
            """
    else:
        candidates_cards_html = """
        <div style="background-color: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 10px; padding: 20px; text-align: center; color: #64748B; font-size: 14px; margin-bottom: 16px;">
            (本期無多因子一致條件之標的，建議防守觀望)
        </div>
        """

    # 輔助函式：模型排行清單產生
    def _build_model_top_html(title: str, results_list: list, is_bearish: bool = False) -> str:
        if not results_list:
            return ""
        items_html = ""
        for i, item in enumerate(results_list[:5], 1):
            tk = str(item.get("ticker", "")).strip()
            url = f"{base_url}/?tab=history&ticker={quote(tk)}"
            tk_name = lookup.get(tk, "")
            lbl = f"{tk} {tk_name}".strip() if tk_name else tk
            safe_lbl = html.escape(lbl)
            pot_v = float(item.get("potential", 0.0))
            if is_bearish:
                badge = f'<span style="background-color: #FEF2F2; color: #991B1B; font-size: 11px; font-weight: 800; padding: 1px 6px; border-radius: 9999px;">{pot_v:+.1f}%</span>'
            else:
                badge = f'<span style="background-color: #ECFDF5; color: #065F46; font-size: 11px; font-weight: 800; padding: 1px 6px; border-radius: 9999px;">{pot_v:+.1f}%</span>'

            curr_p = item.get("current_price") or 0.0
            pred_p = item.get("predicted_price") or item.get("horizon_predicted_price")
            p_info = f"<code>${curr_p:,.1f}</code> → <code>${pred_p:,.1f}</code>" if (curr_p and pred_p) else ""

            items_html += f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;">
                <div>
                    <span style="color: #94A3B8; font-size: 11px; width: 18px; display: inline-block;">#{i}</span>
                    <a href="{url}" target="_blank" style="color: #2563EB; font-weight: 700; text-decoration: none;">{safe_lbl} ↗</a>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 12px; color: #64748B;">{p_info}</span>
                    {badge}
                </div>
            </div>
            """

        header_color = "#991B1B" if is_bearish else "#0F172A"
        bg_color = "#FFF5F5" if is_bearish else "#FFFFFF"
        border_color = "#FECACA" if is_bearish else "#E2E8F0"

        return f"""
        <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: 800; color: {header_color}; margin-bottom: 6px;">
                {title}
            </div>
            {items_html}
        </div>
        """

    # 動態產生各時序模型看漲與看跌清單
    model_sections = get_report_model_sections(results)
    bullish_htmls = []
    bearish_htmls = []

    for m_name, m_list in model_sections:
        b_list = [r for r in m_list if (r.get("potential") or 0) > 0]
        if b_list:
            b_list = sorted(b_list, key=lambda x: x.get("potential", 0.0), reverse=True)
            bullish_htmls.append(_build_model_top_html(f"{m_name} 看漲 TOP 5", b_list))

        d_list = [r for r in m_list if (r.get("potential") or 0) < 0]
        if d_list:
            d_list = sorted(d_list, key=lambda x: x.get("potential", 0.0))
            bearish_htmls.append(_build_model_top_html(f"{m_name} 預測跌幅避險 TOP 5", d_list, is_bearish=True))

    bearish_section = ""
    if bearish_htmls:
        bearish_section = f"""
        <div style="margin-top: 24px;">
            <h2 style="font-size: 16px; font-weight: 800; color: #991B1B; margin: 0 0 12px 0; border-bottom: 2px solid #FECACA; padding-bottom: 6px;">
                時序大模型看跌避險焦點 (提防破底風險)
            </h2>
            {''.join(bearish_htmls)}
        </div>
        """

    # 波段操作 (季線支撐) Top 5
    xuantie_html = ""
    if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
        x_rows = ""
        for idx, row in xuantie_df.head(5).iterrows():
            tk = str(row['ticker']).strip()
            url = f"{base_url}/?tab=history&ticker={quote(tk)}"
            tk_name = lookup.get(tk, "")
            lbl = f"{tk} {tk_name}".strip() if tk_name else tk
            price_val = f"${row['current_price']:,.2f}"
            pb_type = str(row.get('pullback_type', '')).strip()
            x_rows += f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;">
                <div>
                    <span style="color: #94A3B8; font-size: 11px; width: 18px; display: inline-block;">#{idx+1}</span>
                    <a href="{url}" target="_blank" style="color: #2563EB; font-weight: 700; text-decoration: none;">{html.escape(lbl)} ↗</a>
                </div>
                <div style="font-size: 12px; color: #475569;">
                    <span>{price_val}</span> · <span style="font-weight: 600; color: #B45309;">{html.escape(pb_type)}</span>
                </div>
            </div>
            """
        xuantie_html = f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: 800; color: #0F172A; margin-bottom: 6px;">
                波段操作 (季線支撐買點 TOP 5)
            </div>
            {x_rows}
        </div>
        """

    bullish_combined_html = "".join(bullish_htmls)
    if xuantie_html:
        bullish_combined_html += xuantie_html

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>888 StockQuant 多維量化日報 - {html.escape(index_name)}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #F1F5F9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1E293B;">
    <div style="max-width: 680px; margin: 24px auto; background-color: #FFFFFF; border-radius: 16px; overflow: hidden; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
        
        <!-- Top Slate Header Banner -->
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: #FFFFFF; padding: 24px 24px 20px 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="background-color: #F59E0B; color: #1E1B18; font-weight: 800; font-size: 11px; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;">888 StockQuant</span>
                    <h1 style="margin: 8px 0 4px 0; font-size: 22px; font-weight: 900; letter-spacing: -0.5px;">
                        多維量化策略投資日報
                    </h1>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 15px; font-weight: 800; color: #F59E0B;">{html.escape(index_name)}</div>
                    <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">{calculation_time}</div>
                </div>
            </div>
        </div>

        <!-- Main Content Area -->
        <div style="padding: 20px 24px 28px 24px;">
            
            {macro_html}
            {ai_html}

            <!-- 優先推薦區塊 -->
            <div style="margin-top: 20px;">
                <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 12px; border-bottom: 2px solid #E2E8F0; padding-bottom: 6px;">
                    <h2 style="font-size: 17px; font-weight: 900; color: #0F172A; margin: 0;">
                        【多因子優選清單】
                    </h2>
                    <span style="font-size: 13px; font-weight: 700; color: #64748B;">共 {len(candidates)} 檔符合</span>
                </div>
                {candidates_cards_html}
            </div>

            <!-- 時序大模型看漲焦點 -->
            <div style="margin-top: 24px;">
                <h2 style="font-size: 16px; font-weight: 800; color: #0F172A; margin: 0 0 12px 0; border-bottom: 2px solid #E2E8F0; padding-bottom: 6px;">
                    時序大模型獨立預測看漲焦點 (TOP 5)
                </h2>
                {bullish_combined_html}
            </div>

            {bearish_section}

            <!-- Footer CTA Button -->
            <div style="text-align: center; margin: 32px 0 16px 0;">
                <a href="{index_dashboard_url}" target="_blank" style="display: inline-block; background-color: #F59E0B; color: #1E1B18; font-weight: 900; font-size: 15px; padding: 14px 32px; border-radius: 12px; text-decoration: none; box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.25);">
                    進入 888 StockQuant 操盤看板 ↗
                </a>
            </div>

            <!-- Disclaimer -->
            <p style="font-size: 11px; color: #94A3B8; text-align: center; line-height: 1.6; margin-top: 20px; border-top: 1px solid #F1F5F9; padding-top: 16px;">
                免責聲明：本量化日報僅供量化策略研究與歷史回測參考，不構成任何有價證券之投資買賣建議。投資人應獨立評估市場波動風險並自負盈虧。
            </p>

        </div>
    </div>
</body>
</html>"""
    return full_html
    return full_html


def format_dual_strategy_message(
    index_name: str,
    results: dict,
    calculation_time: str,
    name_map: Optional[Dict[str, str]] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
) -> dict:
    """提供 Telegram, Discord, Email (純文字 + 響應式 HTML) 之多渠道整合字典輸出"""
    return {
        "telegram": format_telegram_message(
            index_name, results, calculation_time, name_map=name_map, macro_state=macro_state, ai_summary=ai_summary
        ),
        "discord": format_discord_message(
            index_name, results, calculation_time, name_map=name_map, macro_state=macro_state, ai_summary=ai_summary
        ),
        "email": format_email_message(
            index_name, results, calculation_time, name_map=name_map, macro_state=macro_state, ai_summary=ai_summary
        ),
        "email_html": format_email_html(
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
    overlap_df = report.overlap_results
    macro = report.macro_state

    logger.info(f"\n{'='*100}")
    logger.info(f"投資建議報告 - {index_name}")
    logger.info(f"{'='*100}\n")

    # ====== 頂層 1: 大盤與宏觀風控 ======
    if macro:
        is_jp = (getattr(macro, "market", "") == "JP") or any(k in index_name for k in ("日經", "日股", "JP225", "N225", "NIKKEI")) or index_name.endswith(".T")
        is_tw = (getattr(macro, "market", "") == "TW") or index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051", "台股活躍精選", "台股1000", "TW1000") or any(k in index_name for k in ("台股", "台灣")) or index_name.endswith(".TW")
        if is_jp:
            logger.info("【日股大盤與科技連動風向】")
            logger.info(f"   • 大盤狀態: {macro.regime_name} (建議曝險: {_format_exposure(macro)})")
            n225_str = _format_trend(getattr(macro, 'twii_above_ma60', None), '站穩MA60', '跌破MA60')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '費半站穩季線', '費半破季線')
            logger.info(f"   • 關鍵指標: 日經225={n225_str} | 國際連動={sox_str} (VIX={_format_vix(macro)})")
        elif is_tw:
            logger.info("【台股大盤與籌碼風向】")
            logger.info(f"   • 大盤狀態: {macro.regime_name} (建議曝險: {_format_exposure(macro)})")
            twii_str = _format_trend(getattr(macro, 'twii_above_ma60', None), '站穩MA60', '跌破MA60')
            sox_str = _format_trend(getattr(macro, 'sox_above_ma60', None), '費半站穩季線', '費半破季線')
            logger.info(f"   • 關鍵指標: 加權指數={twii_str} | 國際連動={sox_str} (VIX={_format_vix(macro)})")
        else:
            logger.info("【美股宏觀環境與風控門檻】")
            logger.info(f"   • 市場狀態: {macro.regime_name} (建議曝險: {_format_exposure(macro)})")
            logger.info(f"   • 關鍵指標: VIX={_format_vix(macro)} | SPY={_format_trend(macro.spy_above_ma60, '站穩MA60', '破季線')} | SOX={_format_trend(macro.sox_above_ma60, '站穩MA60', '破季線')}")
        if macro.warnings:
            for w in macro.warnings:
                logger.info(f"   • [警告] {w}")
        logger.info("")

    # ====== 頂層 2: 盤勢重點解讀 ======
    if report.ai_summary:
        logger.info("【盤勢重點解讀】")
        for line in report.ai_summary.split("\n"):
            logger.info(f"   {line}")
        logger.info("")

    # ====== 多策略交集 / 綜合優選 ======
    logger.info("【多因子優選清單】(時序大模型 ∩ 季線支撐 ∩ 法人鎖碼)")
    logger.info(f"   重點交集符合: {len(overlap_df)} 支\n")

    if not overlap_df.empty:
        logger.info(f"   {'排名':<4} {'等級':<12} {'代碼':<10} {'現價':>8} {'目標價':>8} {'預期漲幅':>10} {'盈虧比':>7} {'回調':>6} {'MA60':>8} {'PE':>6} {'PB':>6} {'綜合標籤'}")
        logger.info(f"   {'-'*4} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*30}")
        for idx, row in overlap_df.iterrows():
            cand = next((c for c in report.overlap_candidates if c["ticker"] == row["ticker"]), None)
            tier_str = (cand.get("resonance_tier") or row.get("resonance_tier") or "") if cand else (row.get("resonance_tier") or "")
            tags_str = " | ".join(cand["tags"]) if cand and cand.get("tags") else (" | ".join(row.get("tags", [])) if row.get("tags") else "觀察")

            curr_p = row.get("current_price", 0.0)
            pred_p = row.get("predicted_price", curr_p)
            pot_v = row.get("potential", 0.0)
            pot_str = f"{pot_v:>+9.2f}%" if pd.notna(pot_v) and pot_v is not None else "      N/A"

            rr_val = row.get('risk_reward_ratio')
            rr_str = f"{rr_val:>6.2f}x" if pd.notna(rr_val) and rr_val is not None else "   N/A"

            logger.info(
                f"   {idx+1:<4} {tier_str:<12} {row['ticker']:<10} "
                f"{curr_p:>8.2f} "
                f"{pred_p:>8.2f} "
                f"{pot_str} "
                f"{rr_str} "
                f"{str(row.get('pullback_type', ''))[:6]:>6} "
                f"{format_value(row.get('ma60')):>8} "
                f"{format_value(row.get('pe')):>6} "
                f"{format_value(row.get('pb')):>6} "
                f"{tags_str}"
            )
    else:
        logger.info("   (本期無多因子一致推薦股票)")

    logger.info("")

    # ====== 軌道 1: 季線支撐 ======
    logger.info("【波段操作】季線支撐策略 (持有 2-4 週) - 技術面買點")
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

    # ====== 各時序大模型預測動態列印 ======
    model_sections = get_report_model_sections(report.__dict__)
    for model_name, m_results in model_sections:
        logger.info(f"【時序大模型】{model_name} (隔日 T+1) - 預測漲幅排行與風控")
        logger.info(f"   預測完成: {len(m_results)} 支\n")
        logger.info(f"   {'排名':<4} {'代碼':<10} {'預測漲幅':>10} {'現價':>8} {'→':^3} {'目標價':>8} {'P10~P90區間':>16} {'盈虧比':>8}")
        logger.info(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*3} {'-'*8} {'-'*16} {'-'*8}")
        for i, result in enumerate(m_results[:10], 1):
            p10 = result.get('p10')
            p90 = result.get('p90')
            band_str = f"{p10:.1f}~{p90:.1f}" if p10 is not None and p90 is not None else "N/A"
            rr_val = result.get('risk_reward_ratio')
            rr_str = f"{rr_val:>7.2f}x" if pd.notna(rr_val) and rr_val is not None else "    N/A"
            curr_p = result.get('current_price') or 0.0
            pred_p = result.get('predicted_price') or result.get('horizon_predicted_price') or curr_p
            logger.info(
                f"   {i:<4} {result.get('ticker', ''):<10} "
                f"{result.get('potential', 0.0):>+9.2f}% "
                f"{curr_p:>8.2f} {'→':^3} "
                f"{pred_p:>8.2f} "
                f"{band_str:>16} "
                f"{rr_str:>8}"
            )
        logger.info("")

    logger.info(f"{'='*100}\n")
