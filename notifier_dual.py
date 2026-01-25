"""
雙軌策略專用通知模組
支援 Telegram, Discord, Email
"""
import datetime
import pandas as pd
from typing import Dict
from notifier import send_to_telegram, send_to_discord, send_email
from config import EmailConfig
from logger import logger


def _format_ticker_label(ticker: str, name_map: Dict[str, str]) -> str:
    name = name_map.get(ticker)
    return f"{ticker} {name}" if name else ticker


def _format_ticker_cell(ticker: str, name_map: Dict[str, str], width: int) -> str:
    label = _format_ticker_label(ticker, name_map)
    return f"{label:<{width}}"


def format_dual_strategy_message(index_name: str, results: dict, calculation_time: str, name_map: Dict[str, str] = None) -> dict:
    """
    格式化雙軌策略結果為通知訊息
    
    Returns:
        dict with 'telegram', 'discord', 'email' keys
    """
    xuantie_df = results['xuantie_results']
    lstm_results = results['lstm_results']
    overlap_df = results['overlap_results']
    lookup = name_map or {}
    label_width = 16
    
    # ===== Telegram (HTML) =====
    telegram_msg = f"<b>🚀 雙軌策略投資建議</b>\n"
    telegram_msg += f"⏰ {calculation_time}\n"
    telegram_msg += f"📊 指數: <b>{index_name}</b>\n\n"
    
    # 玄鐵重劍
    telegram_msg += f"<b>🗡️ 波段操作 (玄鐵重劍)</b>\n"
    telegram_msg += f"符合條件: {len(xuantie_df)} 支 (顯示前5名)\n"
    if not xuantie_df.empty:
        telegram_msg += "<pre>\n"
        telegram_msg += f"{'代碼/名稱':<{label_width}} {'價格':>8} {'PE':>5} {'PB':>5} {'EV':>5} {'回調':<10}\n"
        for idx, row in xuantie_df.head(5).iterrows():
            pe_str = f"{row.get('pe', 0):.1f}" if row.get('pe') else "N/A"
            pb_str = f"{row.get('pb', 0):.1f}" if row.get('pb') else "N/A"
            ev_str = f"{row.get('ev_ebitda', 0):.1f}" if row.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(row['ticker'], lookup, label_width)
            telegram_msg += f"{ticker_label} {row['current_price']:>8.2f} {pe_str:>5} {pb_str:>5} {ev_str:>5} {row['pullback_type']:<10}\n"
        telegram_msg += "</pre>\n\n"
    
    # LSTM 預測
    telegram_msg += f"<b>🤖 短線操作 (LSTM)</b>\n"
    telegram_msg += f"預測完成: {len(lstm_results)} 支\n\n"
    if lstm_results:
        # 前5名 (預測上漲)
        telegram_msg += "<b>📈 預測上漲 TOP 5</b>\n"
        telegram_msg += "<pre>\n"
        telegram_msg += f"{'代碼/名稱':<{label_width}} {'漲幅':>8} {'PE':>5} {'PB':>5} {'EV':>5}\n"
        for result in lstm_results[:5]:
            pe_str = f"{result.get('pe', 0):.1f}" if result.get('pe') else "N/A"
            pb_str = f"{result.get('pb', 0):.1f}" if result.get('pb') else "N/A"
            ev_str = f"{result.get('ev_ebitda', 0):.1f}" if result.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(result['ticker'], lookup, label_width)
            telegram_msg += f"{ticker_label} {result['potential']:>+7.2f}% {pe_str:>5} {pb_str:>5} {ev_str:>5}\n"
        telegram_msg += "</pre>\n\n"
        
        # 後5名 (預測下跌)
        if len(lstm_results) > 5:
            telegram_msg += "<b>📉 預測下跌 TOP 5</b>\n"
            telegram_msg += "<pre>\n"
            telegram_msg += f"{'代碼/名稱':<{label_width}} {'跌幅':>8} {'PE':>5} {'PB':>5} {'EV':>5}\n"
            for result in lstm_results[-5:]:
                pe_str = f"{result.get('pe', 0):.1f}" if result.get('pe') else "N/A"
                pb_str = f"{result.get('pb', 0):.1f}" if result.get('pb') else "N/A"
                ev_str = f"{result.get('ev_ebitda', 0):.1f}" if result.get('ev_ebitda') else "N/A"
                ticker_label = _format_ticker_cell(result['ticker'], lookup, label_width)
                telegram_msg += f"{ticker_label} {result['potential']:>+7.2f}% {pe_str:>5} {pb_str:>5} {ev_str:>5}\n"
            telegram_msg += "</pre>\n\n"
    
    # 雙重符合
    telegram_msg += f"<b>⭐ 優先推薦 (雙重符合)</b>\n"
    telegram_msg += f"符合條件: {len(overlap_df)} 支\n"
    if not overlap_df.empty:
        telegram_msg += "<pre>\n"
        for idx, row in overlap_df.iterrows():
            pe_str = f"{row.get('pe', 0):.1f}" if row.get('pe') else "N/A"
            pb_str = f"{row.get('pb', 0):.1f}" if row.get('pb') else "N/A"
            ev_str = f"{row.get('ev_ebitda', 0):.1f}" if row.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_label(row['ticker'], lookup)
            telegram_msg += f"{ticker_label} LSTM:{row['lstm_potential']:+.1f}% EV:{ev_str} {row['pullback_type']} PE:{pe_str} PB:{pb_str}\n"
        telegram_msg += "</pre>"
    
    # ===== Discord (Markdown) =====
    discord_msg = f"**🚀 雙軌策略投資建議**\n"
    discord_msg += f"⏰ {calculation_time}\n"
    discord_msg += f"📊 指數: **{index_name}**\n\n"
    
    # 玄鐵重劍
    discord_msg += f"**🗡️ 波段操作 (玄鐵重劍)** - 符合: {len(xuantie_df)} 支 (顯示前5名)\n"
    if not xuantie_df.empty:
        discord_msg += "```\n"
        discord_msg += f"{'代碼/名稱':<{label_width}} {'價格':>8} {'PE':>5} {'PB':>5} {'EV':>5} {'回調':<10}\n"
        for idx, row in xuantie_df.head(5).iterrows():
            pe_str = f"{row.get('pe', 0):.1f}" if row.get('pe') else "N/A"
            pb_str = f"{row.get('pb', 0):.1f}" if row.get('pb') else "N/A"
            ev_str = f"{row.get('ev_ebitda', 0):.1f}" if row.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(row['ticker'], lookup, label_width)
            discord_msg += f"{ticker_label} {row['current_price']:>8.2f} {pe_str:>5} {pb_str:>5} {ev_str:>5} {row['pullback_type']:<10}\n"
        discord_msg += "```\n"
    
    # LSTM 預測
    discord_msg += f"\n**🤖 短線操作 (LSTM)** - 預測: {len(lstm_results)} 支\n\n"
    if lstm_results:
        # 前5名
        discord_msg += "**📈 預測上漲 TOP 5**\n"
        discord_msg += "```\n"
        discord_msg += f"{'代碼/名稱':<{label_width}} {'漲幅':>8} {'PE':>5} {'PB':>5} {'EV':>5}\n"
        for result in lstm_results[:5]:
            pe_str = f"{result.get('pe', 0):.1f}" if result.get('pe') else "N/A"
            pb_str = f"{result.get('pb', 0):.1f}" if result.get('pb') else "N/A"
            ev_str = f"{result.get('ev_ebitda', 0):.1f}" if result.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(result['ticker'], lookup, label_width)
            discord_msg += f"{ticker_label} {result['potential']:>+7.2f}% {pe_str:>5} {pb_str:>5} {ev_str:>5}\n"
        discord_msg += "```\n\n"
        
        # 後5名
        if len(lstm_results) > 5:
            discord_msg += "**📉 預測下跌 TOP 5**\n"
            discord_msg += "```\n"
            discord_msg += f"{'代碼/名稱':<{label_width}} {'跌幅':>8} {'PE':>5} {'PB':>5} {'EV':>5}\n"
            for result in lstm_results[-5:]:
                pe_str = f"{result.get('pe', 0):.1f}" if result.get('pe') else "N/A"
                pb_str = f"{result.get('pb', 0):.1f}" if result.get('pb') else "N/A"
                ev_str = f"{result.get('ev_ebitda', 0):.1f}" if result.get('ev_ebitda') else "N/A"
                ticker_label = _format_ticker_cell(result['ticker'], lookup, label_width)
                discord_msg += f"{ticker_label} {result['potential']:>+7.2f}% {pe_str:>5} {pb_str:>5} {ev_str:>5}\n"
            discord_msg += "```\n"
    
    # 雙重符合
    discord_msg += f"\n**⭐ 優先推薦 (雙重符合)** - 符合: {len(overlap_df)} 支\n"
    if not overlap_df.empty:
        discord_msg += "```\n"
        for idx, row in overlap_df.iterrows():
            pe_str = f"{row.get('pe', 0):.1f}" if row.get('pe') else "N/A"
            pb_str = f"{row.get('pb', 0):.1f}" if row.get('pb') else "N/A"
            ev_str = f"{row.get('ev_ebitda', 0):.1f}" if row.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_label(row['ticker'], lookup)
            discord_msg += f"{ticker_label} LSTM:{row['lstm_potential']:+.1f}% EV:{ev_str} {row['pullback_type']} PE:{pe_str} PB:{pb_str}\n"
        discord_msg += "```"
    
    # ===== Email (Plain Text) =====
    email_body = f"雙軌策略投資建議\n"
    email_body += f"運算時間: {calculation_time}\n"
    email_body += f"指數: {index_name}\n\n"
    email_body += "=" * 60 + "\n\n"
    
    # 玄鐵重劍
    email_body += f"🗡️  波段操作 (玄鐵重劍) - 符合條件: {len(xuantie_df)} 支 (顯示前10名)\n\n"
    if not xuantie_df.empty:
        email_body += f"{'代碼/名稱':<18} {'價格':>10} {'PE':>8} {'PB':>8} {'EV':>8} 回調類型\n"
        email_body += "-" * 70 + "\n"
        for idx, row in xuantie_df.head(10).iterrows():
            pe_str = f"{row.get('pe', 0):.2f}" if row.get('pe') else "N/A"
            pb_str = f"{row.get('pb', 0):.2f}" if row.get('pb') else "N/A"
            ev_str = f"{row.get('ev_ebitda', 0):.2f}" if row.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(row['ticker'], lookup, 18)
            email_body += f"{ticker_label} {row['current_price']:>10.2f} {pe_str:>8} {pb_str:>8} {ev_str:>8} {row['pullback_type']}\n"
    email_body += "\n\n"
    
    # LSTM 預測
    email_body += f"🤖 短線操作 (LSTM) - 預測完成: {len(lstm_results)} 支\n\n"
    if lstm_results:
        # 前10名 (預測上漲)
        email_body += "📈 預測上漲 TOP 10\n\n"
        email_body += f"{'代碼/名稱':<18} {'預測漲幅':>10} {'現價':>10} {'預測價':>10} {'PE':>8} {'PB':>8} {'EV':>8}\n"
        email_body += "-" * 70 + "\n"
        for result in lstm_results[:10]:
            pe_str = f"{result.get('pe', 0):.2f}" if result.get('pe') else "N/A"
            pb_str = f"{result.get('pb', 0):.2f}" if result.get('pb') else "N/A"
            ev_str = f"{result.get('ev_ebitda', 0):.2f}" if result.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(result['ticker'], lookup, 18)
            email_body += f"{ticker_label} {result['potential']:>+9.2f}% {result['current_price']:>10.2f} {result['predicted_price']:>10.2f} {pe_str:>8} {pb_str:>8} {ev_str:>8}\n"
        email_body += "\n"
        
        # 後10名 (預測下跌)
        if len(lstm_results) > 10:
            email_body += "📉 預測下跌 TOP 10\n\n"
            email_body += f"{'代碼/名稱':<18} {'預測跌幅':>10} {'現價':>10} {'預測價':>10} {'PE':>8} {'PB':>8} {'EV':>8}\n"
            email_body += "-" * 70 + "\n"
            for result in lstm_results[-10:]:
                pe_str = f"{result.get('pe', 0):.2f}" if result.get('pe') else "N/A"
                pb_str = f"{result.get('pb', 0):.2f}" if result.get('pb') else "N/A"
                ev_str = f"{result.get('ev_ebitda', 0):.2f}" if result.get('ev_ebitda') else "N/A"
                ticker_label = _format_ticker_cell(result['ticker'], lookup, 18)
                email_body += f"{ticker_label} {result['potential']:>+9.2f}% {result['current_price']:>10.2f} {result['predicted_price']:>10.2f} {pe_str:>8} {pb_str:>8} {ev_str:>8}\n"
    email_body += "\n\n"
    
    # 雙重符合
    email_body += f"⭐ 優先推薦 (雙重符合) - 符合條件: {len(overlap_df)} 支\n\n"
    if not overlap_df.empty:
        email_body += f"{'代碼/名稱':<18} {'LSTM漲幅':>10} {'EV':>8} {'回調類型':<15} {'PE':>8} {'PB':>8}\n"
        email_body += "-" * 70 + "\n"
        for idx, row in overlap_df.iterrows():
            pe_str = f"{row.get('pe', 0):.2f}" if row.get('pe') else "N/A"
            pb_str = f"{row.get('pb', 0):.2f}" if row.get('pb') else "N/A"
            ev_str = f"{row.get('ev_ebitda', 0):.2f}" if row.get('ev_ebitda') else "N/A"
            ticker_label = _format_ticker_cell(row['ticker'], lookup, 18)
            email_body += f"{ticker_label} {row['lstm_potential']:>+9.2f}% {ev_str:>8} {row['pullback_type'][:15]:<15} {pe_str:>8} {pb_str:>8}\n"
    
    return {
        'telegram': telegram_msg,
        'discord': discord_msg,
        'email': email_body
    }


def send_dual_strategy_results(index_name: str, results: dict, name_map: Dict[str, str] = None):
    """
    發送雙軌策略結果到 Telegram, Discord, Email
    
    Args:
        index_name: 指數名稱
        results: run_dual_strategy() 返回的結果
    """
    calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 格式化訊息
    messages = format_dual_strategy_message(index_name, results, calculation_time, name_map=name_map)
    
    # 發送到各平台
    try:
        send_to_telegram(messages['telegram'])
        logger.info(f"✅ Telegram 發送成功 - {index_name}")
    except Exception as e:
        logger.error(f"❌ Telegram 發送失敗: {e}")
    
    try:
        send_to_discord(messages['discord'])
        logger.info(f"✅ Discord 發送成功 - {index_name}")
    except Exception as e:
        logger.error(f"❌ Discord 發送失敗: {e}")
    
    try:
        subject = f"雙軌策略投資建議 - {index_name} - {calculation_time}"
        send_email(subject, messages['email'], EmailConfig.TO_EMAILS)
        logger.info(f"✅ Email 發送成功 - {index_name}")
    except Exception as e:
        logger.error(f"❌ Email 發送失敗: {e}")
