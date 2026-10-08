"""
雙軌與多維量化策略專用通知模組
支援 Telegram, Discord, Email，包含美股宏觀風控、AI 操盤解讀與多維共振結構化卡片
"""
import datetime
from typing import Dict, Optional, Any
from notifier import send_to_telegram, send_to_discord, send_email
from config import EmailConfig
from logger import logger
from evaluators.formatter import (
    format_dual_strategy_message,
    format_telegram_message,
    format_discord_message,
    format_email_message,
    format_ticker_label
)


def send_dual_strategy_results(
    index_name: str, 
    results: dict, 
    name_map: Dict[str, str] = None,
    macro_state: Optional[Any] = None,
    ai_summary: str = ""
):
    """
    發送量化分析結果到 Telegram, Discord, Email
    """
    calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 格式化訊息 (調用高級排版引擎)
    messages = format_dual_strategy_message(
        index_name, 
        results, 
        calculation_time, 
        name_map=name_map,
        macro_state=macro_state,
        ai_summary=ai_summary
    )
    
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
        subject = f"量化策略投資日報 - {index_name} - {calculation_time}"
        send_email(subject, messages['email'], EmailConfig.TO_EMAILS)
        logger.info(f"✅ Email 發送成功 - {index_name}")
    except Exception as e:
        logger.error(f"❌ Email 發送失敗: {e}")
