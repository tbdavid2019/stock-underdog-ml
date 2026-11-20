"""
Notification services for stock prediction results
Supports Email, Telegram, and Discord
"""
import smtplib
import datetime
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any, Optional
from core.config import EmailConfig, TelegramConfig, DiscordConfig


def send_email(subject: str, body: str, to_emails: List[str], html_body: Optional[str] = None):
    """
    Send email notification

    Args:
        subject: Email subject line
        body: Email body content (plain text fallback)
        to_emails: List of recipient email addresses
        html_body: Optional responsive HTML body content
    """
    if html_body:
        msg = MIMEMultipart('alternative')
        msg['From'] = EmailConfig.SENDER_EMAIL
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    else:
        msg = MIMEMultipart()
        msg['From'] = EmailConfig.SENDER_EMAIL
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

    with smtplib.SMTP_SSL(EmailConfig.SMTP_SERVER, EmailConfig.SMTP_PORT, timeout=15) as server:
        server.login(EmailConfig.SENDER_EMAIL, EmailConfig.EMAIL_PASSWORD)
        server.sendmail(EmailConfig.SENDER_EMAIL, to_emails, msg.as_string())


def send_to_telegram(message: str):
    """
    Send message to Telegram channel (automatically chunks messages exceeding 4000 characters)

    Args:
        message: Message content (supports HTML formatting)
    """
    if not TelegramConfig.BOT_TOKEN or not TelegramConfig.CHANNEL_ID:
        raise RuntimeError("Telegram BOT_TOKEN 或 CHANNEL_ID/CHAT_ID 未設定")

    try:
        from evaluators.formatter import split_telegram_message
        chunks = split_telegram_message(message, max_length=4000)
    except Exception:
        chunks = [message]

    for chunk in chunks:
        url = f"https://api.telegram.org/bot{TelegramConfig.BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TelegramConfig.CHANNEL_ID,
            "text": chunk,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(url, json=payload, timeout=(5, 15))
        except requests.RequestException as exc:
            # Never propagate raw RequestException (it contains the bot token URL)
            raise RuntimeError(f"Telegram 連線發送失敗 ({type(exc).__name__})") from None

        if response.status_code != 200:
            raise RuntimeError(f"Telegram 發送失敗 (HTTP {response.status_code}): {response.text[:300]}")


def send_to_discord(message: str):
    """
    Send message to Discord channel via webhook as a text file

    Args:
        message: Message content
    """
    if not DiscordConfig.WEBHOOK_URL:
        raise RuntimeError("Discord WEBHOOK_URL 未設定")

    import io
    file_data = io.BytesIO(message.encode('utf-8'))
    files = {
        'file': ('prediction_results.txt', file_data, 'text/plain')
    }
    payload = {"content": "📊 每日潛力股分析結果已生成，詳見附件。"}

    try:
        response = requests.post(
            DiscordConfig.WEBHOOK_URL,
            data=payload,
            files=files,
            timeout=(5, 15)
        )
    except requests.RequestException as exc:
        # Never propagate raw RequestException (it contains the Discord webhook token URL)
        raise RuntimeError(f"Discord 連線發送失敗 ({type(exc).__name__})") from None

    if response.status_code in [200, 204]:
        print("訊息及檔案已成功傳送到 Discord 頻道。")
    else:
        raise RuntimeError(f"傳送檔案到 Discord 時發生錯誤 (HTTP {response.status_code}): {response.text[:300]}")


def send_results(index_name: str, stock_predictions: Dict[str, List[Any]], name_map: Dict[str, str] = None):
    """
    Send stock prediction results via all configured notification channels

    Args:
        index_name: Name of the stock index (e.g., "SP500")
        stock_predictions: Dictionary of predictions with format:
            {
                "🥇 前十名 LSTM 🧠": [(ticker, potential, current_price, predicted_price), ...],
                ...
            }
    """
    calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"發送結果: {index_name}")

    # Helper function to format table
    def format_table(predictions, display_count=5):
        """Format predictions as a table string"""
        lookup = name_map or {}
        label_width = 16 if lookup else 8
        label_header = "代碼/名稱" if lookup else "股票"
        table = f"{label_header:<{label_width}} {'現價':>10} {'預測價':>10} {'潛力':>8}\n"
        table += "-" * 40 + "\n"
        for stock, potential, current, predicted in predictions[:display_count]:
            name = lookup.get(stock)
            label = f"{stock} {name}" if name else stock
            table += f"{label:<{label_width}} {current:>10.2f} {predicted:>10.2f} {potential*100:>7.2f}%\n"
        return table

    # ===== Email =====
    email_subject = f"每日潛力股 - {index_name} - {calculation_time}"
    email_body = f"運算日期和時間: {calculation_time}\n\n指數: {index_name}\n"

    for category, predictions in stock_predictions.items():
        email_body += f"\n{category}\n"
        email_body += format_table(predictions)

    errors = []
    try:
        send_email(email_subject, email_body, EmailConfig.TO_EMAILS)
    except Exception as exc:
        errors.append(f"Email: {exc}")

    # ===== Telegram =====
    telegram_message = f"<b>每日潛力股分析</b>\n運算日期和時間: <b>{calculation_time}</b>\n\n指數: <b>{index_name}</b>\n"

    for category, predictions in stock_predictions.items():
        telegram_message += f"\n<b>{category}</b>\n"
        telegram_message += "<pre>\n"
        telegram_message += format_table(predictions)
        telegram_message += "</pre>\n"

    try:
        send_to_telegram(telegram_message)
    except Exception as exc:
        errors.append(f"Telegram: {exc}")

    # ===== Discord =====
    discord_message = f"**每日潛力股分析**\n運算日期和時間: **{calculation_time}**\n\n指數: **{index_name}**\n"

    for category, predictions in stock_predictions.items():
        discord_message += f"\n**{category}**\n"
        discord_message += "```\n"
        discord_message += format_table(predictions)
        discord_message += "```\n"

    try:
        send_to_discord(discord_message)
    except Exception as exc:
        errors.append(f"Discord: {exc}")

    if errors:
        raise RuntimeError("部分推播管道發送失敗: " + "; ".join(errors))
