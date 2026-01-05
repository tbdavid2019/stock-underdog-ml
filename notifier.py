"""
Notification services for stock prediction results
Supports Email, Telegram, and Discord
"""
import smtplib
import datetime
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any
from config import EmailConfig, TelegramConfig, DiscordConfig


def send_email(subject: str, body: str, to_emails: List[str]):
    """
    Send email notification
    
    Args:
        subject: Email subject line
        body: Email body content
        to_emails: List of recipient email addresses
    """
    msg = MIMEMultipart()
    msg['From'] = EmailConfig.SENDER_EMAIL
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP_SSL(EmailConfig.SMTP_SERVER, EmailConfig.SMTP_PORT)
    server.login(EmailConfig.SENDER_EMAIL, EmailConfig.EMAIL_PASSWORD)
    server.sendmail(EmailConfig.SENDER_EMAIL, to_emails, msg.as_string())
    server.quit()


def send_to_telegram(message: str):
    """
    Send message to Telegram channel
    
    Args:
        message: Message content (supports HTML formatting)
    """
    url = f"https://api.telegram.org/bot{TelegramConfig.BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TelegramConfig.CHANNEL_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        print(f"Telegram 發送失敗: {response.text}")


def send_to_discord(message: str):
    """
    Send message to Discord channel via webhook
    
    Args:
        message: Message content (supports Discord markdown)
    """
    try:
        payload = {"content": message}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            DiscordConfig.WEBHOOK_URL,
            json=payload,
            headers=headers
        )
        
        if response.status_code == 204:
            print("訊息已成功傳送到 Discord 頻道。")
        else:
            print(f"傳送訊息到 Discord 時發生錯誤: {response.status_code}, {response.text}")
    
    except Exception as e:
        print(f"傳送訊息到 Discord 時發生錯誤: {str(e)}")


def send_results(index_name: str, stock_predictions: Dict[str, List[Any]]):
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
    from database import save_to_mongodb
    
    calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"發送結果: {index_name}")
    
    # Save to MongoDB (optional feature)
    save_to_mongodb(index_name, stock_predictions)
    
    # Helper function to format table
    def format_table(predictions, display_count=5):
        """Format predictions as a table string"""
        table = f"{'股票':<8} {'現價':>10} {'預測價':>10} {'潛力':>8}\n"
        table += "-" * 40 + "\n"
        for stock, potential, current, predicted in predictions[:display_count]:
            table += f"{stock:<8} {current:>10.2f} {predicted:>10.2f} {potential*100:>7.2f}%\n"
        return table
    
    # ===== Email =====
    email_subject = f"每日潛力股 - {index_name} - {calculation_time}"
    email_body = f"運算日期和時間: {calculation_time}\n\n指數: {index_name}\n"
    
    for category, predictions in stock_predictions.items():
        email_body += f"\n{category}\n"
        email_body += format_table(predictions)
    
    send_email(email_subject, email_body, EmailConfig.TO_EMAILS)
    
    # ===== Telegram =====
    telegram_message = f"<b>每日潛力股分析</b>\n運算日期和時間: <b>{calculation_time}</b>\n\n指數: <b>{index_name}</b>\n"
    
    for category, predictions in stock_predictions.items():
        telegram_message += f"\n<b>{category}</b>\n"
        telegram_message += "<pre>\n"
        telegram_message += format_table(predictions)
        telegram_message += "</pre>\n"
    
    send_to_telegram(telegram_message)
    
    # ===== Discord =====
    discord_message = f"**每日潛力股分析**\n運算日期和時間: **{calculation_time}**\n\n指數: **{index_name}**\n"
    
    for category, predictions in stock_predictions.items():
        discord_message += f"\n**{category}**\n"
        discord_message += "```\n"
        discord_message += format_table(predictions)
        discord_message += "```\n"
    
    print("[DEBUG] discord_message 組裝內容：")
    print(discord_message)
    send_to_discord(discord_message)
