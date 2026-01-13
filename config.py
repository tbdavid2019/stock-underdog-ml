"""
Configuration management for stock prediction application.
Loads environment variables and provides centralized configuration access.
"""
import os
from dotenv import load_dotenv
from typing import Optional, List

# Load .env file
load_dotenv()


class DatabaseConfig:
    """Database configuration settings"""
    
    # MySQL Configuration
    USE_MYSQL = os.getenv("USE_MYSQL", "false").lower() == "true"
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = "stock_predictions"
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")


class EmailConfig:
    """Email and SMTP configuration settings"""
    
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    TO_EMAILS = os.getenv("TO_EMAILS", "").split(",") if os.getenv("TO_EMAILS") else []


class TelegramConfig:
    """Telegram bot configuration settings"""
    
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")


class DiscordConfig:
    """Discord webhook configuration settings"""
    
    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


class ModelConfig:
    """Model feature flags and configuration"""
    
    # Only LSTM model is supported
    pass


class DataAPIConfig:
    """Stock index API endpoints"""
    BASE_URL = "https://answerbook.david888.com"
    TW0050_URL = f"{BASE_URL}/TW0050"
    TW0051_URL = f"{BASE_URL}/TW0051"
    SP500_URL = f"{BASE_URL}/SP500"
    NASDAQ100_URL = f"{BASE_URL}/nasdaq100"
    DOWJONES_URL = f"{BASE_URL}/dowjones"


class Config:
    """Main configuration class - provides easy access to all config sections"""
    
    db = DatabaseConfig
    api = DataAPIConfig
    email = EmailConfig
    telegram = TelegramConfig
    discord = DiscordConfig
    model = ModelConfig
    
    # Legacy compatibility - direct access to common settings
    @property
    def use_mysql(self) -> bool:
        return DatabaseConfig.USE_MYSQL
    
    @property
    def smtp_server(self) -> Optional[str]:
        return EmailConfig.SMTP_SERVER
    
    @property
    def smtp_port(self) -> int:
        return EmailConfig.SMTP_PORT
    
    @property
    def sender_email(self) -> Optional[str]:
        return EmailConfig.SENDER_EMAIL
    
    @property
    def email_password(self) -> Optional[str]:
        return EmailConfig.EMAIL_PASSWORD
    
    @property
    def to_emails(self) -> List[str]:
        return EmailConfig.TO_EMAILS
    
    @property
    def telegram_bot_token(self) -> Optional[str]:
        return TelegramConfig.BOT_TOKEN
    
    @property
    def telegram_channel_id(self) -> Optional[str]:
        return TelegramConfig.CHANNEL_ID
    
    @property
    def discord_webhook_url(self) -> Optional[str]:
        return DiscordConfig.WEBHOOK_URL
    
    @property
    def mongo_uri(self) -> Optional[str]:
        return DatabaseConfig.MONGO_URI
    
    @property
    def db_name(self) -> str:
        return DatabaseConfig.MONGO_DB_NAME
        
    @property
    def supabase_url(self) -> Optional[str]:
        return DatabaseConfig.SUPABASE_URL
        
    @property
    def supabase_key(self) -> Optional[str]:
        return DatabaseConfig.SUPABASE_KEY


# Create singleton instance
config = Config()
