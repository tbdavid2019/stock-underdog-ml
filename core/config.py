"""
Centralized configuration management for stock prediction application.
Loads environment variables and provides structured access to settings.
"""
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict

# Load .env file
load_dotenv()


class DatabaseConfig:
    """Supabase database configuration settings"""
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")


class EmailConfig:
    """Email and SMTP configuration settings"""
    SMTP_SERVER: Optional[str] = os.getenv("SMTP_SERVER")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))
    SENDER_EMAIL: Optional[str] = os.getenv("SENDER_EMAIL")
    EMAIL_PASSWORD: Optional[str] = os.getenv("EMAIL_PASSWORD")
    TO_EMAILS: List[str] = os.getenv("TO_EMAILS", "").split(",") if os.getenv("TO_EMAILS") else []


class TelegramConfig:
    """Telegram bot configuration settings"""
    BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    CHANNEL_ID: Optional[str] = os.getenv("TELEGRAM_CHANNEL_ID")


class DiscordConfig:
    """Discord webhook configuration settings"""
    WEBHOOK_URL: Optional[str] = os.getenv("DISCORD_WEBHOOK_URL")


class DataAPIConfig:
    """Stock index and market data API endpoints"""
    BASE_URL: str = "https://answerbook.david888.com"
    TW0050_URL: str = f"{BASE_URL}/TW0050"
    TW0051_URL: str = f"{BASE_URL}/TW0051"
    SP500_URL: str = f"{BASE_URL}/SP500"
    NASDAQ100_URL: str = f"{BASE_URL}/nasdaq100"
    DOWJONES_URL: str = f"{BASE_URL}/dowjones"


class PipelineConfig:
    """Pipeline and Strategy execution settings"""
    DEFAULT_PERIOD: str = "6mo"
    CACHE_MAX_AGE_HOURS: int = 12
    INDEX_CACHE_MAX_AGE_DAYS: int = 90
    DEFAULT_DEVICE: str = os.getenv("DEVICE", "auto")
    ENABLED_STRATEGIES: List[str] = ["xuantie", "lstm"]
    STRATEGY_WEIGHTS: Dict[str, float] = {
        "xuantie": 0.4,
        "lstm": 0.4,
        "fundamental": 0.2
    }


class Config:
    """Main configuration class providing structured access to all sections"""
    db = DatabaseConfig
    api = DataAPIConfig
    email = EmailConfig
    telegram = TelegramConfig
    discord = DiscordConfig
    pipeline = PipelineConfig

    # Convenience properties for backward compatibility
    @property
    def supabase_url(self) -> Optional[str]:
        return DatabaseConfig.SUPABASE_URL

    @property
    def supabase_key(self) -> Optional[str]:
        return DatabaseConfig.SUPABASE_SERVICE_KEY or DatabaseConfig.SUPABASE_KEY

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


# Singleton instance
config = Config()
