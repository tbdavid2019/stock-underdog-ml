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
    }


class LLMConfig:
    """3-Tier Fallback LLM configuration settings"""
    ENABLE_LLM_SUMMARY: bool = os.getenv("ENABLE_LLM_SUMMARY", "true").lower() in ("true", "1", "yes")
    TIMEOUT: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
    
    PRIMARY_NAME: str = os.getenv("LLM_PRIMARY_NAME", "Gemini-Flash")
    PRIMARY_BASE_URL: str = os.getenv("LLM_PRIMARY_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    PRIMARY_MODEL: str = os.getenv("LLM_PRIMARY_MODEL", "gemini-2.5-flash")
    PRIMARY_API_KEY: Optional[str] = os.getenv("LLM_PRIMARY_API_KEY") or os.getenv("GEMINI_API_KEY")

    FALLBACK1_NAME: str = os.getenv("LLM_FALLBACK1_NAME", "DeepSeek-V3")
    FALLBACK1_BASE_URL: str = os.getenv("LLM_FALLBACK1_BASE_URL", "https://api.deepseek.com/v1")
    FALLBACK1_MODEL: str = os.getenv("LLM_FALLBACK1_MODEL", "deepseek-chat")
    FALLBACK1_API_KEY: Optional[str] = os.getenv("LLM_FALLBACK1_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    FALLBACK2_NAME: str = os.getenv("LLM_FALLBACK2_NAME", "OpenAI-Mini")
    FALLBACK2_BASE_URL: str = os.getenv("LLM_FALLBACK2_BASE_URL", "https://api.openai.com/v1")
    FALLBACK2_MODEL: str = os.getenv("LLM_FALLBACK2_MODEL", "gpt-4o-mini")
    FALLBACK2_API_KEY: Optional[str] = os.getenv("LLM_FALLBACK2_API_KEY") or os.getenv("OPENAI_API_KEY")


class SectorConfig:
    """7 大核心產業板塊分類與個股映射"""
    SECTOR_NAMES: List[str] = [
        "半導體與IC設計",
        "AI伺服器與電子組裝",
        "金融保險",
        "重電與基礎建設",
        "航運與物流",
        "傳統產業與化學",
        "生技醫療"
    ]

    TICKER_SECTOR_MAP: Dict[str, str] = {
        # 1. 半導體
        "2330.TW": "半導體與IC設計", "2454.TW": "半導體與IC設計", "2303.TW": "半導體與IC設計",
        "3711.TW": "半導體與IC設計", "2449.TW": "半導體與IC設計", "3661.TW": "半導體與IC設計",
        "6415.TW": "半導體與IC設計", "2379.TW": "半導體與IC設計", "3034.TW": "半導體與IC設計",
        "NVDA": "半導體與IC設計", "TSM": "半導體與IC設計", "AVGO": "半導體與IC設計",
        "AMD": "半導體與IC設計", "QCOM": "半導體與IC設計", "INTC": "半導體與IC設計",
        "TXN": "半導體與IC設計", "AMAT": "半導體與IC設計", "MU": "半導體與IC設計",
        # 2. AI 伺服器與電子
        "2317.TW": "AI伺服器與電子組裝", "2382.TW": "AI伺服器與電子組裝", "2308.TW": "AI伺服器與電子組裝",
        "3231.TW": "AI伺服器與電子組裝", "6669.TW": "AI伺服器與電子組裝", "2376.TW": "AI伺服器與電子組裝",
        "2356.TW": "AI伺服器與電子組裝", "2383.TW": "AI伺服器與電子組裝", "2327.TW": "AI伺服器與電子組裝",
        "AAPL": "AI伺服器與電子組裝", "MSFT": "AI伺服器與電子組裝", "GOOGL": "AI伺服器與電子組裝",
        "AMZN": "AI伺服器與電子組裝", "META": "AI伺服器與電子組裝", "DELL": "AI伺服器與電子組裝",
        # 3. 金融保險
        "2881.TW": "金融保險", "2882.TW": "金融保險", "2886.TW": "金融保險", "2891.TW": "金融保險",
        "2884.TW": "金融保險", "2885.TW": "金融保險", "2892.TW": "金融保險", "2880.TW": "金融保險",
        "JPM": "金融保險", "BAC": "金融保險", "WFC": "金融保險", "GS": "金融保險", "MS": "金融保險", "COF": "金融保險",
        # 4. 重電與基礎建設
        "1519.TW": "重電與基礎建設", "1513.TW": "重電與基礎建設", "1503.TW": "重電與基礎建設",
        "1514.TW": "重電與基礎建設", "2345.TW": "重電與基礎建設", "GE": "重電與基礎建設", "CAT": "重電與基礎建設",
        # 5. 航運與物流
        "2603.TW": "航運與物流", "2609.TW": "航運與物流", "2615.TW": "航運與物流", "2618.TW": "航運與物流",
        "UNP": "航運與物流", "UPS": "航運與物流", "FDX": "航運與物流",
        # 6. 傳統產業與化學
        "1101.TW": "傳統產業與化學", "1102.TW": "傳統產業與化學", "1301.TW": "傳統產業與化學",
        "1303.TW": "傳統產業與化學", "2002.TW": "傳統產業與化學", "1216.TW": "傳統產業與化學",
        "9945.TW": "傳統產業與化學", "XOM": "傳統產業與化學", "CVX": "傳統產業與化學",
        # 7. 生技與太空概念
        "6472.TW": "生技醫療", "4147.TW": "生技醫療", "4137.TW": "生技醫療",
        "LLY": "生技醫療", "JNJ": "生技醫療", "PFE": "生技醫療", "ABBV": "生技醫療",
        "SPCX": "太空與航太防衛", "DXYZ": "太空與航太防衛", "ASTS": "太空與航太防衛", "RKLB": "太空與航太防衛",
        "RTX": "太空與航太防衛", "LMT": "太空與航太防衛", "BA": "太空與航太防衛"
    }

    @classmethod
    def get_sector(cls, ticker: str) -> str:
        """取得標的對應板塊，未知者歸類為其他綜合板塊"""
        if ticker in cls.TICKER_SECTOR_MAP:
            return cls.TICKER_SECTOR_MAP[ticker]
        if ticker.startswith("28"):
            return "金融保險"
        elif ticker.startswith("26"):
            return "航運與物流"
        elif ticker.startswith("15"):
            return "重電與基礎建設"
        elif ticker.startswith("23") or ticker.startswith("24") or ticker.startswith("30"):
            return "電子與科技綜合"
        return "綜合產業"



class Config:
    """Main configuration class providing structured access to all sections"""
    db = DatabaseConfig
    api = DataAPIConfig
    email = EmailConfig
    telegram = TelegramConfig
    discord = DiscordConfig
    pipeline = PipelineConfig
    llm = LLMConfig
    sector = SectorConfig

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
