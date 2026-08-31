"""
Legacy compatibility wrapper for config.
Redirects to core.config.
"""
from core.config import (
    Config,
    config,
    DatabaseConfig,
    DataAPIConfig,
    EmailConfig,
    TelegramConfig,
    DiscordConfig,
    PipelineConfig
)

__all__ = [
    "Config",
    "config",
    "DatabaseConfig",
    "DataAPIConfig",
    "EmailConfig",
    "TelegramConfig",
    "DiscordConfig",
    "PipelineConfig"
]
