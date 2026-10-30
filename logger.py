import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name='stock_app', log_file='logs/app.log', level=logging.INFO):
    """
    Setup a logger with console and file handlers.
    
    Args:
        name: Logger name (module name)
        log_file: Path to log file
        level: Logging level
    
    Returns:
        logging.Logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File Handler (Rotating) - Handle read-only or permission-constrained environments gracefully
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3) # 5MB per file
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except (OSError, PermissionError):
            pass

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

# Default logger instance
logger = setup_logger()
