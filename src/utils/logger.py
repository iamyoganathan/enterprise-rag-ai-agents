"""
Logging configuration module using Loguru.
Provides structured logging with rotation and formatting.
"""

import sys
from pathlib import Path
from loguru import logger
from .config import get_settings


def setup_logger():
    """
    Configure the logger with file and console handlers.
    Uses Loguru for enhanced logging capabilities.
    """
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # File handler with rotation
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        enqueue=True  # Thread-safe logging
    )
    
    logger.info("Logger initialized successfully")
    return logger


# Initialize logger on import
log = setup_logger()


def get_logger(name: str = None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message."""
    logger.info(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log error message."""
    logger.error(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message."""
    logger.warning(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message."""
    logger.debug(message, **kwargs)


def log_exception(exception: Exception, message: str = None):
    """Log exception with traceback."""
    if message:
        logger.exception(f"{message}: {exception}")
    else:
        logger.exception(exception)


if __name__ == "__main__":
    # Test logging
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.debug("This is a debug message")
    
    try:
        1 / 0
    except Exception as e:
        log_exception(e, "Division by zero error")
