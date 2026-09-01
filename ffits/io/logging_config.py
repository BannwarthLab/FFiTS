"""
Centralized logging configuration for the FFiTS package.

Provides a uniform logging system with support for environment variables
and programmatic configuration of log levels.
"""

import logging
import sys
from typing import Optional

# Global logger instance
_LOGGER: Optional[logging.Logger] = None


def setup_logger(log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return the root logger for FFiTS.

    Log level defaults to INFO unless ``log_level`` is given.

    Args:
        log_level (str, optional): Log level as a string (e.g., 'DEBUG', 'INFO').

    Returns:
        logging.Logger: Configured logger instance for the 'ffits' package.

    Examples:
        >>> logger = setup_logger('DEBUG')  # Forces DEBUG level
    """
    global _LOGGER

    # Normalize log level string
    log_level = log_level.upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Get or create logger
    logger = logging.getLogger("ffits")
    logger.setLevel(numeric_level)

    # Only add handler if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        # Create console handler with formatted output
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(numeric_level)

        # Create formatter with format: [LEVEL] message
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        c_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(c_handler)

    _LOGGER = logger
    return logger
