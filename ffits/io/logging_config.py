"""
Centralized logging configuration for the FFiTS package.

Provides a uniform logging system with support for environment variables
and programmatic configuration of log levels.
"""

import logging
import os
import sys
from typing import Optional


# Global logger instance
_LOGGER: Optional[logging.Logger] = None


def setup_logger(log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return the root logger for FFiTS.

    Log level is determined by (in order of precedence):
    1. log_level parameter (if provided)
    2. Default to INFO

    Parameters
        log_level (str, optional): Log level as a string (e.g., 'DEBUG', 'INFO'). Overrides environment variable if provided.

    Returns
        logging.Logger: Configured logger instance for the 'ffits' package.

    Examples
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


# def get_logger(module_name: str) -> logging.Logger:
#     """
#     Get a logger for a specific module.

#     Use this function in individual modules to get a logger with the
#     module name for better organization and debugging.

#     Parameters
#     ----------
#     module_name : str
#         The module name, typically __name__

#     Returns
#     -------
#     logging.Logger
#         Logger instance for the specified module.

#     Examples
#     --------
#     >>> logger = get_logger(__name__)
#     >>> logger.debug("Debug message")
#     """
#     global _LOGGER

#     # Ensure root logger is set up
#     if _LOGGER is None:
#         setup_logger()

#     return logging.getLogger(module_name)


# def set_log_level(log_level: str) -> None:
#     """
#     Change the log level after logger initialization.

#     Parameters
#     ----------
#     log_level : str
#         Log level as string: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

#     Examples
#     --------
#     >>> set_log_level('DEBUG')
#     """
#     log_level = log_level.upper()
#     numeric_level = getattr(logging, log_level, logging.INFO)

#     root_logger = logging.getLogger("ffits")
#     root_logger.setLevel(numeric_level)

#     # Update handler level
#     for handler in root_logger.handlers:
#         handler.setLevel(numeric_level)
