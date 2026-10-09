"""
Centralized logging configuration for the FFiTS package.

Provides a uniform logging system with support for environment variables
and programmatic configuration of log levels.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

# Global logger instance
_LOGGER: Optional[logging.Logger] = None
TIMING_LEVEL = 5


def _record_timing(
    timing_entries: list[str],
    timing_logger: logging.Logger,
    label: str,
    start: float,
    end: float,
) -> None:
    elapsed_seconds = end - start
    entry = f"{label}: {elapsed_seconds:.3f}s"
    timing_entries.append(entry)
    timing_logger.timing(entry)


def _write_timing_report(
    timing_entries: list[str],
    total_elapsed: float,
    success: bool,
    output_path: Path = Path("ffits_timing.log"),
) -> None:
    report_lines = [
        "FFiTS timing report",
        "=" * 70,
        *timing_entries,
        f"Total time: {total_elapsed:.3f}s",
        f"Status: {'COMPLETED' if success else 'FAILED'}",
        "=" * 70,
        "",
    ]
    output_path.write_text("\n".join(report_lines))


def _register_timing_level() -> None:
    """Register the custom TIMING log level once."""
    logging.addLevelName(TIMING_LEVEL, "TIMING")

    def timing(self: logging.Logger, message, *args, **kwargs):
        if self.isEnabledFor(TIMING_LEVEL):
            self._log(TIMING_LEVEL, message, args, **kwargs)

    if not hasattr(logging.Logger, "timing"):
        setattr(logging.Logger, "timing", timing)


_register_timing_level()


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
    log_level_name = log_level.upper() if log_level else "INFO"
    if log_level_name == "TIMING":
        numeric_level = TIMING_LEVEL
    else:
        numeric_level = getattr(logging, log_level_name, logging.INFO)

    # Get or create logger
    logger = logging.getLogger("ffits")
    logger.setLevel(numeric_level)

    # Only add handler if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        # Create console handler with formatted output
        c_handler = logging.StreamHandler(sys.stdout)

        # Create formatter with format: [LEVEL] message
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        c_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(c_handler)

    for handler in logger.handlers:
        handler.setLevel(numeric_level)

    _LOGGER = logger
    return logger
