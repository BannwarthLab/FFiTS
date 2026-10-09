"""Tests for ffits.io.logging_config.setup_logger."""
import logging

import pytest

from ffits.io.logging_config import setup_logger

from ffits.io.logging_config import TIMING_LEVEL, setup_logger


def test_setup_logger_supports_timing_level(caplog):
    logger = setup_logger("TIMING")

    assert logger.level == TIMING_LEVEL
    assert hasattr(logger, "timing")

    with caplog.at_level(TIMING_LEVEL, logger="ffits"):
        logger.timing("timing message")

    assert any(
        record.levelname == "TIMING" and record.message == "timing message"
        for record in caplog.records
    )


def test_setup_logger_defaults_to_info_when_level_is_missing():
    logger = setup_logger(None)

    assert logger.level == 20
    
@pytest.fixture(autouse=True)
def _reset_ffits_logger():
    """setup_logger mutates the shared 'ffits' logger; reset it around each
    test so tests don't leak handlers/levels into each other."""
    logger = logging.getLogger("ffits")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    logger.handlers = []
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)


def test_setup_logger_returns_the_ffits_logger():
    logger = setup_logger("INFO")
    assert logger.name == "ffits"


def test_setup_logger_defaults_to_info_level_name_casing():
    logger = setup_logger("info")
    assert logger.level == logging.INFO


def test_setup_logger_sets_debug_level():
    logger = setup_logger("DEBUG")
    assert logger.level == logging.DEBUG


def test_setup_logger_unknown_level_falls_back_to_info():
    logger = setup_logger("NOT_A_REAL_LEVEL")
    assert logger.level == logging.INFO


def test_setup_logger_adds_exactly_one_handler():
    logger = setup_logger("INFO")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_does_not_duplicate_handlers_on_repeated_calls():
    setup_logger("INFO")
    setup_logger("DEBUG")
    logger = setup_logger("WARNING")
    assert len(logger.handlers) == 1


def test_setup_logger_handler_uses_expected_format():
    logger = setup_logger("INFO")
    formatter = logger.handlers[0].formatter
    assert formatter._fmt == "[%(levelname)s] %(message)s"
