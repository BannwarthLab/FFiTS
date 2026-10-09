"""Tests for ffits.io.print.summary.print_run_summary."""

import re

from ffits.io.print.summary import print_run_summary


def test_prints_completed_status_on_success(capsys):
    print_run_summary(start_time=1000.0, end_time=1005.0, success=True)
    out = capsys.readouterr().out
    assert "Calculation COMPLETED" in out
    assert "Calculation FAILED" not in out


def test_prints_failed_status_on_failure(capsys):
    print_run_summary(start_time=1000.0, end_time=1005.0, success=False)
    out = capsys.readouterr().out
    assert "Calculation FAILED" in out
    assert "Calculation COMPLETED" not in out


def test_message_included_when_given(capsys):
    print_run_summary(
        start_time=1000.0, end_time=1005.0, success=False, message="xtb crashed"
    )
    out = capsys.readouterr().out
    assert "Message:      xtb crashed" in out


def test_message_omitted_when_not_given(capsys):
    print_run_summary(start_time=1000.0, end_time=1005.0, success=True)
    out = capsys.readouterr().out
    assert "Message:" not in out


def test_seconds_only_formatting_under_a_minute(capsys):
    print_run_summary(start_time=1000.0, end_time=1042.0)
    out = capsys.readouterr().out
    assert "Total Time:   42s" in out


def test_minutes_and_seconds_formatting(capsys):
    print_run_summary(start_time=1000.0, end_time=1000.0 + 5 * 60 + 3)
    out = capsys.readouterr().out
    assert "Total Time:   5m 3s" in out


def test_hours_minutes_and_seconds_formatting(capsys):
    print_run_summary(start_time=0.0, end_time=2 * 3600 + 5 * 60 + 7)
    out = capsys.readouterr().out
    assert "Total Time:   2h 5m 7s" in out


def test_end_time_defaults_to_now(capsys):
    # Should not raise, and should report a small non-negative elapsed time.
    import time

    start = time.time()
    print_run_summary(start_time=start, end_time=None)
    out = capsys.readouterr().out
    match = re.search(r"Total Time:\s+(\d+)s", out)
    assert match is not None
    assert int(match.group(1)) >= 0
