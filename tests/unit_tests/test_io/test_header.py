"""Tests for ffits.io.print.header (program banner printed at startup)."""
import subprocess

from ffits.io.print import header as header_mod
from ffits.io.print.header import (
    print_program_header,
    _get_build_date,
    _get_last_commit_date,
    _load_pyproject_metadata,
)


def test_print_program_header_smoke(capsys):
    """The header should print without raising and mention the program name."""
    print_program_header()
    out = capsys.readouterr().out
    assert "FFiTS" in out
    assert "Force Field-interpolated Transition States" in out
    assert "Version:" in out
    assert "License:" in out


def test_print_program_header_uses_given_version(capsys):
    print_program_header(version="9.9.9")
    out = capsys.readouterr().out
    assert "Version:     9.9.9" in out


def test_load_pyproject_metadata_has_expected_keys():
    metadata = _load_pyproject_metadata()
    assert set(metadata) == {"license", "authors", "version"}
    assert metadata["license"] == "MIT"
    assert metadata["authors"] == "Daria Babushkina"


def test_get_build_date_returns_parseable_timestamp():
    import datetime

    build_date = _get_build_date()
    # Should not raise, and should be "YYYY-MM-DD HH:MM"
    datetime.datetime.strptime(build_date, "%Y-%m-%d %H:%M")


def test_get_build_date_falls_back_to_now_on_error(monkeypatch):
    def boom(_path):
        raise OSError("no such package dir")

    monkeypatch.setattr(header_mod.os.path, "getmtime", boom)
    # Should not raise even though getmtime fails.
    build_date = _get_build_date()
    assert isinstance(build_date, str)
    assert len(build_date) == 16  # "YYYY-MM-DD HH:MM"


def test_get_last_commit_date_returns_none_when_git_unavailable(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(header_mod.subprocess, "run", fake_run)
    assert _get_last_commit_date() is None


def test_get_last_commit_date_returns_none_on_nonzero_returncode(monkeypatch):
    class FakeResult:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(
        header_mod.subprocess, "run", lambda *a, **k: FakeResult()
    )
    assert _get_last_commit_date() is None


def test_get_last_commit_date_parses_git_output(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = "2024-02-09 10:30:45 +0100\n"

    monkeypatch.setattr(
        header_mod.subprocess, "run", lambda *a, **k: FakeResult()
    )
    assert _get_last_commit_date() == "2024-02-09 10:30"


def test_print_program_header_omits_last_commit_when_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(header_mod, "_get_last_commit_date", lambda: None)
    print_program_header()
    out = capsys.readouterr().out
    assert "Last Commit:" not in out


def test_print_program_header_includes_last_commit_when_available(
    monkeypatch, capsys
):
    monkeypatch.setattr(header_mod, "_get_last_commit_date", lambda: "2024-02-09 10:30")
    print_program_header()
    out = capsys.readouterr().out
    assert "Last Commit: 2024-02-09 10:30" in out
