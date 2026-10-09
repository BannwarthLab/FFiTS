"""Tests for ffits.io.commandline_parser (argument parsing/validation)."""
import logging

import pytest

from ffits.io.commandline_parser import _get_version, parse_args

from ffits.io.commandline_parser import parse_args


def test_parse_args_includes_timing_flag(monkeypatch, tmp_path):
    reactant = tmp_path / "reactant.xyz"
    product = tmp_path / "product.xyz"
    reactant.write_text("1\n\nH 0.0 0.0 0.0\n")
    product.write_text("1\n\nH 0.0 0.0 0.1\n")

    monkeypatch.setattr(
        "sys.argv",
        ["ffits", str(reactant), str(product), "--timing"],
    )

    args = parse_args()

    assert args["timing"] is True
    assert args["debug"] is False
    assert args["structures"] == [str(reactant), str(product)]
    
def test_get_version_returns_a_string():
    # Whatever pyproject.toml resolution finds (or "unknown" as a fallback),
    # this should never raise.
    version = _get_version()
    assert isinstance(version, str)
    assert version != ""


def test_version_flag_prints_version_and_exits(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ffits", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        parse_args()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "FFiTS version:" in out


def test_short_version_flag_also_exits(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ffits", "-v"])
    with pytest.raises(SystemExit):
        parse_args()


def test_tsguess_mode_requires_exactly_two_structures(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    monkeypatch.setattr("sys.argv", ["ffits", str(reac)])
    with pytest.raises(SystemExit):
        parse_args()


def test_optimizer_mode_requires_exactly_one_structure(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    prod = tmp_path / "prod.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    prod.write_text("1\n\nC 0.0 0.0 0.0\n")
    monkeypatch.setattr(
        "sys.argv", ["ffits", str(reac), str(prod), "--opt", "ff.csv"]
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_reaction_path_mode_requires_exactly_two_structures(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    monkeypatch.setattr("sys.argv", ["ffits", str(reac), "--reaction_path", "5"])
    with pytest.raises(SystemExit):
        parse_args()


def test_missing_structure_file_raises_file_not_found(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    missing = tmp_path / "does_not_exist.xyz"
    monkeypatch.setattr("sys.argv", ["ffits", str(reac), str(missing)])
    with pytest.raises(FileNotFoundError, match="Structure file not found"):
        parse_args()


def test_missing_config_file_raises_file_not_found(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    prod = tmp_path / "prod.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    prod.write_text("1\n\nC 0.0 0.0 0.0\n")
    missing_config = tmp_path / "no_such_config.toml"
    monkeypatch.setattr(
        "sys.argv",
        ["ffits", str(reac), str(prod), "--config", str(missing_config)],
    )
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        parse_args()


def test_valid_tsguess_args_are_parsed_and_returned(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    prod = tmp_path / "prod.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    prod.write_text("1\n\nC 0.0 0.0 0.0\n")
    monkeypatch.setattr(
        "sys.argv",
        ["ffits", str(reac), str(prod), "--charge", "1", "--multiplicity", "2"],
    )
    result = parse_args()
    assert result["structures"] == [str(reac), str(prod)]
    assert result["charge"] == 1
    assert result["multiplicity"] == 2
    assert result["config_file"] is None
    assert result["parameterize"] is False
    assert result["reaction_path"] is None
    assert result["optff"] is None
    assert result["debug"] is False


def test_debug_and_parameterize_flags_are_parsed(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    prod = tmp_path / "prod.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    prod.write_text("1\n\nC 0.0 0.0 0.0\n")
    monkeypatch.setattr(
        "sys.argv", ["ffits", str(reac), str(prod), "--parameterize", "--debug"]
    )
    result = parse_args()
    assert result["parameterize"] is True
    assert result["debug"] is True


def test_config_with_multiplicity_logs_warning(monkeypatch, tmp_path, caplog):
    reac = tmp_path / "reac.xyz"
    prod = tmp_path / "prod.xyz"
    config = tmp_path / "config.toml"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    prod.write_text("1\n\nC 0.0 0.0 0.0\n")
    config.write_text("")
    monkeypatch.setattr(
        "sys.argv",
        [
            "ffits",
            str(reac),
            str(prod),
            "--config",
            str(config),
            "--multiplicity",
            "2",
        ],
    )
    with caplog.at_level(logging.WARNING):
        parse_args()
    assert any("Multiplicity" in record.message for record in caplog.records)


def test_config_with_charge_logs_warning(monkeypatch, tmp_path, caplog):
    reac = tmp_path / "reac.xyz"
    prod = tmp_path / "prod.xyz"
    config = tmp_path / "config.toml"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    prod.write_text("1\n\nC 0.0 0.0 0.0\n")
    config.write_text("")
    monkeypatch.setattr(
        "sys.argv",
        ["ffits", str(reac), str(prod), "--config", str(config), "--charge", "-1"],
    )
    with caplog.at_level(logging.WARNING):
        parse_args()
    assert any("Charge" in record.message for record in caplog.records)


def test_optimizer_mode_with_one_structure_succeeds(monkeypatch, tmp_path):
    reac = tmp_path / "reac.xyz"
    reac.write_text("1\n\nC 0.0 0.0 0.0\n")
    monkeypatch.setattr(
        "sys.argv", ["ffits", str(reac), "--opt", "ff.csv"]
    )
    result = parse_args()
    assert result["structures"] == [str(reac)]
    assert result["optff"] == "ff.csv"
