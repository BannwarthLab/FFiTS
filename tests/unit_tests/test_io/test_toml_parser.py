from ffits.io.toml_parser import (
    overwrite_from_commandline,
)
import pytest
import os
import tempfile
import logging
from ffits.datatype.calculation_data import CalculationData, load_toml


class TestLoadToml:
    """Tests for load_toml function."""

    def test_load_toml_valid_file(self):
        """Test loading a valid TOML file."""
        path = os.path.join(os.getcwd(), "tests", "examples", "default_config.toml")
        config = load_toml(path)
        assert isinstance(config, dict)
        assert "system" in config

    def test_load_toml_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_toml("/nonexistent/path/to/file.toml")

    def test_load_toml_invalid_toml(self):
        """Test that invalid TOML raises an error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [[[")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(
                Exception
            ):  # tomllib raises various exceptions for invalid TOML
                load_toml(temp_path)
        finally:
            os.unlink(temp_path)


class TestOverwriteFromCommandline:
    """Tests for overwrite_from_commandline function."""

    def test_overwrite_both_multiplicity_and_charge(self, caplog):
        """Test overwriting both multiplicity and charge."""
        cd = CalculationData()
        caplog.set_level(logging.INFO)
        original_charge = cd.system.charge
        original_mult = cd.system.multiplicity

        overwrite_from_commandline(cd, multiplicity=3, charge=2)

        assert cd.system.multiplicity == 3
        assert cd.system.charge == 2

    def test_overwrite_only_multiplicity(self, caplog):
        """Test overwriting only multiplicity."""
        cd = CalculationData()
        caplog.set_level(logging.INFO)
        overwrite_from_commandline(cd, multiplicity=2, charge=None)

        assert cd.system.multiplicity == 2

    def test_overwrite_only_charge(self, caplog):
        """Test overwriting only charge."""
        cd = CalculationData()
        caplog.set_level(logging.INFO)
        overwrite_from_commandline(cd, multiplicity=None, charge=-2)

        assert cd.system.charge == -2

    def test_overwrite_with_same_values(self, caplog):
        """Test that no message is printed when values are the same."""
        cd = CalculationData()
        original_charge = cd.system.charge
        original_mult = cd.system.multiplicity

        overwrite_from_commandline(
            cd, multiplicity=original_mult, charge=original_charge
        )

        assert cd.system.multiplicity == original_mult
        assert cd.system.charge == original_charge
        # Should not log info messages when values are the same
        assert caplog.text.strip() == ""

    def test_overwrite_with_none_values(self, caplog):
        """Test that None values don't change anything."""
        cd = CalculationData()
        original_charge = cd.system.charge
        original_mult = cd.system.multiplicity

        overwrite_from_commandline(cd, multiplicity=None, charge=None)

        assert cd.system.multiplicity == original_mult
        assert cd.system.charge == original_charge
        assert caplog.text.strip() == ""


def test_overwrite_from_commandline_no_change(caplog):
    cd = CalculationData()
    overwrite_from_commandline(cd, multiplicity=1, charge=0)
    # no message expected because nothing changed
    assert caplog.text.strip() == ""


def test_overwrite_from_commandline_partial(caplog):
    cd = CalculationData()
    caplog.set_level(logging.INFO)
    overwrite_from_commandline(cd, multiplicity=None, charge=-2)
    assert cd.system.multiplicity == 1  # unchanged
    assert cd.system.charge == -2
