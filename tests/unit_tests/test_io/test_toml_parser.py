from ffits.io.toml_parser import (
    load_toml,
    load_config,
    load_calculation_data,
    overwrite_from_commandline,
    deep_update,
)
import pytest
import os
import tempfile
from pathlib import Path
from ffits.datatype.calculation_data import CalculationData


class TestLoadToml:
    """Tests for load_toml function."""

    def test_load_toml_valid_file(self):
        """Test loading a valid TOML file."""
        path = os.path.join(os.getcwd(), "ffits", "data", "default_config.toml")
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
            with pytest.raises(Exception):  # tomllib raises various exceptions for invalid TOML
                load_toml(temp_path)
        finally:
            os.unlink(temp_path)


class TestDeepUpdate:
    """Tests for deep_update function."""

    def test_deep_update_simple_dict(self):
        """Test updating a simple flat dictionary."""
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        result = deep_update(base, updates)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_update_nested_dict(self):
        """Test updating nested dictionaries."""
        base = {"system": {"charge": 0, "multiplicity": 1}, "other": "value"}
        updates = {"system": {"charge": -1}}
        result = deep_update(base, updates)
        assert result["system"]["charge"] == -1
        assert result["system"]["multiplicity"] == 1  # Should be preserved
        assert result["other"] == "value"

    def test_deep_update_overwrites_non_dict_with_dict(self):
        """Test that non-dict values can be overwritten with dicts."""
        base = {"key": "value"}
        updates = {"key": {"nested": "dict"}}
        result = deep_update(base, updates)
        assert result["key"] == {"nested": "dict"}

    def test_deep_update_overwrites_dict_with_non_dict(self):
        """Test that dict values can be overwritten with non-dict."""
        base = {"key": {"nested": "dict"}}
        updates = {"key": "simple_value"}
        result = deep_update(base, updates)
        assert result["key"] == "simple_value"

    def test_deep_update_empty_updates(self):
        """Test that empty updates dict returns unchanged base."""
        base = {"a": 1, "b": 2}
        result = deep_update(base, {})
        assert result == base

    def test_deep_update_deeply_nested(self):
        """Test deeply nested dictionary updates."""
        base = {
            "level1": {"level2": {"level3": {"value": "original", "keep": "this"}}}
        }
        updates = {"level1": {"level2": {"level3": {"value": "updated"}}}}
        result = deep_update(base, updates)
        assert result["level1"]["level2"]["level3"]["value"] == "updated"
        assert result["level1"]["level2"]["level3"]["keep"] == "this"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_no_user_path(self):
        """Test loading default config without user override."""
        config = load_config()
        assert isinstance(config, dict)
        assert "system" in config
        assert config["system"]["charge"] == 0

    def test_load_config_with_nonexistent_user_path(self, capsys):
        """Test that nonexistent user path falls back to defaults."""
        config = load_config("/nonexistent/path/config.toml")
        assert isinstance(config, dict)
        captured = capsys.readouterr()
        assert "No user config found" in captured.out

    def test_load_config_merges_correctly(self):
        """Test that user config correctly merges with defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[system]
charge = 5
"""
            )
            f.flush()
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["system"]["charge"] == 5
            # Other system settings should still be present from defaults
            assert "multiplicity" in config["system"]
        finally:
            os.unlink(temp_path)


class TestLoadCalculationData:
    """Tests for load_calculation_data function."""

    def test_load_calculation_data_no_user_path(self):
        """Test loading calculation data with defaults."""
        cd = load_calculation_data()
        assert isinstance(cd, CalculationData)
        assert cd.system.charge == 0
        assert cd.system.multiplicity == 1
        assert cd.postprocessing.relaxation == "None"

    def test_load_calculation_data_invalid_multiplicity_too_high(self):
        """Test that invalid multiplicity (too high) raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[system]
multiplicity = 5
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="chemically unreasonable"):
                load_calculation_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_invalid_multiplicity_too_low(self):
        """Test that invalid multiplicity (too low) raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[system]
multiplicity = 0
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="chemically unreasonable"):
                load_calculation_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_xtb_path(self):
        """Test that xtb_path is correctly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[system]
xtb_path = "custom_xtb_path"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            cd = load_calculation_data(temp_path)
            assert cd.system.xtb_path == "custom_xtb_path"
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_invalid_factor_sum(self):
        """Test that invalid factor sum raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
factor_reactant = 0.3
factor_product = 0.3
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="factor_reactant.*factor_product.*must equal 1.0"):
                load_calculation_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_invalid_optimizer(self):
        """Test that invalid optimizer raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
optimizer = "invalid-optimizer"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid optimization option"):
                load_calculation_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_invalid_relaxation(self):
        """Test that invalid relaxation option raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[postprocessing]
relaxation = "invalid-relaxation"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid relaxation option"):
                load_calculation_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_sets_reactant_paths(self):
        """Test that reactant paths are correctly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.path]
wbo_filename = "wbooo1"
hessian_filename = "struc1test.hess"
ff_filename = "ff1test.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            cd = load_calculation_data(temp_path)
            assert cd.reactant_path.wbo_filename == "wbooo1"
            assert cd.reactant_path.hessian_filename == "struc1test.hess"
            assert cd.reactant_path.ff_filename == "ff1test.csv"
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_sets_product_paths(self):
        """Test that product paths are correctly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[product.path]
wbo_filename = "wbooo1"
hessian_filename = "struc1test.hess"
ff_filename = "ff1test.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            cd = load_calculation_data(temp_path)
            assert cd.product_path.wbo_filename == "wbooo1"
            assert cd.product_path.hessian_filename == "struc1test.hess"
            assert cd.product_path.ff_filename == "ff1test.csv"
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_sets_ts_paths(self):
        """Test that TS guess paths are correctly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.path]
ff_filename = "custom_ts_ff.txt"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            cd = load_calculation_data(temp_path)
            assert cd.ts_path.ff_filename == "custom_ts_ff.txt"
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_sets_calculation_flags(self):
        """Test that calculation flags are correctly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
geometry_optimization = true
wbo_calc = false
hessian_calc = true
"""
            )
            f.flush()
            temp_path = f.name

        try:
            cd = load_calculation_data(temp_path)
            assert cd.reactant_calc.geometry_optimization is True
            assert cd.reactant_calc.wbo_calc is False
            assert cd.reactant_calc.hessian_calc is True
        finally:
            os.unlink(temp_path)

    def test_load_calculation_data_sets_optimization_parameters(self):
        """Test that optimization parameters are correctly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
factor_reactant = 0.6
factor_product = 0.4
molbar_optimizer_e_tol = 1e-6
"""
            )
            f.flush()
            temp_path = f.name

        try:
            cd = load_calculation_data(temp_path)
            assert cd.ts_calc.factor_reactant == 0.6
            assert cd.ts_calc.factor_product == 0.4
            assert cd.ts_calc.molbar_optimizer_e_tol == 1e-6
        finally:
            os.unlink(temp_path)


class TestOverwriteFromCommandline:
    """Tests for overwrite_from_commandline function."""

    def test_overwrite_both_multiplicity_and_charge(self, capsys):
        """Test overwriting both multiplicity and charge."""
        cd = CalculationData()
        original_charge = cd.system.charge
        original_mult = cd.system.multiplicity

        overwrite_from_commandline(cd, multiplicity=3, charge=2)

        assert cd.system.multiplicity == 3
        assert cd.system.charge == 2
        captured = capsys.readouterr()
        assert "Multiplicity" in captured.out
        assert "Charge" in captured.out

    def test_overwrite_only_multiplicity(self, capsys):
        """Test overwriting only multiplicity."""
        cd = CalculationData()
        overwrite_from_commandline(cd, multiplicity=2, charge=None)

        assert cd.system.multiplicity == 2
        captured = capsys.readouterr()
        assert "Multiplicity" in captured.out

    def test_overwrite_only_charge(self, capsys):
        """Test overwriting only charge."""
        cd = CalculationData()
        overwrite_from_commandline(cd, multiplicity=None, charge=-2)

        assert cd.system.charge == -2
        captured = capsys.readouterr()
        assert "Charge" in captured.out

    def test_overwrite_with_same_values(self, capsys):
        """Test that no message is printed when values are the same."""
        cd = CalculationData()
        original_charge = cd.system.charge
        original_mult = cd.system.multiplicity

        overwrite_from_commandline(
            cd, multiplicity=original_mult, charge=original_charge
        )

        assert cd.system.multiplicity == original_mult
        assert cd.system.charge == original_charge
        captured = capsys.readouterr()
        # Should not print info messages when values are the same
        assert captured.out.strip() == ""

    def test_overwrite_with_none_values(self, capsys):
        """Test that None values don't change anything."""
        cd = CalculationData()
        original_charge = cd.system.charge
        original_mult = cd.system.multiplicity

        overwrite_from_commandline(cd, multiplicity=None, charge=None)

        assert cd.system.multiplicity == original_mult
        assert cd.system.charge == original_charge
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

def test_overwrite_from_commandline_no_change(capsys):
    cd = CalculationData()
    overwrite_from_commandline(cd, multiplicity=1, charge=0)
    captured = capsys.readouterr()
    # no message expected because nothing changed
    assert captured.out.strip() == ""

def test_overwrite_from_commandline_partial(capsys):
    cd = CalculationData()
    overwrite_from_commandline(cd, multiplicity=None, charge=-2)
    captured = capsys.readouterr()
    assert cd.system.multiplicity == 1  # unchanged
    assert cd.system.charge == -2
    assert "Charge" in captured.out
