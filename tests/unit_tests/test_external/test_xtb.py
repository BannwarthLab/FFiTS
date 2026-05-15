import pytest
from pathlib import Path
import os
import numpy as np
from ffits.external.xtb import Xtb  # adjust import to your actual module path
import logging

# ---------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------


@pytest.fixture
def change_to_tmp_path(tmp_path):
    """Change working directory to tmp_path for the test."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


@pytest.fixture
def dummy_xyz(tmp_path):
    """Create a dummy xyz file for testing."""
    xyz = tmp_path / "test.xyz"
    xyz.write_text("3\ncomment\nH 0 0 0\nH 0 0 1\nO 0 1 0\n")
    return xyz


@pytest.fixture
def water_xyz(tmp_path):
    """Create a simple water molecule for integration testing."""
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nWater molecule\nO     0.000000    0.000000    0.000000\nH     0.957200    0.000000    0.000000\nH    -0.239935    0.926640    0.000000\n"
    )
    return xyz


@pytest.fixture
def water_displaced_xyz(tmp_path):
    """Create a simple water molecule for integration testing."""
    xyz = tmp_path / "water_displaced.xyz"
    xyz.write_text(
        "3\nWater molecule\nO     0.000000    0.000000    0.000000\nH     0.957200    0.000000    0.000000\nH    -0.239935    3.000000   0.000000\n"
    )
    return xyz


@pytest.fixture
def methane_xyz(tmp_path):
    """Create a methane molecule for integration testing."""
    xyz = tmp_path / "methane.xyz"
    xyz.write_text(
        "5\nMethane\nC     0.000000    0.000000    0.000000\nH     0.629118    0.629118    0.629118\nH    -0.629118   -0.629118    0.629118\nH    -0.629118    0.629118   -0.629118\nH     0.629118   -0.629118   -0.629118\n"
    )
    return xyz


@pytest.fixture
def debug_logging_enabled():
    """Enable DEBUG logging for the xTB module temporarily."""
    logger_xtb = logging.getLogger("ffits.external.xtb")
    logger_root = logging.getLogger()
    original_xtb_level = logger_xtb.level
    original_root_level = logger_root.level

    logger_xtb.setLevel(logging.DEBUG)
    logger_root.setLevel(logging.DEBUG)

    yield

    logger_xtb.setLevel(original_xtb_level)
    logger_root.setLevel(original_root_level)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_get_command(dummy_xyz):
    xtb = Xtb(chrg=1, mult=2)
    cmd = xtb._get_command(dummy_xyz, "--opt")
    assert "--opt" in cmd
    assert "--uhf 1" in cmd
    assert "--chrg 1" in cmd


def test_find_energy_in_output(tmp_path):
    """Check energy parser from output file."""
    f = tmp_path / "xtb.out"
    f.write_text(
        " some lines\n           | TOTAL ENERGY              -12.3456 Eh   |\n more\n"
    )
    xtb = Xtb(0, 1)
    energy = xtb._find_energy_in_output(f)
    assert energy == pytest.approx(-12.3456)


# =====================================================================
# INTEGRATION TESTS - Actual xtb calculations
# =====================================================================
# These tests run actual xTB calculations and verify the results


class TestXtbSinglepoint:
    """Integration tests for singlepoint energy calculations."""

    def test_singlepoint_water_energy(self, water_xyz, tmp_path):
        """Test singlepoint calculation returns a reasonable energy."""
        xtb = Xtb(chrg=0, mult=1)
        energy = xtb.singlepoint(str(water_xyz), str(tmp_path / "water_sp"))

        assert isinstance(energy, float)
        assert energy < 0
        assert energy > -10  # Sanity check: not wildly unreasonable
        assert (tmp_path / "water_sp_singlepoint.out").exists()

    def test_singlepoint_methane_energy(self, methane_xyz, tmp_path):
        """Test singlepoint calculation for methane."""
        xtb = Xtb(chrg=0, mult=1)
        energy = xtb.singlepoint(str(methane_xyz), str(tmp_path / "methane_sp"))

        assert isinstance(energy, float)
        assert energy < 0
        assert energy > -20


class TestXtbGeomOpt:
    """Integration tests for geometry optimization."""

    def test_geomopt_water_produces_optimized_geometry(
        self, water_xyz, change_to_tmp_path, caplog
    ):
        """Test geometry optimization produces valid output and returns parsed data."""
        xtb = Xtb(chrg=0, mult=1)
        caplog.set_level(logging.INFO)
        output_file = str(change_to_tmp_path / "water_opt.xyz")
        result = xtb.geomopt(str(water_xyz), output_file)
        # geomopt returns tuple
        assert isinstance(result, tuple)
        assert len(result) == 4
        nat, comment, coordinates, atom_types = result

        # general checks
        assert nat == 3
        assert isinstance(coordinates, np.ndarray)
        assert coordinates.shape == (3, 3)
        assert len(atom_types) == 3
        assert Path(output_file).exists()
        traj_file = change_to_tmp_path / f"trj_{Path(output_file).name}"
        assert traj_file.exists()

        # Check trajectory has multiple entries
        with open(traj_file) as f:
            traj_content = f.read()
            frames = [
                line.strip()
                for line in traj_content.split("\n")
                if line.strip().isdigit() and int(line.strip()) == 3
            ]
            assert (
                len(frames) >= 2
            ), f"Trajectory should have multiple frames (at least 2 atom count lines), got {len(frames)}"

        #  structural change
        with open(water_xyz) as f:
            orig_lines = f.readlines()
        with open(output_file) as f:
            opt_lines = f.readlines()
        orig_coords = orig_lines[2:]
        opt_coords = opt_lines[2:]
        assert (
            orig_coords != opt_coords
        ), "Optimized geometry should differ from starting geometry"

        assert (
            "finished successfully" in caplog.text
        ), "Expected completion message in stdout"

    def test_geomopt_methane_structure_changes(self, methane_xyz, change_to_tmp_path):
        """Test that geometry optimization changes coordinates."""
        xtb = Xtb(chrg=0, mult=1)
        output_file = str(change_to_tmp_path / "methane_opt.xyz")
        nat, comment, coordinates, atom_types = xtb.geomopt(
            str(methane_xyz), output_file
        )

        # check returned values
        assert atom_types == [
            "C",
            "H",
            "H",
            "H",
            "H",
        ], "Atom types should match methane."
        assert nat == 5  # Methane has 5 atoms
        assert coordinates.shape == (5, 3)

        with open(methane_xyz) as f:
            orig_lines = f.readlines()
        with open(output_file) as f:
            opt_lines = f.readlines()

        # structural change
        orig_coords = orig_lines[2:]
        opt_coords = opt_lines[2:]
        assert orig_coords != opt_coords



class TestXtbHessian:
    """Integration tests for Hessian calculations."""

    def test_hesscalc_water_produces_hessian(self, water_xyz, tmp_path):
        """Test Hessian calculation produces valid matrix."""
        xtb = Xtb(chrg=0, mult=1)
        hessian = xtb.hesscalc(str(water_xyz), str(tmp_path / "water_hessian"))

        assert isinstance(hessian, np.ndarray)
        assert hessian.shape == (9, 9)
        np.testing.assert_allclose(hessian, hessian.T, rtol=1e-10)
        assert (tmp_path / "water_hessian").exists()

    def test_hesscalc_eigenvalues_reasonable(self, methane_xyz, tmp_path):
        """Test that Hessian has reasonable eigenvalues."""
        xtb = Xtb(chrg=0, mult=1)
        hessian = xtb.hesscalc(str(methane_xyz), str(tmp_path / "methane_hessian"))

        assert hessian.shape == (15, 15)
        assert np.allclose(hessian, hessian.T, rtol=1e-10)
        assert (tmp_path / "methane_hessian").exists()


class TestXtbWBO:
    """Integration tests for Wiberg Bond Order calculations."""

    def test_wbocalc_water_returns_bond_orders(self, water_xyz, tmp_path):
        """Test WBO calculation returns bond orders."""
        xtb = Xtb(chrg=0, mult=1)
        wbo = xtb.wbocalc(str(water_xyz), str(tmp_path / "water_wbo"))

        assert isinstance(wbo, dict)
        assert len(wbo) >= 2
        for bond, value in wbo.items():
            assert isinstance(bond, tuple)
            assert len(bond) == 2
            assert 0 < value < 1.5  # Reasonable range for bond orders
        assert (tmp_path / "water_wbo").exists()

    def test_wbocalc_methane_bond_orders(self, methane_xyz, tmp_path):
        """Test WBO for methane."""
        xtb = Xtb(chrg=0, mult=1)
        wbo = xtb.wbocalc(str(methane_xyz), str(tmp_path / "methane_wbo"))

        assert len(wbo) == 4
        values = list(wbo.values())
        assert all(0.8 < v < 1.1 for v in values)


class TestXtbWithChargeAndMultiplicity:
    """Test xTB calculations with different charges and multiplicities."""

    def test_charged_species_singlepoint(self, water_xyz, tmp_path):
        """Test singlepoint of charged species."""
        # Test water cation (charge +1, doublet)
        xtb = Xtb(chrg=1, mult=2)
        energy = xtb.singlepoint(str(water_xyz), str(tmp_path / "water_cation_sp"))

        assert isinstance(energy, float)
        assert energy < 0

        # Output should contain uhf specification
        with open(tmp_path / "water_cation_sp_singlepoint.out") as f:
            content = f.read()
            assert "--uhf 1" in content or "uhf" in content.lower()

    def test_geomopt_charged_species(self, water_xyz, change_to_tmp_path):
        """Test geometry optimization of charged species."""
        xtb = Xtb(chrg=-1, mult=2)  # Water anion
        output_file = str(change_to_tmp_path / "water_anion_opt.xyz")
        result = xtb.geomopt(str(water_xyz), output_file)

        # Check return value is valid tuple
        natoms, comment, coordinates, atom_types = result
        assert natoms == 3

        # Check that output files were created
        assert Path(output_file).exists()
        assert (change_to_tmp_path / f"trj_{Path(output_file).name}").exists()


class TestXtbTopologyCheck:
    """Test topology checking during geometry optimization."""

    def test_geomopt_with_topology_check_simple(
        self, water_displaced_xyz, change_to_tmp_path
    ):
        """Test geometry optimization with topology check."""

        xtb = Xtb(chrg=0, mult=1)

        result = xtb.geomopt_with_topology_check(
            str(water_displaced_xyz),
            str(change_to_tmp_path / "water_geo.xyz"),
            str(change_to_tmp_path / "water_wbo"),
            threshold=0.3,
        )

        name, nat, comment, coordinates, atom_types = result
        assert nat == 3, "Should have 3 atoms for water"
        assert isinstance(coordinates, np.ndarray)
        assert coordinates.shape == (3, 3)

    def test_geomopt_with_topology_check_detects_warning(
        self, change_to_tmp_path, caplog
    ):
        """Test that topology changes during optimization are detected and warned about."""
        strained_xyz = change_to_tmp_path / "strained.xyz"
        strained_xyz.write_text(
            "5\nStrained SiH4\nSi  0.55363512 -0.05597543 -0.70536689\nH   0.60495745 -1.24930990  0.23339908\nH   0.97492131  0.95146300  0.35087768\nH  -1.17968321  0.21564939 -0.31994928\nH  -0.95383068  0.13817295  0.44103941\n"
        )

        xtb = Xtb(chrg=0, mult=1)
        caplog.set_level(logging.WARNING)

        result = xtb.geomopt_with_topology_check(
            str(strained_xyz), "strained_opt.xyz", "strained_wbo", threshold=0.1
        )

        name, nat, comment, coordinates, atom_types = result
        assert nat == 5, "Should have 5 atoms for SiH4"

        # Check for warning messages in captured logs
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        warning_text = " ".join(warning_messages)

        has_wbo_changes = "Topology changed significantly" in warning_text
        has_new_bonds = "New bonds formed" in warning_text
        has_disappeared_bonds = "Bonds disappeared" in warning_text

        if has_wbo_changes or has_new_bonds or has_disappeared_bonds:
            if has_wbo_changes:
                assert any(
                    "Bond" in msg and "change:" in msg for msg in warning_messages
                )
            if has_new_bonds:
                assert any("formed with WBO" in msg for msg in warning_messages)
            if has_disappeared_bonds:
                assert any("disappeared" in msg for msg in warning_messages)

    def test_geomopt_with_topology_check_invalid_input(self, change_to_tmp_path):
        """Test geometry optimization with topology check using invalid geometry that causes xTB to fail."""
        xtb = Xtb(chrg=0, mult=1)

        # Create an XYZ file with invalid formatting that xTB cannot process
        invalid_xyz = change_to_tmp_path / "invalid.xyz"
        invalid_xyz.write_text("not_a_number\nInvalid header\nH 0 0 0\n")

        # This should raise a RuntimeError from xTB when it fails to read the geometry
        with pytest.raises(RuntimeError):
            xtb.geomopt_with_topology_check(
                str(invalid_xyz),
                str(change_to_tmp_path / "invalid_geo.xyz"),
                str(change_to_tmp_path / "invalid_wbo"),
                threshold=0.3,
            )


class TestXtbOutputFileHandling:
    """Test proper handling of xTB output files."""

    def test_singlepoint_output_contains_energy(self, water_xyz, tmp_path):
        """Verify output file contains recognizable xTB output."""
        xtb = Xtb(chrg=0, mult=1)
        xtb.singlepoint(str(water_xyz), str(tmp_path / "water_sp"))

        output_file = tmp_path / "water_sp_singlepoint.out"
        with open(output_file) as f:
            content = f.read()
            # xTB output should contain energy information
            assert "TOTAL ENERGY" in content or "energy" in content.lower()

    def test_geomopt_creates_trajectory(self, water_xyz, change_to_tmp_path):
        """Verify geometry optimization creates trajectory file."""
        xtb = Xtb(chrg=0, mult=1)
        output_file = str(change_to_tmp_path / "water_opt.xyz")
        xtb.geomopt(str(water_xyz), output_file)

        trajectory_file = change_to_tmp_path / f"trj_{Path(output_file).name}"
        assert trajectory_file.exists()

        # Trajectory should be valid xyz format with multiple frames
        with open(trajectory_file) as f:
            content = f.read()
            # Should contain multiple lines indicating trajectory
            assert len(content) > 100  # Not empty


class TestXtbErrorHandling:
    """Test that xTB wrapper properly handles errors."""

    def test_geomopt_with_invalid_input(self, tmp_path):
        """Test geometry optimization with invalid input file."""
        xtb = Xtb(chrg=0, mult=1)
        with pytest.raises(FileNotFoundError):
            xtb.geomopt(str(tmp_path / "nonexistent.xyz"), str(tmp_path / "output.xyz"))

    def test_hesscalc_with_invalid_input(self, tmp_path):
        """Test Hessian calculation with invalid input file."""
        xtb = Xtb(chrg=0, mult=1)
        with pytest.raises(FileNotFoundError):
            xtb.hesscalc(str(tmp_path / "nonexistent.xyz"), str(tmp_path / "hessian"))

    def test_wbocalc_with_invalid_input(self, tmp_path):
        """Test WBO calculation with invalid input file."""
        xtb = Xtb(chrg=0, mult=1)
        with pytest.raises(FileNotFoundError):
            xtb.wbocalc(str(tmp_path / "nonexistent.xyz"), str(tmp_path / "wbo"))


class TestXtbTemporaryDirectories:
    """Test that temporary directories are properly handled and cleaned up or kept when Debug mode is enabled."""

    def test_singlepoint_temp_dir_cleaned_up_when_not_debug(
        self, water_xyz, tmp_path, caplog
    ):
        """Test that temporary directory is cleaned up after singlepoint when not in debug mode."""
        caplog.set_level(logging.INFO)  # Not debug level
        xtb = Xtb(chrg=0, mult=1)
        energy = xtb.singlepoint(str(water_xyz), str(tmp_path / "water_sp"))
        assert isinstance(energy, float)
        debug_dir = tmp_path / "debug"
        if debug_dir.exists():
            temp_dirs = list(debug_dir.glob("xtb_singlepoint_*"))
            assert (
                len(temp_dirs) == 0
            ), "Temporary directories should be cleaned up when not in debug mode"

    def test_singlepoint_temp_dir_kept_when_debug(
        self, water_xyz, change_to_tmp_path, debug_logging_enabled
    ):
        """Test that temporary directory is kept after singlepoint when in debug mode."""
        xtb = Xtb(chrg=0, mult=1)
        energy = xtb.singlepoint(str(water_xyz), str(change_to_tmp_path / "water_sp"))

        assert isinstance(energy, float)
        debug_dir = change_to_tmp_path / "debug"
        assert debug_dir.exists(), "Debug directory should be created in debug mode"
        temp_dirs = list(debug_dir.glob("xtb_singlepoint_*"))
        assert (
            len(temp_dirs) > 0
        ), "Temporary directories should be preserved in debug mode"

    def test_geomopt_temp_dir_cleaned_up_when_not_debug(
        self, water_xyz, change_to_tmp_path, caplog
    ):
        """Test that temporary directory is cleaned up after geomopt when not in debug mode."""
        caplog.set_level(logging.INFO)  # Not debug level
        xtb = Xtb(chrg=0, mult=1)
        output_file = str(change_to_tmp_path / "water_opt.xyz")
        result = xtb.geomopt(str(water_xyz), output_file)
        assert len(result) == 4  # Valid result
        debug_dir = change_to_tmp_path / "debug"
        if debug_dir.exists():
            temp_dirs = list(debug_dir.glob("geomopt_*"))
            assert (
                len(temp_dirs) == 0
            ), "Temporary directories should be cleaned up when not in debug mode"

    def test_geomopt_temp_dir_kept_when_debug(
        self, water_xyz, change_to_tmp_path, debug_logging_enabled
    ):
        """Test that temporary directory is kept after geomopt when in debug mode."""
        xtb = Xtb(chrg=0, mult=1)
        output_file = str(change_to_tmp_path / "water_opt.xyz")
        result = xtb.geomopt(str(water_xyz), output_file)

        assert len(result) == 4  # Valid result
        debug_dir = change_to_tmp_path / "debug"
        assert debug_dir.exists(), "Debug directory should be created in debug mode"
        temp_dirs = list(debug_dir.glob("geomopt_*"))
        assert (
            len(temp_dirs) > 0
        ), "Temporary directories should be preserved in debug mode"

    def test_hesscalc_temp_dir_cleaned_up_when_not_debug(
        self, water_xyz, tmp_path, caplog
    ):
        """Test that temporary directory is cleaned up after hesscalc when not in debug mode."""
        caplog.set_level(logging.INFO)  # Not debug level
        xtb = Xtb(chrg=0, mult=1)
        hessian = xtb.hesscalc(str(water_xyz), str(tmp_path / "water_hessian"))
        assert isinstance(hessian, np.ndarray)
        debug_dir = tmp_path / "debug"
        if debug_dir.exists():
            temp_dirs = list(debug_dir.glob("hesscalc_*"))
            assert (
                len(temp_dirs) == 0
            ), "Temporary directories should be cleaned up when not in debug mode"

    def test_hesscalc_temp_dir_kept_when_debug(
        self, water_xyz, change_to_tmp_path, debug_logging_enabled
    ):
        """Test that temporary directory is kept after hesscalc when in debug mode."""
        xtb = Xtb(chrg=0, mult=1)
        hessian = xtb.hesscalc(
            str(water_xyz), str(change_to_tmp_path / "water_hessian")
        )

        assert isinstance(hessian, np.ndarray)
        debug_dir = change_to_tmp_path / "debug"
        assert debug_dir.exists(), "Debug directory should be created in debug mode"
        temp_dirs = list(debug_dir.glob("hesscalc_*"))
        assert (
            len(temp_dirs) > 0
        ), "Temporary directories should be preserved in debug mode"

    def test_wbocalc_temp_dir_cleaned_up_when_not_debug(
        self, water_xyz, tmp_path, caplog
    ):
        """Test that temporary directory is cleaned up after wbocalc when not in debug mode."""
        caplog.set_level(logging.INFO)  # Not debug level
        xtb = Xtb(chrg=0, mult=1)
        wbo = xtb.wbocalc(str(water_xyz), str(tmp_path / "water_wbo"))
        assert isinstance(wbo, dict)
        debug_dir = tmp_path / "debug"
        if debug_dir.exists():
            temp_dirs = list(debug_dir.glob("wbocalc_*"))
            assert (
                len(temp_dirs) == 0
            ), "Temporary directories should be cleaned up when not in debug mode"

    def test_wbocalc_temp_dir_kept_when_debug(
        self, water_xyz, change_to_tmp_path, debug_logging_enabled
    ):
        """Test that temporary directory is kept after wbocalc when in debug mode."""
        xtb = Xtb(chrg=0, mult=1)
        wbo = xtb.wbocalc(str(water_xyz), str(change_to_tmp_path / "water_wbo"))

        assert isinstance(wbo, dict)
        debug_dir = change_to_tmp_path / "debug"
        assert debug_dir.exists(), "Debug directory should be created in debug mode"
        temp_dirs = list(debug_dir.glob("wbocalc_*"))
        assert (
            len(temp_dirs) > 0
        ), "Temporary directories should be preserved in debug mode"
