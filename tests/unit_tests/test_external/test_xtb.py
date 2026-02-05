import pytest
import builtins
from pathlib import Path
import tempfile
import os
import numpy as np

import ffits.io.reader as reader
from ffits.external.xtb import Xtb  # adjust import to your actual module path


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
    xyz.write_text("3\nWater molecule\nO     0.000000    0.000000    0.000000\nH     0.957200    0.000000    0.000000\nH    -0.239935    0.926640    0.000000\n")
    return xyz


@pytest.fixture
def methane_xyz(tmp_path):
    """Create a methane molecule for integration testing."""
    xyz = tmp_path / "methane.xyz"
    xyz.write_text("5\nMethane\nC     0.000000    0.000000    0.000000\nH     0.629118    0.629118    0.629118\nH    -0.629118   -0.629118    0.629118\nH    -0.629118    0.629118   -0.629118\nH     0.629118   -0.629118   -0.629118\n")
    return xyz

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
    f.write_text(" some lines\n           | TOTAL ENERGY              -12.3456 Eh   |\n more\n")
    xtb = Xtb(str(f), 0, 1)
    energy = xtb._find_energy_in_output(f)
    assert energy == pytest.approx(-12.3456)

# TODO needs to be done differently somehow
# def test_geomopt_with_topology_check_no_change(dummy_xyz, tmp_path, mock_run_xtb):
#     xtb = Xtb(0, 1)
#     result = xtb.geomopt_with_topology_check(str(dummy_xyz), tmp_path / "geo")
#     assert result is str( tmp_path / "geo")


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
    
    def test_geomopt_water_produces_optimized_geometry(self, water_xyz, change_to_tmp_path, capsys):
        """Test geometry optimization produces valid output and returns parsed data."""
        xtb = Xtb(chrg=0, mult=1)
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
            frames = [line.strip() for line in traj_content.split('\n') if line.strip().isdigit() and int(line.strip()) == 3]
            assert len(frames) >= 2, f"Trajectory should have multiple frames (at least 2 atom count lines), got {len(frames)}"
        
        #  structural change
        with open(water_xyz) as f:
            orig_lines = f.readlines()
        with open(output_file) as f:
            opt_lines = f.readlines()
        orig_coords = orig_lines[2:]
        opt_coords = opt_lines[2:]
        assert orig_coords != opt_coords, "Optimized geometry should differ from starting geometry"
        
        captured = capsys.readouterr()
        assert "finished successfully" in captured.out, "Expected completion message in stdout"
    
    def test_geomopt_methane_structure_changes(self, methane_xyz, change_to_tmp_path):
        """Test that geometry optimization changes coordinates."""
        xtb = Xtb(chrg=0, mult=1)
        output_file = str(change_to_tmp_path / "methane_opt.xyz")
        nat, comment, coordinates, atom_types = xtb.geomopt(str(methane_xyz), output_file)
        
        # check returned values
        assert atom_types == ['C', 'H', 'H', 'H', 'H'], 'Atom types should match methane.'
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
    
    def test_geomopt_with_output_dir(self, water_xyz, change_to_tmp_path, capsys):
        """Test that all xtb output files are copied to output_dir."""
        xtb = Xtb(chrg=0, mult=1)
        output_dir = change_to_tmp_path / "xtb_output"
        output_file = str(change_to_tmp_path / "water_opt.xyz")
        
        result = xtb.geomopt(
            str(water_xyz), 
            output_file,
            output_dir=str(output_dir)
        )
        
        #  valid tuple
        nat, comment, coordinates, atom_types = result
        assert nat == 3
        
        # Check that output directory was created and populated
        assert output_dir.exists()
        
        # for debugging
        output_files = list(output_dir.glob("**/*"))
        print(f"\n=== Files in output_dir ===")
        for f in sorted(output_files):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f.relative_to(output_dir)}: {size} bytes")

        # Check files are in output_dir
        assert (output_dir / "xtbopt.xyz").exists(), "xtbopt.xyz should be in output_dir"
        assert (output_dir / "xtbopt.log").exists(), "xtbopt.log should be in output_dir"
        assert (output_dir / "xtb.out").exists(), "xtb.out should be in output_dir"
        assert (output_dir / "xtbrestart").exists(), "xtbrestart should be in output_dir"
        assert (output_dir / "charges").exists(), "charges should be in output_dir"
        assert (output_dir / "wbo").exists(), "wbo should be in output_dir"
        assert (output_dir / "xtbtopo.mol").exists(), "xtbtopo.mol should be in output_dir"
        
        assert Path(output_file).exists(), "Optimized geometry should be saved in tmp_path"
        
        captured = capsys.readouterr()
        assert "finished successfully" in captured.out


class TestXtbHessian:
    """Integration tests for Hessian calculations."""
    
    def test_hesscalc_water_produces_hessian(self, water_xyz, tmp_path):
        """Test Hessian calculation produces valid matrix."""
        xtb = Xtb(chrg=0, mult=1)
        hessian = xtb.hesscalc(str(water_xyz), str(tmp_path / "water_hessian"))
        
        # Check that we got a numpy array
        assert isinstance(hessian, np.ndarray)
        
        # For water (3 atoms), Hessian should be 9x9 (3*3)
        assert hessian.shape == (9, 9)
        
        # Hessian should be symmetric
        np.testing.assert_allclose(hessian, hessian.T, rtol=1e-10)
        
        # Check that Hessian file was created
        assert (tmp_path / "water_hessian").exists()
    
    def test_hesscalc_eigenvalues_reasonable(self, methane_xyz, tmp_path):
        """Test that Hessian has reasonable eigenvalues."""
        xtb = Xtb(chrg=0, mult=1)
        hessian = xtb.hesscalc(str(methane_xyz), str(tmp_path / "methane_hessian"))
        
        # For methane (5 atoms), Hessian should be 15x15
        assert hessian.shape == (15, 15)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(hessian)
        
        # All eigenvalues should be real (within numerical precision)
        assert np.allclose(eigenvalues.imag, 0, atol=1e-10)
        
        # Most eigenvalues should be positive for a minimum
        real_eigenvalues = eigenvalues.real
        num_positive = np.sum(real_eigenvalues > 1e-4)
        assert num_positive >= 12  # Should have many positive eigenvalues


class TestXtbWBO:
    """Integration tests for Wiberg Bond Order calculations."""
    
    def test_wbocalc_water_returns_bond_orders(self, water_xyz, tmp_path):
        """Test WBO calculation returns bond orders."""
        xtb = Xtb(chrg=0, mult=1)
        wbo = xtb.wbocalc(str(water_xyz), str(tmp_path / "water_wbo"))
        
        # Check that we got a dictionary
        assert isinstance(wbo, dict)
        
        # Water should have 2 O-H bonds
        assert len(wbo) >= 2
        
        # WBO values should be positive and reasonable (< 1 for single bonds)
        for bond, value in wbo.items():
            assert isinstance(bond, tuple)
            assert len(bond) == 2
            assert 0 < value < 1.5  # Reasonable range for bond orders
        
        # Check that WBO file was created
        assert (tmp_path / "water_wbo").exists()
    
    def test_wbocalc_methane_bond_orders(self, methane_xyz, tmp_path):
        """Test WBO for methane."""
        xtb = Xtb(chrg=0, mult=1)
        wbo = xtb.wbocalc(str(methane_xyz), str(tmp_path / "methane_wbo"))
        
        # Methane should have 4 C-H bonds
        assert len(wbo) == 4
        
        # All C-H bonds should have similar bond orders (all single bonds)
        values = list(wbo.values())
        assert all(0.8 < v < 1.1 for v in values)  # All should be ~1


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
    
    def test_geomopt_with_topology_check_simple(self, water_xyz, tmp_path):
        """Test geometry optimization with topology check."""
        xtb = Xtb(chrg=0, mult=1)
        result_path, wbo_after = xtb.geomopt_with_topology_check(
            str(water_xyz), 
            str(tmp_path / "water_geo"),
            str(tmp_path / "water_wbo"),
            threshold=0.3
        )
        
        # Should return optimized xyz path and final WBO dict
        assert isinstance(result_path, str)
        assert Path(result_path).exists()
        assert isinstance(wbo_after, dict)
        
        # Water should still have 2 bonds
        assert len(wbo_after) >= 2


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
