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
def dummy_xyz(tmp_path):
    """Create a dummy xyz file for testing."""
    xyz = tmp_path / "test.xyz"
    xyz.write_text("3\ncomment\nH 0 0 0\nH 0 0 1\nO 0 1 0\n")
    return xyz


@pytest.fixture(autouse=True)
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to avoid calling real executables."""
    def fake_run(cmd, **kwargs):
        class Result:
            def __init__(self):
                self.returncode = 0
                self.stdout = "/usr/bin/xtb\n"
                self.stderr = ""
        return Result()
    monkeypatch.setattr("subprocess.run", fake_run)
    yield


@pytest.fixture
def mock_run_xtb(monkeypatch):
    """Mock Xtb._run_xtb to skip external call and just create fake outputs."""
    def fake_run_xtb(self, command, cwd):
        (cwd / "xtb.out").write_text("   | TOTAL ENERGY              -10.356129503637 Eh   |\n")
        (cwd / "xtbopt.xyz").write_text("H 0 0 0\n")
        (cwd / "xtbopt.log").write_text("H 0 0 0\n")
        (cwd / "hessian").write_text("1 2 3\n4 5 6 \n7 8 9 \n")
        (cwd / "wbo").write_text("1 2 0.9\n2 3 0.8\n")
        return 0
    monkeypatch.setattr(Xtb, "_run_xtb", fake_run_xtb)
    yield


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_check_xtb_loaded_success(dummy_xyz):
    xtb = Xtb(chrg=0, mult=1, xtb_path="xtb")
    assert xtb.xtb_path == "xtb"
    assert xtb.chrg == 0
    assert xtb.uhf == 0


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


def test_singlepoint_creates_output(dummy_xyz, tmp_path, mock_run_xtb):
    xtb = Xtb(chrg=0, mult=1)
    energy = xtb.singlepoint(str(dummy_xyz), tmp_path / "out")
    assert isinstance(energy, float)
    assert Path(tmp_path / "out_singlepoint.out").exists()


def test_hesscalc_reads_hessian(dummy_xyz, tmp_path, mock_run_xtb):
    xtb = Xtb(0, 1)
    hessian = xtb.hesscalc(str(dummy_xyz), tmp_path / "hess_output")
    assert isinstance(hessian, np.ndarray)
    np.testing.assert_equal(hessian, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_wbocalc_reads_wbo(dummy_xyz, tmp_path, mock_run_xtb):
    xtb = Xtb(0, 1)
    wbo = xtb.wbocalc(str(dummy_xyz), tmp_path / "wbo_output")
    assert isinstance(wbo, dict)
    assert (1, 2) in wbo

# TODO needs to be done differently somehow
# def test_geomopt_with_topology_check_no_change(dummy_xyz, tmp_path, mock_run_xtb):
#     xtb = Xtb(0, 1)
#     result = xtb.geomopt_with_topology_check(str(dummy_xyz), tmp_path / "geo")
#     assert result is str( tmp_path / "geo")
    
