import tempfile

from ffits.io.reader import readin_xyz
from ffits.main import run_tsguess_mode, run_optimizer_mode
from tests.test_utils import SMALL_MOLECULE_DIR
import os
import shutil
from ffits.utils.rmsd import kabsch_rmsd
from pathlib import Path


def test_main_tsguess_mode_without_defining_calcopt():
    """Test that main function runs without errors in tsguess mode when no calculation options are provided, and that it produces the expected output files."""

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        temp_path = Path(temp_dir)
        shutil.copy2(SMALL_MOLECULE_DIR / "struc1.xyz", temp_path / "reac.xyz")
        shutil.copy2(SMALL_MOLECULE_DIR / "struc2.xyz", temp_path / "prod.xyz")
        try:
            os
            run_tsguess_mode(
                f"reac.xyz",
                f"prod.xyz",
            )
            assert (
                temp_path / "optimized.xyz"
            ).exists(), "optimized.xyz was not created."
            assert (temp_path / "tsff.csv").exists(), "tsff.csv was not created."
            assert (temp_path / "ff1.csv").exists(), "ff1.csv was not created."
            assert (temp_path / "ff2.csv").exists(), "ff2.csv was not created."
            assert (temp_path / "wbo1").exists(), "wbo1 was not created."
            assert (temp_path / "wbo2").exists(), "wbo2 was not created."
            assert (temp_path / "struc1.hess").exists(), "struc1.hess was not created."
            assert (temp_path / "struc2.hess").exists(), "struc2.hess was not created."
            os.chdir(cwd)
        finally:
            os.chdir(cwd)


def test_main_opt_mode_without_defining_calcopt():
    """Test that main function runs without errors in optimizer mode when no calculation options are provided, and that it produces the expected output files."""

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        temp_path = Path(temp_dir)
        shutil.copy2(SMALL_MOLECULE_DIR / "struc1.xyz", temp_path / "reac.xyz")
        shutil.copy2(SMALL_MOLECULE_DIR / "struc2.xyz", temp_path / "prod.xyz")
        shutil.copy2(SMALL_MOLECULE_DIR / "ff2.csv", temp_path / "ff2.csv")
        try:
            os
            run_optimizer_mode(
                f"reac.xyz",
                f"ff2.csv",
            )
            assert (
                temp_path / "optimized.xyz"
            ).exists(), "optimized.xyz was not created."
            _, _, xyz1, _ = readin_xyz(temp_path / "optimized.xyz")
            _, _, xyz2, _ = readin_xyz(temp_path / "prod.xyz")
            assert (
                kabsch_rmsd(xyz1, xyz2) < 0.1
            ), "Optimized geometry is not close to reactant geometry, which is expected when no calculation options are defined."
        finally:
            os.chdir(cwd)
            os.chdir(cwd)
