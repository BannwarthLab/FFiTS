from tests.test_utils import run_ffits_as_subprocess, SMALL_MOLECULE_DIR
import shutil
from pathlib import Path
import tempfile
import os

def test_standard_parameterize_run():
    """
    Test that ffits creates optimized.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with standard configuration
    """
    examples_dir = SMALL_MOLECULE_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
        shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            stdout, stderr, returncode = run_ffits_as_subprocess(
                ["reac.xyz", "prod.xyz", "--parameterize"]
            )
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            assert (
                returncode == 0
            ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

            output = temp_path / "ff1.csv"
            assert (
                output.exists()
            ), f"ff1.csv was not created in {temp_path}"

            ff_csv_file = temp_path / "ff1.csv"
            assert ff_csv_file.exists(), f"ff1.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "wbo1"
            assert ff_csv_file.exists(), f"wbo1 was not created in {temp_path}"
            ff_csv_file = temp_path / "struc1.hess"
            assert ff_csv_file.exists(), f"struc1.hess was not created in {temp_path}"
        finally:
            os.chdir(original_cwd)
