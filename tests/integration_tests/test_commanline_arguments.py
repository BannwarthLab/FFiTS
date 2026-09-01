# test here different combinations of commandline arguments for the main function, and check that the expected output files are generated and have the expected content. Also check that the expected errors are raised for invalid arguments.
import shutil
from tests.test_utils import (
    run_ffits_as_subprocess,
    SMALL_MOLECULE_DIR,
    check_for_string_in_file,
)


def test_ffits_with_multiplicity_and_charge(tmp_workdir):
    """
    Test that ffits creates optimized.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with standard configuration
    """
    examples_dir = SMALL_MOLECULE_DIR
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--mult 2", "--charge 1"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

    optimized_file = temp_path / "optimized.xyz"
    assert (
        optimized_file.exists()
    ), f"optimized.xyz was not created in {temp_path}"
    assert (
        "xTB will be run with uhf = 1, chrg = 1" in stdout
    ), "Defined charge and multiplicity were not correctly passed to xTB as indicated in optimized.xyz"
    ff_csv_file = temp_path / "tsff.csv"
    assert ff_csv_file.exists(), f"tsff.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "ff1.csv"
    assert ff_csv_file.exists(), f"ff1.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "ff2.csv"
    assert ff_csv_file.exists(), f"ff2.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "wbo1"
    assert ff_csv_file.exists(), f"wbo1 was not created in {temp_path}"
    ff_csv_file = temp_path / "wbo2"
    assert ff_csv_file.exists(), f"wbo2 was not created in {temp_path}"
    ff_csv_file = temp_path / "struc1.hess"
    assert ff_csv_file.exists(), f"struc1.hess was not created in {temp_path}"
    ff_csv_file = temp_path / "struc2.hess"
    assert ff_csv_file.exists(), f"struc2.hess was not created in {temp_path}"
