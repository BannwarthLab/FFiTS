from tests.test_utils import run_ffits_as_subprocess, SMALL_MOLECULE_DIR
import shutil
import pandas as pd


def test_standard_parameterize_run(tmp_workdir):
    """
    Test that ffits creates ff1.csv, etc when run with a single structure (reac.xyz) with standard configuration
    """
    examples_dir = SMALL_MOLECULE_DIR
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")

    stdout, stderr, returncode = run_ffits_as_subprocess(["reac.xyz", "--parameterize"])
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

    output = temp_path / "ff1.csv"
    assert output.exists(), f"ff1.csv was not created in {temp_path}"

    ff_csv_file = temp_path / "ff1.csv"
    assert ff_csv_file.exists(), f"ff1.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "wbo1"
    assert ff_csv_file.exists(), f"wbo1 was not created in {temp_path}"
    ff_csv_file = temp_path / "struc1.hess"
    assert ff_csv_file.exists(), f"struc1.hess was not created in {temp_path}"


def test_parameterize_run_includes_improper_dihedrals(tmp_workdir):
    """
    Test that parameterization mode creates improper dihedrals by default (only_proper_dihedrals = false) and none with only_proper_dihedrals = true
    """
    temp_path = tmp_workdir
    shutil.copy2(SMALL_MOLECULE_DIR / "struc1.xyz", temp_path / "reac.xyz")

    stdout, stderr, returncode = run_ffits_as_subprocess(["reac.xyz", "--parameterize"])
    assert returncode == 0, f"ffits failed with return code {returncode}\nstderr: {stderr}"
    dihedrals = pd.read_csv(temp_path / "ff1.csv").query("type == 'dihedrals'")
    assert (dihedrals["proper_dihedral"] == False).any()
    assert (dihedrals["proper_dihedral"] == True).any()

    (temp_path / "ff1.csv").unlink()
    (temp_path / "config.toml").write_text(
        "[reactant.calculation]\nonly_proper_dihedrals = true\n"
        "[product.calculation]\nonly_proper_dihedrals = true\n"
    )
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "--parameterize", "--config", "config.toml"]
    )
    assert returncode == 0, f"ffits failed with return code {returncode}\nstderr: {stderr}"
    dihedrals = pd.read_csv(temp_path / "ff1.csv").query("type == 'dihedrals'")
    assert (dihedrals["proper_dihedral"] == True).all()
