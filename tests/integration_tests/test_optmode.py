import os
import shutil
from pathlib import Path
from ffits.io.reader import readin_xyz
from ffits.utils.rmsd import kabsch_rmsd
from tests.test_utils import run_ffits_as_subprocess


def test_ffits_optmode(tmp_workdir):
    """
    This test runs ffits in optimizer mode on a single structure and checks if the optimized geometry is consistent with the expected product geometry. It uses the example files for a small single molecule reaction and compares the optimized geometry to the known product geometry using RMSD. The test also checks that the optimization runs successfully without errors.
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    # run ffits normally to generate ff files
    stdout, stderr, returncode = run_ffits_as_subprocess(["reac.xyz", "prod.xyz"])
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"
    _, _, prod_xyz, _ = readin_xyz(temp_path / "prod.xyz")
    _, _, reac_xyz, _ = readin_xyz(temp_path / "reac.xyz")

    if os.path.exists(temp_path / "optimized.xyz"):
        os.remove(temp_path / "optimized.xyz") 

    # list every file in temp_path
    for file in temp_path.iterdir():
        print(file.name)

    # opt from reac to prod
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "--opt", "ff2.csv"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    _, _, optprod_xyz, _ = readin_xyz(temp_path / "optimized.xyz")
    assert (
        kabsch_rmsd(prod_xyz, optprod_xyz) < 5e-2
    ), "Optimized geometries differ between runs with and without hessian calculation."

    # with config and readin
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["prod.xyz", "--opt", "ff1.csv"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    _, _, optreac_xyz, _ = readin_xyz(temp_path / "optimized.xyz")

    # compare both ts guess through rmsd
    assert (
        kabsch_rmsd(reac_xyz, optreac_xyz) < 5e-2
    ), "Optimized geometries differ between runs with and without hessian calculation."
