import os
import shutil
from pathlib import Path
from molbar.barcode import get_molbar_from_file
from tests.test_utils import run_ffits_as_subprocess


def test_ffits_optmode(tmp_workdir):
    """
    This test runs ffits in optimizer mode on a single structure and checks if the optimized geometry is consistent with the expected product geometry. It uses the example files for a small single molecule reaction and compares the optimized geometry to the known product geometry by MolBar (same molecule incl. stereochemistry, independent of atom labels). The test also checks that the optimization runs successfully without errors.
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
    # Compare chemically (MolBar: constitution + stereo, independent of atom
    # labels) instead of by RMSD: an H atom moving onto a planar C can attach
    # from either face, which for equivalent atoms (e.g. a CH3 group) gives
    # the same molecule with permuted labels (labeled RMSD ~1 A).
    assert get_molbar_from_file(
        str(temp_path / "optimized.xyz")
    ) == get_molbar_from_file(
        str(temp_path / "prod.xyz")
    ), "Optimization with ff2 from the reactant did not reach the product."

    # with config and readin
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["prod.xyz", "--opt", "ff1.csv"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert get_molbar_from_file(
        str(temp_path / "optimized.xyz")
    ) == get_molbar_from_file(
        str(temp_path / "reac.xyz")
    ), "Optimization with ff1 from the product did not reach the reactant."
