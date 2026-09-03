import shutil
import os
from pathlib import Path
from ffits.io.reader import readin_xyz
from ffits.utils.rmsd import kabsch_rmsd
from tests.test_utils import run_ffits_as_subprocess, SMALL_MOLECULE_DIR


def test_ffits_standard_tsguess_run(tmp_workdir):
    """
    Test that ffits creates ts_guess.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with standard configuration
    """
    examples_dir = SMALL_MOLECULE_DIR
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

    ts_guess_file = temp_path / "ts_guess.xyz"
    assert (
        ts_guess_file.exists()
    ), f"ts_guess.xyz was not created in {temp_path}"

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


def test_ffits_with_hessreadin(tmp_workdir):
    """
    Test which first calculates with present hessian and then without hessian and wbo calculation but reading in the files created in the first run, by changing the filenames in the config file
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    custom_input = """
[reactant.calculation]
hessian_calc = false
[product.calculation]
hessian_calc = false
"""
    with open(temp_path / "custom_config.toml", "w") as f:
        f.write(custom_input)

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"
    _, _, optimized_xyz, _ = readin_xyz(temp_path / "ts_guess.xyz")

    os.remove(temp_path / "ts_guess.xyz")

    # list every file in temp_path
    for file in temp_path.iterdir():
        print(file.name)

    # with config and readin
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--config", "custom_config.toml"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    _, _, optimized_xyz_withreadin, _ = readin_xyz(temp_path / "ts_guess.xyz")

    # compare both ts guess through rmsd
    assert (
        kabsch_rmsd(optimized_xyz, optimized_xyz_withreadin) < 1e-5
    ), "Optimized geometries differ between runs with and without hessian calculation."


def test_ffits_with_improperdihedrals(tmp_workdir):
    """
    Test which first calculates with present hessian and then without hessian and wbo calculation but reading in the files created in the first run, by changing the filenames in the config file
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    custom_input = """
[reactant.calculation]
only_proper_dihedrals = false
[product.calculation]
only_proper_dihedrals = false
"""
    with open(temp_path / "custom_config.toml", "w") as f:
        f.write(custom_input)

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--config", "custom_config.toml"]
    )

    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

    ts_guess_file = temp_path / "ts_guess.xyz"
    assert (
        ts_guess_file.exists()
    ), f"ts_guess.xyz was not created in {temp_path}"

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


def test_ffits_with_nohessweighting(tmp_workdir):
    """
    Test which first calculates with present hessian and then without hessian and wbo calculation but reading in the files created in the first run, by changing the filenames in the config file
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    custom_input = """
[ts_guess.calculation]
average_with_hess_weight = false
"""
    with open(temp_path / "custom_config.toml", "w") as f:
        f.write(custom_input)

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--config", "custom_config.toml"]
    )

    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

    ts_guess_file = temp_path / "ts_guess.xyz"
    assert (
        ts_guess_file.exists()
    ), f"ts_guess.xyz was not created in {temp_path}"

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


def test_ffits_with_wboreadin(tmp_workdir):
    """
    Test which first calculates with present hessian and then without hessian and wbo calculation but reading in the files created in the first run, by changing the filenames in the config file
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    custom_input = """
[reactant.calculation]
wbo_calc = false
[product.calculation]
wbo_calc = false
"""
    with open(temp_path / "custom_config.toml", "w") as f:
        f.write(custom_input)

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"
    _, _, optimized_xyz, _ = readin_xyz(temp_path / "ts_guess.xyz")

    os.remove(temp_path / "ts_guess.xyz")

    # list every file in temp_path
    for file in temp_path.iterdir():
        print(file.name)

    # with config and readin
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--config", "custom_config.toml"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    _, _, optimized_xyz_withreadin, _ = readin_xyz(temp_path / "ts_guess.xyz")

    # compare both ts guess through rmsd
    assert (
        kabsch_rmsd(optimized_xyz, optimized_xyz_withreadin) < 1e-5
    ), "Optimized geometries differ between runs with and without hessian calculation."


def test_ffits_with_ffreadin(tmp_workdir):
    """
    Test which first calculates with present hessian and then without hessian and ff parameterization calculation but reading in the files created in the first run, by changing the filenames in the config file
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    custom_input = """
[reactant.calculation]
ff_parameterization = false
[product.calculation]
ff_parameterization = false
"""
    with open(temp_path / "custom_config.toml", "w") as f:
        f.write(custom_input)

    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"
    _, _, optimized_xyz, _ = readin_xyz(temp_path / "ts_guess.xyz")

    os.remove(temp_path / "ts_guess.xyz")

    # list every file in temp_path
    for file in temp_path.iterdir():
        print(file.name)

    # with config and readin
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--config", "custom_config.toml"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    _, _, optimized_xyz_withreadin, _ = readin_xyz(temp_path / "ts_guess.xyz")

    # compare both ts guess through rmsd
    assert (
        kabsch_rmsd(optimized_xyz, optimized_xyz_withreadin) < 1e-5
    ), "Optimized geometries differ between runs with and without ff parameterization calculation."


def test_ffits_with_filename_change_config(tmp_workdir):
    """
    Test that ffits creates ts_guess.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with custom toml input
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    temp_path = tmp_workdir
    shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
    shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

    custom_input = """
[reactant.path]
wbo_filename = "wbooo1"
hessian_filename = "struc1test.hess"
ff_filename = "ff1test.csv"

[product.path]
wbo_filename = "wbooo2"
hessian_filename = "struc2test.hess"
ff_filename = "ff2test.csv"

"""
    with open(temp_path / "custom_config.toml", "w") as f:
        f.write(custom_input)
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--config", "custom_config.toml"]
    )
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

    ts_guess_file = temp_path / "ts_guess.xyz"
    assert (
        ts_guess_file.exists()
    ), f"ts_guess.xyz was not created in {temp_path}"

    ff_csv_file = temp_path / "tsff.csv"
    assert ff_csv_file.exists(), f"tsff.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "ff2test.csv"
    assert ff_csv_file.exists(), f"ff2test.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "ff1test.csv"
    assert ff_csv_file.exists(), f"ff1test.csv was not created in {temp_path}"
    ff_csv_file = temp_path / "wbooo1"
    assert ff_csv_file.exists(), f"wbooo1 was not created in {temp_path}"
    ff_csv_file = temp_path / "wbooo2"
    assert ff_csv_file.exists(), f"wbooo2 was not created in {temp_path}"
    ff_csv_file = temp_path / "struc1test.hess"
    assert (
        ff_csv_file.exists()
    ), f"struc1test.hess was not created in {temp_path}"
    ff_csv_file = temp_path / "struc2test.hess"
    assert (
        ff_csv_file.exists()
    ), f"struc2test.hess was not created in {temp_path}"


# TODO dieser Test wird fertig gemacht, wenn die temporary directories printbar sind (muss in der config geändert werden)
# def test_ffits_charge_multiplicity_passed_to_xtb():
#     """
#     Test that ffits correctly passes charge and multiplicity to xTB
#     by reading the xTB output file and verifying the values.
#     """
#     examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_path = Path(temp_dir)
#         for item in examples_dir.iterdir():
#             if item.is_file():
#                 shutil.copy2(item, temp_path / item.name)
#             elif item.is_dir():
#                 shutil.copytree(item, temp_path / item.name)

#         original_cwd = os.getcwd()
#         os.chdir(temp_path)

#         try:
#             # Run ffits with specific charge and multiplicity
#             charge = 2
#             multiplicity = 2
#             stdout, stderr, returncode = run_ffits_as_subprocess([
#                 'struc1.xyz', 'struc2.xyz',
#                 '--charge', str(charge),
#                 '--multiplicity', str(multiplicity)
#             ])

#             if returncode == 0:
#                 # Look for xTB output file (may be in subdirectories or current dir)
#                 xtb_out_file = None
#                 for root, dirs, files in os.walk(temp_path):
#                     if 'xtb.out' in files:
#                         xtb_out_file = Path(root) / 'xtb.out'
#                         break

#                 with open(Path(temp_path, 'xtb.out'), 'r') as f:
#                     content = f.read()

#                     # Extract net charge from xTB output
#                     net_charge = None
#                     unpaired_electrons = None

#                     for line in content.split('\n'):
#                         if ':  net charge' in line:
#                             parts = line.split()
#                             net_charge = int(parts[-2])
#                         if ':  unpaired electrons' in line:
#                             parts = line.split()
#                             unpaired_electrons = int(parts[-2])
#                     print(net_charge, unpaired_electrons)
#                     assert False
#                     assert net_charge is not None, "Could not find net charge in xtb.out"
#                     assert net_charge == charge, f"Expected charge {charge}, but xTB got {net_charge}"

#                     expected_unpaired = multiplicity - 1
#                     assert unpaired_electrons is not None, "Could not find unpaired electrons in xtb.out"
#                     assert unpaired_electrons == expected_unpaired, \
#                         f"Expected unpaired electrons {expected_unpaired} (multiplicity {multiplicity}), but xTB got {unpaired_electrons}"
#         finally:
#             os.chdir(original_cwd)
