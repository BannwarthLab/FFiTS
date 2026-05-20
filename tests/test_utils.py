import subprocess
import numpy as np
import pytest
from pathlib import Path
from ffits.datatype.forcefield_data import ForceField
from ffits.io.reader import readin_xyz, read_wbo_file, read_xtb_hessian
from ffits.external.molbar import get_combinded_priorities, _define_bonds_for_molbar
from ffits.datatype.structure_data import (
    Structure,
    StructurePath,
    StructuralInformation,
)
from ffits.datatype.forcefield_data import ForceField
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)

XYZ = np.array(
    [
        [-2.33287094, 3.31176687, 0.20110100],
        [-0.91630217, 2.85867268, -0.04327585],
        [0.06256276, 3.55862185, 0.08938727],
        [-2.92591176, 3.18375556, -0.71288068],
        [-2.79725946, 2.67965821, 0.96793335],
        [-0.81484498, 1.79716556, -0.36699673],
        [-2.35087245, 4.35655928, 0.51563164],
    ]
)

WBO = {
    (0, 1): 1.02668632226515,
    (1, 2): 1.92755303185758,
    (0, 3): 0.955689824153634,
    (0, 4): 0.955863695522291,
    (1, 5): 0.933812077856736,
    (0, 6): 0.982636418257069,
}

ATOM_TYPES = ["C", "C", "O", "H", "H", "H", "H"]

NAT = 7

SMALL_MOLECULE_DIR = Path(__file__).parent / "examples" / "small_single_molecule"


def run_ffits_as_subprocess(args):
    """Joins all args to a command and runs it as a subprocess, capturing stdout, stderr, and return code.

    Args:
        args (list): string list of command line arguments to pass to ffits (excluding "ffits" itself)

    Returns:
        tuple: stdout, stderr, and return code of the subprocess
    """
    command = "ffits " + " ".join(args)
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def check_for_string_in_file(file_path, expected_string):
    """Checks if the expected string is present in the file at the given path.

    Args:
        file_path (str or Path): Path to the file to check.
        expected_string (str): The string to search for in the file.

    Returns:
        bool: True if the expected string is found in the file, False otherwise.
    """
    with open(file_path, "r") as f:
        content = f.read()
        return expected_string in content


def reactant_structure_main():
    """Create a reactant structure for testing using real example files."""
    test_dir = SMALL_MOLECULE_DIR

    xyz_file = str(test_dir / "struc1.xyz")
    wbo_file = str(test_dir / "wbo1")
    ff_file = str(test_dir / "ff1.csv")
    hessian_file = str(test_dir / "struc1.hess")

    nat, _, xyz, atom_types = readin_xyz(xyz_file)
    wbo = read_wbo_file(wbo_file)
    hessian = read_xtb_hessian(hessian_file)

    path = StructurePath(
        xyz_filename=xyz_file,
        wbo_filename=wbo_file,
        hessian_filename=hessian_file,
        ff_filename=ff_file,
    )
    ff = ForceField(
        nat=nat,
        ff_filename=ff_file,
        readff=True,
        energy_calculator=energy_ff,
        gradient_calculator=complete_gradient,
        hessian_calculator=complete_hessian,
    )
    info = StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo,
        atom_types=np.array(atom_types),
        hessian=hessian,
    )
    return Structure(path=path, ff=ff, info=info)


def product_structure_main():
    """Create a product structure for testing using real example files."""
    test_dir = SMALL_MOLECULE_DIR

    xyz_file = str(test_dir / "struc2.xyz")
    wbo_file = str(test_dir / "wbo2")
    ff_file = str(test_dir / "ff2.csv")
    hessian_file = str(test_dir / "struc2.hess")
    nat, _, xyz, atom_types = readin_xyz(xyz_file)
    wbo = read_wbo_file(wbo_file)
    hessian = read_xtb_hessian(hessian_file)

    path = StructurePath(
        xyz_filename=xyz_file,
        wbo_filename=wbo_file,
        hessian_filename=hessian_file,
        ff_filename=ff_file,
    )
    ff = ForceField(
        nat=nat,
        ff_filename=ff_file,
        readff=True,
        energy_calculator=energy_ff,
        gradient_calculator=complete_gradient,
        hessian_calculator=complete_hessian,
    )
    info = StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo,
        atom_types=np.array(atom_types),
        hessian=hessian,
    )
    return Structure(path=path, ff=ff, info=info)
