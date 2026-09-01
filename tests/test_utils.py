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

SMALL_MOLECULE_DIR = Path(__file__).parent / "examples" / "small_single_molecule"

# NAT/XYZ/ATOM_TYPES/WBO used to be hardcoded literal copies of struc1.xyz and
# wbo1. That's a second source of truth for the same data: regenerating the
# example fixtures (see scripts/regenerate_examples.py) silently left these
# constants stale, since nothing re-derived them. Reading the fixture files
# directly here means there is exactly one source of truth, and these
# constants automatically track the fixtures whenever they're regenerated.
NAT, _reactant_comment, XYZ, ATOM_TYPES = readin_xyz(str(SMALL_MOLECULE_DIR / "struc1.xyz"))
WBO = read_wbo_file(str(SMALL_MOLECULE_DIR / "wbo1"))


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
