import subprocess
import numpy as np
import pytest
from pathlib import Path
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.io.reader import readin_xyz, read_wbo_file, read_xtb_hessian
from ffits.external.molbar import get_combinded_priorities, _define_bonds_for_molbar
from ffits.datatype.structure_data import (
    Structure,
    StructurePath,
    ForceField,
    StructuralInformation,
)
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


def run_ffits_as_subprocess(args):
    command = "ffits " + " ".join(args)
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def reactant_structure_main():
    """Create a reactant structure for testing using real example files."""
    test_dir = Path(__file__).parent / "examples" / "small_single_molecule"

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
    test_dir = Path(__file__).parent / "examples" / "small_single_molecule"

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
