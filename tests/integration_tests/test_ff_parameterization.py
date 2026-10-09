import copy
import pandas as pd
import pytest
import shutil
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import (
    StructuralInformation,
    StructurePath,
    Structure,
)

from ffits.ts_guess.parameterize_ff import (
    fit_ff_to_hessian,
)
from ffits.io.reader import read_xtb_hessian
from ffits.forcefield.python_interface.ff_energy import complete_hessian
from ffits.ts_guess.define_starting_parameters import fill_ff
from tests.test_utils import NAT, WBO, XYZ, ATOM_TYPES, SMALL_MOLECULE_DIR


def test_fit_ff_to_hessian(tmp_workdir):
    """
    tests whether the update_bond function changes only the ff parameter
    """
    examples_dir = SMALL_MOLECULE_DIR
    shutil.copy2(examples_dir / "ff1.csv", tmp_workdir / "ff1.csv")
    shutil.copy2(examples_dir / "struc1.hess", tmp_workdir / "struc1.hess")

    path = "ff1.csv"
    path2hess = "struc1.hess"
    hessian = read_xtb_hessian(path2hess)

    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
    ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian)
    fill_ff(ff, info)

    result = fit_ff_to_hessian(
        Structure(StructurePath("d", "d", "d", "d"), ff, info),
        stepsize=0.05,
        threshold=0.001,
    )
    print(ff.bonds)
    print(ff.angles)
    print(ff.dihedrals)
    assert result["final_rmsd"] <= 0.1
    assert result["iterations"] <= 700


def test_fit_ff_to_hessian_with_repulsion(tmp_workdir):
    """
    tests whether the update_bond function changes only the ff parameter
    """
    examples_dir = SMALL_MOLECULE_DIR
    shutil.copy2(examples_dir / "ff1.csv", tmp_workdir / "ff1.csv")
    shutil.copy2(examples_dir / "struc1.hess", tmp_workdir / "struc1.hess")

    path = "ff1.csv"
    path2hess = "struc1.hess"
    hessian = read_xtb_hessian(path2hess)

    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
    ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian)
    fill_ff(ff, info)

    result = fit_ff_to_hessian(
        Structure(StructurePath("d", "d", "d", "d"), ff, info),
        constant_repulsion=False,
        stepsize=0.05,
        threshold=0.001,
    )
    print(ff.bonds)
    print(ff.angles)
    print(ff.dihedrals)
    print(ff.repulsive)
    # repulsive terms are all fitted to the same value but that could be due to only little repulsive forces in this molecule?
    assert result["final_rmsd"] <= 0.1
    assert result["iterations"] <= 700
