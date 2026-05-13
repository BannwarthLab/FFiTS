from ffits.ts_guess.guess import get_ts_guess
from ffits.forcefield.python_interface.optimization import (
    optimize_with_forcefield,
    optimize_with_forcefield_from_file,
)
from ffits.external.molbar import anc_optimizer
from ffits.forcefield.python_interface.ff_energy import (
    complete_gradient,
    complete_hessian,
)
from ffits.datatype.structure_data import ForceField, StructuralInformation, Structure, StructurePath
from ffits.datatype.calculation_data import TSCalculationOptions
from ffits.io.reader import read_wbo_file, read_xtb_hessian, readin_xyz
from ffits.ts_guess.define_starting_parameters import fill_ff
from ffits.forcefield.python_interface.ff_energy import ForceField, energy_ff
from ffits.utils.temp_dir_manager import TempDirManager
import numpy as np
import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def test_molecule_data1():
    """Load test molecule data (small_single_molecule example)."""
    path1 = os.path.join(os.getcwd(), "tests/examples/small_single_molecule")
    hesspath = os.path.join(path1, "struc1.hess")
    wbopath = os.path.join(path1, "wbo1")
    xyzpath = os.path.join(path1, "struc1.xyz")
    ffpath = os.path.join(path1, "ff1.csv")
    path = StructurePath(xyzpath, hesspath, wbopath, ffpath)
    
    nat, _, xyz, atom_types = readin_xyz(xyzpath)
    wbo = read_wbo_file(wbopath)
    hessian = read_xtb_hessian(hesspath)

    info = StructuralInformation(nat, xyz, wbo, atom_types, hessian)
    ff = ForceField(
        nat,
        ffpath,
        readff=False,
        energy_calculator=energy_ff,
        gradient_calculator=complete_gradient,
        hessian_calculator=complete_hessian,
    )
    fill_ff(ff, info, repulsive_start=0.0)
    struc = Structure(path, ff, info)

    return {"struc": struc, "info": info, "ff": ff, "nat": nat}

@pytest.fixture
def test_molecule_data2():
    """Load test molecule data (small_single_molecule example)."""
    path1 = os.path.join(os.getcwd(), "tests/examples/small_single_molecule")
    hesspath = os.path.join(path1, "struc2.hess")
    wbopath = os.path.join(path1, "wbo2")
    xyzpath = os.path.join(path1, "struc2.xyz")
    ffpath = os.path.join(path1, "ff2.csv")
    path = StructurePath(xyzpath, hesspath, wbopath, ffpath)
    nat, _, xyz, atom_types = readin_xyz(xyzpath)
    wbo = read_wbo_file(wbopath)
    hessian = read_xtb_hessian(hesspath)

    info = StructuralInformation(nat, xyz, wbo, atom_types, hessian)
    ff = ForceField(
        nat,
        ffpath,
        readff=False,
        energy_calculator=energy_ff,
        gradient_calculator=complete_gradient,
        hessian_calculator=complete_hessian,
    )
    fill_ff(ff, info, repulsive_start=0.0)
    struc = Structure(path, ff, info)

    return {"struc": struc, "info": info, "ff": ff, "nat": nat}


def test_get_ts_guess_withoutcalcdata(test_molecule_data1, test_molecule_data2):
    """Test that get_ts_guess runs without errors and returns a Structure object with the expected attributes."""
    struc1: Structure = test_molecule_data1["struc"]
    struc2: Structure = test_molecule_data2["struc"]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cwd = os.getcwd()
        os.chdir(temp_path)
        # pipe all output files to temp_path

        struc1.ff.ff_filename = str(temp_path / "ff1.csv")
        struc2.ff.ff_filename = str(temp_path / "ff2.csv")
        tsff, converged, energy, final_geom = get_ts_guess(
            struc1,
            struc2
        )

        assert isinstance(tsff, ForceField), "get_ts_guess did not return a ForceField object."
        assert converged is True, "TS guess optimization did not converge."
        assert isinstance(energy, float), "get_ts_guess did not return a float for energy."
        assert isinstance(final_geom, np.ndarray), "get_ts_guess did not return a numpy array for final geometry."
        os.chdir(cwd)
