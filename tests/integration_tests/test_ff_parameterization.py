import os
import numpy as np
import copy
import pandas as pd
import pytest
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
from ffits.datatype.structure_data import (
    ForceField,
    StructuralInformation,
    StructurePath,
    Structure,
)
from ffits.ts_guess.parameterize_ff import (
    update_bond,
    update_angle,
    update_dihedral,
    update_repulsive,
    fit_ff_to_hessian,
)
from ffits.io.reader import read_xtb_hessian
from ffits.forcefield.python_interface.ff_energy import complete_hessian
from ffits.ts_guess.define_starting_parameters import fill_ff

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


def test_fit_ff_to_hessian():
    """
    tests whether the update_bond function changes only the ff parameter
    """
    examples_dir = Path(os.getcwd()) / "tests" / "examples" / "small_single_molecule"

    with TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        shutil.copy2(examples_dir / "ff1.csv", temp_path / "ff1.csv")
        shutil.copy2(examples_dir / "struc1.hess", temp_path / "struc1.hess")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

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
        finally:
            os.chdir(original_cwd)


def test_fit_ff_to_hessian_with_repulsion():
    """
    tests whether the update_bond function changes only the ff parameter
    """
    examples_dir = Path(os.getcwd()) / "tests" / "examples" / "small_single_molecule"

    with TemporaryDirectory() as tmpdir:
        # Copy necessary files to temp directory
        temp_path = Path(tmpdir)
        shutil.copy2(examples_dir / "ff1.csv", temp_path / "ff1.csv")
        shutil.copy2(examples_dir / "struc1.hess", temp_path / "struc1.hess")

        # Work in temporary directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

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
        finally:
            os.chdir(original_cwd)


def test_update_bond():
    """
    tests whether the update_bond function changes only the ff parameter
    """
    examples_dir = Path(os.getcwd()) / "tests" / "examples" / "small_single_molecule"

    with TemporaryDirectory() as tmpdir:
        # Copy necessary files to temp directory
        temp_path = Path(tmpdir)
        shutil.copy2(examples_dir / "ff1.csv", temp_path / "ff1.csv")
        shutil.copy2(examples_dir / "struc1.hess", temp_path / "struc1.hess")

        # Work in temporary directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            path = "ff1.csv"
            path2hess = "struc1.hess"
            hessian = read_xtb_hessian(path2hess)

            info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
            ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian)
            fill_ff(ff, info)
            hessian_ff = ff.get_hessian(info.fortran_xyz)

            old_bonds = copy.deepcopy(ff.bonds)
            new_values = []
            for row in ff.bonds.itertuples():
                new_param = update_bond(row, info, hessian_ff, 1)
                new_values.append((row.Index, new_param))
            for idx, val in new_values:
                ff.bonds.at[idx, "parameter"] = val

            with pytest.raises(AssertionError):
                pd.testing.assert_series_equal(
                    ff.bonds["parameter"], old_bonds["parameter"]
                )
            pd.testing.assert_series_equal(
                ff.bonds["reference_value"], old_bonds["reference_value"]
            )
            pd.testing.assert_series_equal(ff.bonds["atoms"], old_bonds["atoms"])
        finally:
            os.chdir(original_cwd)


def test_update_angle():
    """
    tests whether the update_angle function changes only the ff parameter
    """
    examples_dir = Path(os.getcwd()) / "tests" / "examples" / "small_single_molecule"

    with TemporaryDirectory() as tmpdir:
        # Copy necessary files to temp directory
        temp_path = Path(tmpdir)
        shutil.copy2(examples_dir / "ff1.csv", temp_path / "ff1.csv")
        shutil.copy2(examples_dir / "struc1.hess", temp_path / "struc1.hess")

        # Work in temporary directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            path = "ff1.csv"
            path2hess = "struc1.hess"
            hessian = read_xtb_hessian(path2hess)

            info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
            ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian)
            fill_ff(ff, info)
            hessian_ff = ff.get_hessian(info.fortran_xyz)

            old_angles = copy.deepcopy(ff.angles)
            new_values = []
            for row in ff.angles.itertuples():
                new_param = update_angle(row, info, hessian_ff, 1)
                new_values.append((row.Index, new_param))
            for idx, val in new_values:
                ff.angles.at[idx, "parameter"] = val

            with pytest.raises(AssertionError):
                pd.testing.assert_series_equal(
                    ff.angles["parameter"], old_angles["parameter"]
                )
            pd.testing.assert_series_equal(
                ff.angles["reference_value"], old_angles["reference_value"]
            )
            pd.testing.assert_series_equal(ff.angles["atoms"], old_angles["atoms"])
        finally:
            os.chdir(original_cwd)


def test_update_dihedral():
    """
    tests whether the update_dihedral function changes only the ff parameter
    """
    examples_dir = Path(os.getcwd()) / "tests" / "examples" / "small_single_molecule"

    with TemporaryDirectory() as tmpdir:
        # Copy necessary files to temp directory
        temp_path = Path(tmpdir)
        shutil.copy2(examples_dir / "ff1.csv", temp_path / "ff1.csv")
        shutil.copy2(examples_dir / "struc1.hess", temp_path / "struc1.hess")

        # Work in temporary directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            path = "ff1.csv"
            path2hess = "struc1.hess"
            hessian = read_xtb_hessian(path2hess)

            info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
            ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian)
            fill_ff(ff, info)
            hessian_ff = ff.get_hessian(info.fortran_xyz)

            old_dihedrals = copy.deepcopy(ff.dihedrals)
            new_values = []
            for row in ff.dihedrals.itertuples():
                new_param = update_dihedral(row, info, hessian_ff, 1)
                new_values.append((row.Index, new_param))
            for idx, val in new_values:
                ff.dihedrals.at[idx, "parameter"] = val

            with pytest.raises(AssertionError):
                pd.testing.assert_series_equal(
                    ff.dihedrals["parameter"], old_dihedrals["parameter"]
                )
            pd.testing.assert_series_equal(
                ff.dihedrals["reference_value"], old_dihedrals["reference_value"]
            )
            pd.testing.assert_series_equal(
                ff.dihedrals["atoms"], old_dihedrals["atoms"]
            )
        finally:
            os.chdir(original_cwd)


def test_update_repulsive():
    """
    tests whether the update_repulsive function changes only the ff parameter
    """
    examples_dir = Path(os.getcwd()) / "tests" / "examples" / "small_single_molecule"

    with TemporaryDirectory() as tmpdir:
        # Copy necessary files to temp directory
        temp_path = Path(tmpdir)
        shutil.copy2(examples_dir / "ff1.csv", temp_path / "ff1.csv")
        shutil.copy2(examples_dir / "struc1.hess", temp_path / "struc1.hess")

        # Work in temporary directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            path = "ff1.csv"
            path2hess = "struc1.hess"
            hessian = read_xtb_hessian(path2hess)

            info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
            ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian)
            fill_ff(ff, info)
            hessian_ff = ff.get_hessian(info.fortran_xyz)

            old_repulsive = copy.deepcopy(ff.repulsive)
            new_values = []
            for row in ff.repulsive.itertuples():
                new_param = update_repulsive(row, info, hessian_ff, 1)
                new_values.append((row.Index, new_param))
            for idx, val in new_values:
                ff.repulsive.at[idx, "parameter"] = val

            with pytest.raises(AssertionError):
                pd.testing.assert_series_equal(
                    ff.repulsive["parameter"], old_repulsive["parameter"]
                )
            pd.testing.assert_series_equal(
                ff.repulsive["reference_value"], old_repulsive["reference_value"]
            )
            pd.testing.assert_series_equal(
                ff.repulsive["atoms"], old_repulsive["atoms"]
            )
        finally:
            os.chdir(original_cwd)
