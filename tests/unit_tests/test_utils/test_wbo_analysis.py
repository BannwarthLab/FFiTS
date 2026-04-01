"""
Unit tests for WBO analysis module.

Tests compare_wbo_differences and related functions.
"""

import pytest
import numpy as np
from pathlib import Path
from ffits.io.reader import read_xtb_hessian, readin_xyz, read_wbo_file
from ffits.datatype.structure_data import (
    Structure,
    StructurePath,
    ForceField,
    StructuralInformation,
)
from ffits.utils.wbo_analysis import (
    get_wbo_matrix_difference,
)


@pytest.fixture
def reactant_structure():
    """Create a reactant structure for testing using real example files."""
    test_dir = (
        Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
    )

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
    ff = ForceField(nat=nat, ff_filename=ff_file, readff=True)
    info = StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo,
        atom_types=np.array(atom_types),
        hessian=hessian,
    )
    return Structure(path=path, ff=ff, info=info)


@pytest.fixture
def product_structure():
    """Create a product structure for testing using real example files."""
    test_dir = (
        Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
    )

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
    ff = ForceField(nat=nat, ff_filename=ff_file, readff=True)
    info = StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo,
        atom_types=np.array(atom_types),
        hessian=hessian,
    )
    return Structure(path=path, ff=ff, info=info)


class TestGetWboMatrixDifference:
    """Test get_wbo_matrix_difference function."""

    def test_matrix_shape(self, reactant_structure, product_structure):
        """Test that returned matrix has correct shape."""
        diff_matrix = get_wbo_matrix_difference(reactant_structure, product_structure)

        nat = reactant_structure.info.nat
        assert diff_matrix.shape == (nat, nat)

    def test_matrix_symmetry(self, reactant_structure, product_structure):
        """Test that difference matrix is symmetric."""
        diff_matrix = get_wbo_matrix_difference(reactant_structure, product_structure)

        np.testing.assert_array_almost_equal(diff_matrix, diff_matrix.T)

    def test_matrix_values(self, reactant_structure, product_structure):
        """Test that matrix values are correct."""
        diff_matrix = get_wbo_matrix_difference(reactant_structure, product_structure)

        # Get a bond from the structures to verify
        reactant_wbo = reactant_structure.info.wbo
        product_wbo = product_structure.info.wbo

        # Check a specific bond that exists in both
        for bond in reactant_wbo.keys():
            if bond in product_wbo:
                i, j = bond[0] - 1, bond[1] - 1  # Convert to 0-based
                expected_change = product_wbo[bond] - reactant_wbo[bond]
                np.testing.assert_almost_equal(diff_matrix[i, j], expected_change)
                break  # Just test one bond

    def test_error_on_atom_count_mismatch(self, reactant_structure, product_structure):
        """Test error on atom count mismatch."""
        product_structure.info.nat = product_structure.info.nat + 1

        with pytest.raises(ValueError, match="same number of atoms"):
            get_wbo_matrix_difference(reactant_structure, product_structure)
